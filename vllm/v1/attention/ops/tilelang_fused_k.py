# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang fused Indexer-K kernel.

One kernel does: LayerNorm(k_norm) + NeoX RoPE on first rope_dim + ue8m0
FP8 quant + direct write to the indexer K cache. Replaces the chain
`k_norm -> split -> rotary_emb (K side) -> cat ->
ops.indexer_k_quant_and_cache` and forces the downstream
`sparse_attn_indexer` to skip its own cache write.

Cache layout matches `csrc/cache_kernels.cu:indexer_k_quant_and_cache_kernel`:
per block, `block_size * head_dim` FP8 bytes followed by `block_size * 4`
fp32 scale bytes. See TILELANG_FINDINGS.md for bench data; dequant
matches the existing CUDA op within 1 FP8 ULP.
"""

import functools

import torch

from vllm.v1.attention.ops.tilelang_utils import (
    ensure_fp32_cos_sin as _ensure_fp32_cos_sin,
)

try:
    import tilelang
    from tilelang import language as T

    TILELANG_AVAILABLE = True
except ImportError:
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


_LAYERNORM_EPS_DEFAULT = 1.0e-6
_FP8_MAX = 448.0
_QUANT_EPS = 1.0e-4  # matches csrc/cache_kernels.cu:indexer_k_quant_and_cache_kernel


def _build_kernel(
    head_dim: int,
    d_plus_h: int,
    rope_dim: int,
    max_pos: int,
    block_size: int,
    eps: float = _LAYERNORM_EPS_DEFAULT,
    threads: int = 64,
):
    """Build the fused Indexer-K-cache-write kernel.

    Each CTA handles one token. The kernel writes directly into the
    indexer K cache; no bf16 K tensor is produced as a side effect.
    Takes the full `wk_weights_proj` output tensor `KW [N, head_dim + n_head]`
    and reads only the first `head_dim` columns — this avoids forcing a
    contiguous copy on the column-sliced K at the caller.
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")

    N = T.dynamic("num_tokens")
    NB = T.dynamic("num_blocks")
    D = head_dim
    DPH = d_plus_h
    ROPE = rope_dim
    HALF = rope_dim // 2
    BS = block_size
    BYTES_PER_BLOCK = BS * (head_dim + 4)
    inv_dim = 1.0 / head_dim

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _make():
        @T.prim_func
        def kernel(
            KW: T.Tensor([N, DPH], T.bfloat16),  # type: ignore[valid-type]
            Positions: T.Tensor([N], T.int64),  # type: ignore[valid-type]
            CosSin: T.Tensor([max_pos, ROPE], T.float32),  # type: ignore[valid-type]
            Gamma: T.Tensor([D], T.float32),  # type: ignore[valid-type]
            Beta: T.Tensor([D], T.float32),  # type: ignore[valid-type]
            SlotMapping: T.Tensor([N], T.int64),  # type: ignore[valid-type]
            # KVCache is the indexer K cache flattened to 2D per-block:
            # [num_blocks, block_size * (head_dim + 4)] uint8.
            KVCache: T.Tensor([NB, BYTES_PER_BLOCK], T.uint8),  # type: ignore[valid-type]
        ):
            with T.Kernel(N, threads=threads) as n_i:
                slot = T.alloc_var(T.int64)
                slot = SlotMapping[n_i]
                # CUDA op skips when slot < 0 (padded token).
                if slot >= 0:
                    block_idx = T.alloc_var(T.int64)
                    block_offset = T.alloc_var(T.int64)
                    block_idx = slot // BS
                    block_offset = slot % BS

                    x_sh = T.alloc_shared([D], T.float32)
                    x_sum = T.alloc_fragment([D], T.float32)
                    x_sq = T.alloc_fragment([D], T.float32)
                    abs_buf = T.alloc_fragment([D], T.float32)
                    sum_val = T.alloc_fragment([1], T.float32)
                    sq_val = T.alloc_fragment([1], T.float32)
                    absmax_val = T.alloc_fragment([1], T.float32)
                    mean = T.alloc_var(T.float32)
                    inv_std = T.alloc_var(T.float32)
                    scale = T.alloc_var(T.float32)
                    cos_sh = T.alloc_shared([HALF], T.float32)
                    sin_sh = T.alloc_shared([HALF], T.float32)
                    pos = T.alloc_var(T.int64)

                    # 1) bf16 -> fp32 in shared
                    for i in T.Parallel(D):
                        x_sh[i] = T.Cast(T.float32, KW[n_i, i])

                    # 2) Mean = sum(x) / D
                    for i in T.Parallel(D):
                        x_sum[i] = x_sh[i]
                    T.reduce_sum(x_sum, sum_val, dim=0, clear=True)
                    mean = sum_val[0] * inv_dim

                    # 3) Variance = sum((x - mean)^2) / D
                    for i in T.Parallel(D):
                        diff = T.alloc_var(T.float32)
                        diff = x_sh[i] - mean
                        x_sq[i] = diff * diff
                    T.reduce_sum(x_sq, sq_val, dim=0, clear=True)
                    inv_std = 1.0 / T.sqrt(sq_val[0] * inv_dim + eps)

                    # 4) Normalize + affine
                    for i in T.Parallel(D):
                        x_sh[i] = (x_sh[i] - mean) * inv_std * Gamma[i] + Beta[i]

                    # 5) Load cos/sin once per token
                    pos = Positions[n_i]
                    for i in T.Parallel(HALF):
                        cos_sh[i] = CosSin[pos, i]
                        sin_sh[i] = CosSin[pos, HALF + i]

                    # 6) NeoX RoPE on first ROPE elements
                    t_lo = T.alloc_fragment([HALF], T.float32)
                    t_hi = T.alloc_fragment([HALF], T.float32)
                    for i in T.Parallel(HALF):
                        t_lo[i] = x_sh[i]
                    for i in T.Parallel(HALF):
                        t_hi[i] = x_sh[i + HALF]
                    for i in T.Parallel(HALF):
                        x_sh[i] = t_lo[i] * cos_sh[i] - t_hi[i] * sin_sh[i]
                    for i in T.Parallel(HALF):
                        x_sh[i + HALF] = t_hi[i] * cos_sh[i] + t_lo[i] * sin_sh[i]

                    # 7) absmax + ue8m0 scale (matches csrc: max(absmax, 1e-4)
                    # / fp8_max -> exp2(ceil(log2(...))) )
                    for i in T.Parallel(D):
                        abs_buf[i] = T.abs(x_sh[i])
                    T.reduce_max(abs_buf, absmax_val, dim=0, clear=True)
                    scale_raw = T.alloc_var(T.float32)
                    scale_raw = T.max(absmax_val[0], _QUANT_EPS) * (1.0 / _FP8_MAX)
                    scale = T.exp2(T.ceil(T.log2(scale_raw)))

                    # 8) Write FP8 bytes at per-block offset
                    # `block_offset * head_dim + d`.
                    fp8_byte_base = block_offset * D
                    for i in T.Parallel(D):
                        v = T.alloc_var(T.float32)
                        v = x_sh[i] / scale
                        v = T.max(v, -_FP8_MAX)
                        v = T.min(v, _FP8_MAX)
                        fp8_v = T.Cast(T.float8_e4m3fn, v)
                        KVCache[block_idx, fp8_byte_base + i] = T.reinterpret(
                            fp8_v, T.uint8
                        )

                    # 9) Write fp32 scale as 4 uint8 bytes at offset
                    # `block_size * head_dim + block_offset * 4`.
                    if T.get_thread_binding() == 0:
                        scale_u32 = T.alloc_var(T.uint32)
                        scale_u32 = T.reinterpret(scale, T.uint32)
                        scale_byte_base = BS * D + block_offset * 4
                        KVCache[block_idx, scale_byte_base + 0] = T.Cast(
                            T.uint8, scale_u32 & 0xFF
                        )
                        KVCache[block_idx, scale_byte_base + 1] = T.Cast(
                            T.uint8, (scale_u32 >> 8) & 0xFF
                        )
                        KVCache[block_idx, scale_byte_base + 2] = T.Cast(
                            T.uint8, (scale_u32 >> 16) & 0xFF
                        )
                        KVCache[block_idx, scale_byte_base + 3] = T.Cast(
                            T.uint8, (scale_u32 >> 24) & 0xFF
                        )

        return kernel

    return _make()


@functools.lru_cache(maxsize=16)
def _cached_kernel(
    head_dim: int,
    d_plus_h: int,
    rope_dim: int,
    max_pos: int,
    block_size: int,
    eps: float,
):
    return _build_kernel(
        head_dim=head_dim,
        d_plus_h=d_plus_h,
        rope_dim=rope_dim,
        max_pos=max_pos,
        block_size=block_size,
        eps=eps,
    )


def tilelang_fused_indexer_k_cache(
    kw: torch.Tensor,
    head_dim: int,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float = _LAYERNORM_EPS_DEFAULT,
) -> None:
    """Fused LayerNorm + NeoX RoPE + FP8 (ue8m0) quant + indexer K cache write.

    Mutates `kv_cache` in place. No return value (matches the existing
    `ops.indexer_k_quant_and_cache` contract).

    Args:
        kw:            [num_tokens, head_dim + n_head] bf16, the full
            `wk_weights_proj` output. The kernel reads only the first
            `head_dim` columns; this avoids forcing a contiguous copy on a
            column-sliced K (saves ~512 KB/layer at prefill).
        head_dim:      number of K columns to read from `kw` (the rest are
            ignored — they hold the indexer's `weights` slice).
        positions:     [num_tokens] int64
        cos_sin_cache: [max_pos, rope_dim] float32 — cos|sin concatenated
        gamma:         [head_dim] float32 — k_norm.weight
        beta:          [head_dim] float32 — k_norm.bias
        slot_mapping:  [num_tokens] int64 (CUDA op semantics: skip on -1)
        kv_cache:      [num_blocks, block_size, head_dim + 4] uint8 — indexer K cache
        eps:           LayerNorm epsilon
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")
    if __debug__:
        assert kw.dtype == torch.bfloat16
        assert kw.is_contiguous(), "kw must be contiguous"
    N, d_plus_h = kw.shape
    rope_dim = cos_sin_cache.shape[1]
    max_pos = cos_sin_cache.shape[0]

    # kv_cache is allocated as `[num_blocks, block_size, head_dim + 4]` uint8
    # by `DeepseekV32IndexerCache`. The actual byte layout per block is packed
    # (FP8 region then scale region), not row-major over the last two dims —
    # see the kernel docstring. We view it as a flat 2D `[num_blocks,
    # block_size * (head_dim + 4)]` so the kernel can index by byte.
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    cache_stride = kv_cache.shape[2]  # = head_dim + 4 for FP8 path
    assert cache_stride == head_dim + 4, (
        f"Unexpected cache stride {cache_stride}; expected head_dim+4 = "
        f"{head_dim + 4}. tilelang_fused_indexer_k_cache only supports the "
        "FP8 (non-FP4) cache layout for now."
    )
    kv_cache_2d = kv_cache.view(num_blocks, block_size * cache_stride)

    # k_norm weight/bias stored as fp32 in vllm.
    if gamma.dtype != torch.float32:
        gamma = gamma.to(torch.float32)
    if beta.dtype != torch.float32:
        beta = beta.to(torch.float32)

    cos_sin_fp32 = _ensure_fp32_cos_sin(cos_sin_cache)
    kernel = _cached_kernel(
        head_dim=head_dim,
        d_plus_h=d_plus_h,
        rope_dim=rope_dim,
        max_pos=max_pos,
        block_size=int(block_size),
        eps=float(eps),
    )
    kernel(kw, positions, cos_sin_fp32, gamma, beta, slot_mapping, kv_cache_2d)
