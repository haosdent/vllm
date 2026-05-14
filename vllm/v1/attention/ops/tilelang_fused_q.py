# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang fused Q kernel for the GLM-5.1 / DSv3.2 indexer Q-side path.

Replaces the chain `rotary_emb -> cat -> per_token_group_quant_fp8 ->
weights * q_scale * weight_scale_const` in `Indexer.forward` with one
kernel. See TILELANG_FINDINGS.md for the bench data + design notes.
"""

import functools

import torch

from vllm.v1.attention.ops.tilelang_utils import (
    ensure_fp32_cos_sin as _ensure_fp32_cos_sin,
)
from vllm.v1.attention.ops.tilelang_utils import (
    get_buffer as _get_buffer,
)

try:
    import tilelang
    from tilelang import language as T

    TILELANG_AVAILABLE = True
except ImportError:
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


_FP8_MAX = 448.0
_EPS = 1.0e-10


def _build_kernel(
    n_head: int,
    head_dim: int,
    rope_dim: int,
    max_pos: int,
    weight_scale_const: float,
    k_heads: int = 4,
    threads: int = 32,
):
    """Build the fused-Q TileLang kernel.

    Each CTA handles 1 token × k_heads heads. cos/sin loaded once per CTA
    and reused across the head group. NeoX-style RoPE applied to the
    first `rope_dim` elements of each head's `head_dim`-element vector.
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")

    N = T.dynamic("num_tokens")
    H = n_head
    D = head_dim
    ROPE = rope_dim
    HALF = rope_dim // 2

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _make():
        @T.prim_func
        def kernel(
            Q: T.Tensor([N, H, D], T.bfloat16),  # type: ignore[valid-type]
            W: T.Tensor([N, H], T.bfloat16),  # type: ignore[valid-type]
            Positions: T.Tensor([N], T.int64),  # type: ignore[valid-type]
            # cos_sin_cache row layout: [cos (rope_dim/2), sin (rope_dim/2)]
            CosSin: T.Tensor([max_pos, ROPE], T.float32),  # type: ignore[valid-type]
            Q_fp8: T.Tensor([N, H, D], T.float8_e4m3fn),  # type: ignore[valid-type]
            Q_scale: T.Tensor([N, H], T.float32),  # type: ignore[valid-type]
            W_scaled: T.Tensor([N, H], T.float32),  # type: ignore[valid-type]
        ):
            with T.Kernel(N, T.ceildiv(H, k_heads), threads=threads) as (n_i, hg_i):
                cos_sh = T.alloc_shared([HALF], T.float32)
                sin_sh = T.alloc_shared([HALF], T.float32)
                pos = T.alloc_var(T.int64)

                pos = Positions[n_i]
                for i in T.Parallel(HALF):
                    cos_sh[i] = CosSin[pos, i]
                    sin_sh[i] = CosSin[pos, HALF + i]

                for k in T.serial(k_heads):
                    h_i = hg_i * k_heads + k
                    if h_i < H:
                        # bf16 -> fp32 staged in shared (fits in registers
                        # would be cleaner; shared works fine at this size)
                        x_sh = T.alloc_shared([D], T.float32)
                        x_abs = T.alloc_fragment([D], T.float32)
                        absmax = T.alloc_fragment([1], T.float32)
                        scale = T.alloc_var(T.float32)

                        for i in T.Parallel(D):
                            x_sh[i] = T.Cast(T.float32, Q[n_i, h_i, i])

                        # NeoX RoPE on first ROPE elements. Split tmp into
                        # low/high halves because TileLang's parallel-access
                        # analyzer rejects two non-identical index reads
                        # of the same buffer in one iteration.
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

                        # Per-head per-token-group FP8 quant (group_size = head_dim).
                        # ue8m0: scale = 2^ceil(log2(max(absmax, eps) / fp8_max))
                        for i in T.Parallel(D):
                            x_abs[i] = T.abs(x_sh[i])
                        T.reduce_max(x_abs, absmax, dim=0, clear=True)
                        scale = T.exp2(
                            T.ceil(T.log2(T.max(absmax[0], _EPS) * (1.0 / _FP8_MAX)))
                        )

                        for i in T.Parallel(D):
                            v = T.alloc_var(T.float32)
                            v = x_sh[i] / scale
                            v = T.max(v, -_FP8_MAX)
                            v = T.min(v, _FP8_MAX)
                            Q_fp8[n_i, h_i, i] = T.Cast(T.float8_e4m3fn, v)

                        # Single thread per (token, head) writes scale & weights.
                        # W is bf16 input, scaled output is fp32 (matches the
                        # existing chain's `bf16 * fp32 * fp32 * fp32 -> fp32`).
                        if T.get_thread_binding() == 0:
                            Q_scale[n_i, h_i] = scale
                            W_scaled[n_i, h_i] = (
                                T.Cast(T.float32, W[n_i, h_i])
                                * scale
                                * weight_scale_const
                            )

        return kernel

    return _make()


@functools.lru_cache(maxsize=16)
def _cached_kernel(
    n_head: int,
    head_dim: int,
    rope_dim: int,
    max_pos: int,
    weight_scale_const: float,
):
    return _build_kernel(
        n_head=n_head,
        head_dim=head_dim,
        rope_dim=rope_dim,
        max_pos=max_pos,
        weight_scale_const=weight_scale_const,
    )


def tilelang_fused_indexer_q(
    q: torch.Tensor,
    weights_raw: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight_scale_const: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Indexer Q-side pipeline.

    Args:
        q:                  [num_tokens, n_head, head_dim] bf16 (post-`wq_b`, pre-RoPE)
        weights_raw:        [num_tokens, n_head] bf16 (post-`wk_weights_proj` split)
        positions:          [num_tokens] int64
        cos_sin_cache:      [max_pos, rope_dim] float32 (cos then sin halves)
        weight_scale_const: pre-folded `softmax_scale * n_head**-0.5`

    Returns:
        q_fp8:    [num_tokens, n_head, head_dim] fp8_e4m3fn (RoPE'd + quantized)
        q_scale:  [num_tokens, n_head] float32 (ue8m0 per-head scale)
        weights:  [num_tokens, n_head] float32
                  (= weights_raw * q_scale * weight_scale_const)
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")
    N, H, D = q.shape
    if __debug__:
        assert q.dtype == torch.bfloat16
        assert weights_raw.dtype == torch.bfloat16
        assert weights_raw.shape == (N, H)
    rope_dim = cos_sin_cache.shape[1]
    max_pos = cos_sin_cache.shape[0]

    # Indexer's `weights = kw[:, head_dim:]` is a column slice → its
    # stride[0] is `head_dim + n_head` rather than `n_head`. TileLang's
    # JIT-compiled kernel rejects non-default strides; force contiguous.
    # The clone is ~N*H*2 bytes (256 KB at N=2048,H=64) — negligible vs
    # the ~85 µs/layer the fusion saves elsewhere.
    if not weights_raw.is_contiguous():
        weights_raw = weights_raw.contiguous()
    # q can also be a non-contiguous view (`.view(-1, n_head, head_dim)`
    # is contiguous when the source is contiguous, but caller might pass
    # a slice). Guard the same way.
    if not q.is_contiguous():
        q = q.contiguous()

    # Kernel expects float32 cos_sin_cache for stable RoPE precision; cast
    # once and cache against the source cache identity so the conversion
    # cost amortizes across all token calls.
    cos_sin_fp32 = _ensure_fp32_cos_sin(cos_sin_cache)

    kernel = _cached_kernel(
        n_head=H,
        head_dim=D,
        rope_dim=rope_dim,
        max_pos=max_pos,
        weight_scale_const=weight_scale_const,
    )
    # Cache output buffers by shape+dtype+device so the same address is
    # reused across replays of the same cudagraph capture bucket. Without
    # this, `torch.empty` allocates fresh memory each call → during
    # cudagraph capture the alloc joins the graph pool, and on replay the
    # pool allocator may hand out memory that overlaps with another graph's
    # captured buffers, triggering illegal-access in downstream kernels.
    device = q.device
    q_fp8 = _get_buffer("q_fp8", (N, H, D), torch.float8_e4m3fn, device)
    q_scale = _get_buffer("q_scale", (N, H), torch.float32, device)
    weights = _get_buffer("weights_scaled", (N, H), torch.float32, device)
    kernel(q, weights_raw, positions, cos_sin_fp32, q_fp8, q_scale, weights)
    return q_fp8, q_scale, weights
