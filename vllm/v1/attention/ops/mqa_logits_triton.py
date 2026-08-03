# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback for DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits."""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut

_IS_SM80 = current_platform.is_cuda() and current_platform.get_device_capability() == (
    8,
    0,
)

# Paged decode: num_warps=4 dominated on A100/SM80 across {2,4,8}; the others
# were 1.5–1.7× slower at (num_heads=32, head_dim=128, block_size=64), so
# narrow the sweep to keep autotune from latching onto a bad pick under noise.
_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Prefill: BLOCK_N=128 with num_warps=4 measured fastest at every shape
# swept (M 1..2048, N 2048..131072) -- BN=64 is 1.25-1.40x worse, BN=32 up to
# 2.41x. The autotune key is (num_heads, head_dim), both fixed for a model,
# so a wider sweep cannot adapt per request; it only adds cold-cache JIT.
# On A100 at the DSv4 H=64, D=128 shape, maxnreg=128 raises residency from
# three to four CTAs/SM and measured 7.7-8.5% faster. Other fallback devices
# keep Triton's default register policy.
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_N": 128},
        num_warps=4,
        num_stages=ns,
        maxnreg=128 if _IS_SM80 else None,
    )
    for ns in (2, 4)
]

# Grouping eight KV tiles reuses each query and weight load. It is profitable
# only for large A100 prefill grids; smaller grids keep the original kernel.
_KV_GROUP = 8
_KV_GROUP_MIN_M = 512
_KV_GROUP_MIN_N = 16384

# Warmup shape mirrors the chunked-prefill regime (small M, long N) so
# autotune picks a tile sized for real serving rather than a launch-overhead-
# dominated dummy grid.
_PREFILL_WARMUP_M = 8
_PREFILL_WARMUP_N = 8192


# NaN bytes pin to +-480 so a NaN cache entry cannot poison the whole
# logits row through the dot product.
_INDEXER_LUT_NAN_VALUE = 480.0


def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    return get_e4m3fn_bf16_lut(device, nan_value=_INDEXER_LUT_NAN_VALUE)


@triton.jit
def _decode_e4m3fn_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_fp8_ptr,
    kv_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kvf_block,
    stride_kvf_s,
    stride_kvf_d,
    stride_kvs_block,
    stride_kvs_s,
    stride_w_t,
    stride_w_h,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_rk = tl.program_id(1)

    batch_id = token_id // next_n
    next_n_id = token_id % next_n

    context_len = tl.load(context_lens_ptr + batch_id)
    if block_rk * block_size >= context_len:
        return

    q_offset = context_len - next_n + next_n_id

    # int64: unified-KV-pool layer views carry a large block stride (~1e6
    # elements), so int32 `block_idx * stride` wraps once a batch touches
    # roughly >2k pool blocks (long context or concurrent sequences).
    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    ).to(tl.int64)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size

    q_base = q_ptr + batch_id * stride_q_b + next_n_id * stride_q_n
    q_byte = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0,
    )
    q = _decode_e4m3fn_bf16_lut(q_byte, fp8_lut_ptr)

    kvf_base = kv_fp8_ptr + block_idx * stride_kvf_block
    k_byte = tl.load(
        kvf_base + offs_n[:, None] * stride_kvf_s + offs_d[None, :] * stride_kvf_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    kvs_base = kv_scale_ptr + block_idx * stride_kvs_block
    k_scale = tl.load(
        kvs_base + offs_n * stride_kvs_s,
        mask=mask_n,
        other=0.0,
    )
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    # Scale in fp32 after the dot to avoid an extra bf16 round-trip on K.
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    k_offset = block_rk * block_size + offs_n
    # Store mask below covers mask_n and the context bound; -inf only has to
    # mask the causal tail inside the written region.
    out = tl.where(k_offset <= q_offset, out, float("-inf"))

    tl.store(
        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,
        out,
        mask=mask_n & (k_offset < context_len),
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_paged_mqa_logits.

    Args:
        q:             [B, next_n, H, D] fp8_e4m3fn
        kv_cache:      [num_blocks, block_size, 1, D+4] uint8 (FP8 + fp32 scale)
        weights:       [B*next_n, H] float32
        context_lens:  [B] int32
        block_tables:  [B, max_blocks] int32
        max_model_len: output width. Caller passes the active batch max so
            the logits buffer and grid stay tight.
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[:context_len]` per row).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4

    # Cache layout from `indexer_k_quant_and_cache`: per block, FP8 K bytes
    # (block_size * head_dim) followed by fp32 scales (block_size * 4). The
    # `[NB, block_size, 1, head_dim+4]` shape is a stride trick; re-slice flat.
    # Kernel decodes FP8 from uint8 via LUT (SM80 Triton can't load fp8e4nv).
    num_blocks = kv_cache.shape[0]
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    q_byte = q.view(torch.uint8)

    if clean_logits:
        logits = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    # The block table is allocated at full max_model_len width and only
    # narrowed along dim 0, so sizing the grid by it launches a CTA per
    # possible block rather than per live one; each surplus CTA loads
    # context_lens and returns. max_model_len here is the active batch max.
    num_block_cols = min(block_tables.shape[1], triton.cdiv(max_model_len, block_size))
    grid = (B * next_n, num_block_cols)
    _fp8_paged_mqa_logits_kernel[grid](
        q_byte,
        kv_byte,
        kv_scale,
        weights,
        fp8_lut,
        context_lens,
        block_tables,
        logits,
        q_byte.stride(0),
        q_byte.stride(1),
        q_byte.stride(2),
        q_byte.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    return logits


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    # Per-program work is N-independent; key on (heads, dim) only so chunked
    # prefill with varying N doesn't re-tune on every new chunk size.
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP: tl.constexpr,
):
    # bf16 q/k inputs: the wrapper pre-decodes FP8 → bf16. At compute-bound
    # prefill this is ~2× the in-kernel LUT (LUT lookups contend with the
    # matmul for ALU/regs). Paged-decode keeps the LUT path.
    m = tl.program_id(0)
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)

    if KV_GROUP == 1:
        # Keep the original single-tile program for every non-grouped shape.
        # Expressing it as a one-iteration static loop regresses that path.
        n_start = tl.program_id(1) * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        if (n_start >= ke) | (n_start + BLOCK_N <= ks):
            tl.store(
                logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=mask_n,
            )
            return

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        mask_h = offs_h < num_heads
        mask_d = offs_d < head_dim
        q = tl.load(
            q_ptr
            + m * stride_q_m
            + offs_h[:, None] * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0,
        )
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
        s = tl.dot(q, tl.trans(k)) * k_scale[None, :]
        w = tl.load(
            weights_ptr + m * stride_w_m + offs_h * stride_w_h,
            mask=mask_h,
            other=0.0,
        )
        s = tl.where(s > 0, s, 0.0) * w[:, None]
        out = tl.sum(s, axis=0)
        out = tl.where((offs_n >= ks) & (offs_n < ke), out, float("-inf"))
        tl.store(
            logits_ptr + m * stride_l_m + offs_n * stride_l_n,
            out,
            mask=mask_n,
        )
    else:
        # Each grouped CTA owns consecutive KV tiles for one query row.
        group_start = tl.program_id(1) * (KV_GROUP * BLOCK_N)
        if (group_start >= ke) | (group_start + KV_GROUP * BLOCK_N <= ks):
            offs_span = group_start + tl.arange(0, BLOCK_N)
            for _ in tl.static_range(KV_GROUP):
                tl.store(
                    logits_ptr + m * stride_l_m + offs_span * stride_l_n,
                    tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                    mask=offs_span < N,
                )
                offs_span += BLOCK_N
            return

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        mask_h = offs_h < num_heads
        mask_d = offs_d < head_dim
        q = tl.load(
            q_ptr
            + m * stride_q_m
            + offs_h[:, None] * stride_q_h
            + offs_d[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0,
        )
        w = tl.load(
            weights_ptr + m * stride_w_m + offs_h * stride_w_h,
            mask=mask_h,
            other=0.0,
        )

        # The per-tile branch prevents Triton from pipelining all K tiles and
        # exceeding the A100 register budget.
        for g in tl.static_range(KV_GROUP):
            n_start = group_start + g * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            if (n_start >= ke) | (n_start + BLOCK_N <= ks):
                tl.store(
                    logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                    tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                    mask=mask_n,
                )
            else:
                k = tl.load(
                    k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
                s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

                s = tl.where(s > 0, s, 0.0) * w[:, None]
                out = tl.sum(s, axis=0)
                out = tl.where((offs_n >= ks) & (offs_n < ke), out, float("-inf"))

                tl.store(
                    logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                    out,
                    mask=mask_n,
                )


def _select_prefill_kv_group(m: int, n: int) -> int:
    if _IS_SM80 and m >= _KV_GROUP_MIN_M and n >= _KV_GROUP_MIN_N:
        return _KV_GROUP
    return 1


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_mqa_logits.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D], k_scales [N]) — fp8_e4m3fn, float32
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[ks, ke)` per row). Matches DeepGEMM.
    Returns:
        logits:       [M, N] float32
    """
    return _fp8_mqa_logits_triton_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        _select_prefill_kv_group(q.shape[0], kv[0].shape[0]),
    )


def _fp8_mqa_logits_triton_impl(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    kv_group: int,
) -> torch.Tensor:
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    # The grid covers every (m, n_block) and each tile stores its full row
    # span, so a -inf pre-fill would be entirely overwritten; `clean_logits`
    # is accepted for DeepGEMM signature parity only.
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Pre-decode FP8 → bf16; the kernel runs a straight `tl.dot`.
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    # Grid depends on the autotuned BLOCK_N and the N-selected KV_GROUP.
    grid = lambda meta: (  # noqa: E731
        M,
        triton.cdiv(N, meta["BLOCK_N"] * meta["KV_GROUP"]),
    )
    _fp8_mqa_logits_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        KV_GROUP=kv_group,
    )
    return logits


@functools.lru_cache
def warmup_fp8_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Tune the ungrouped kernel, then compile the grouped SM80 specialization.

    The cache makes this model-construction warmup process-idempotent even
    though every sparse indexer layer requests it.
    """
    _warmup_fp8_mqa_logits_shape(
        num_heads, head_dim, _PREFILL_WARMUP_M, _PREFILL_WARMUP_N, device
    )
    if _IS_SM80:
        _warmup_fp8_mqa_logits_shape(
            num_heads,
            head_dim,
            _PREFILL_WARMUP_M,
            _KV_GROUP_MIN_N,
            device,
            kv_group_override=_KV_GROUP,
        )


def _warmup_fp8_mqa_logits_shape(
    num_heads: int,
    head_dim: int,
    m: int,
    n: int,
    device: torch.device,
    kv_group_override: int | None = None,
) -> None:
    q = torch.empty(m, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k = torch.empty(n, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    kv_group = (
        _select_prefill_kv_group(m, n)
        if kv_group_override is None
        else kv_group_override
    )
    _fp8_mqa_logits_triton_impl(
        q,
        (k, scales),
        weights,
        ks,
        ke,
        kv_group,
    )


def warmup_fp8_paged_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Prime the paged-decode `@triton.autotune` cache for the indexer's
    logits kernel (see `warmup_fp8_mqa_logits_triton` for rationale).
    """
    num_blocks = 2
    q = torch.empty(1, 1, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, head_dim + 4, dtype=torch.uint8, device=device
    )
    weights = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    context_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len=block_size
    )
