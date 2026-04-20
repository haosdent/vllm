# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementations of DeepGEMM's fp8_mqa_logits and
fp8_paged_mqa_logits for GPUs where DeepGEMM is not available.

The computation is:

    Q, K                := dequant(Q_fp8), dequant(K_fp8) * k_scales
    score[H, M, N]      = Q[M, H, D] @ K[N, D].T
    logits[M, N]        = (relu(score) * weights[M, H]).sum(axis=0)
    logits[M, N]       := -inf  outside of valid range

Q/K are cast to bf16 for the matmul; the matmul uses an fp32 accumulator.

K-side scale multiplication is done in fp32 before downcasting to bf16
so the per-row dequant scale is applied at full precision.
"""

import torch

from vllm.triton_utils import tl, triton

# Paged decode config sweep. `num_warps=4` dominated in A100/SM80 bench
# across {2,4,8}×{2,4}; the sub-optimal warps=2/8 picks were 1.5–1.7× slower
# at the autotune key shape (num_heads=32, head_dim=128, block_size=64), and
# autotune timing noise occasionally latched onto them. Keep only the two
# `num_warps=4` configs so that path is always selected.
_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Prefill kernel adds BLOCK_N as a free tile axis along the K dimension.
# Bench on A100/SM80 at (M=2048, N=8192, H=32, D=128) shows BN=32/64 with
# num_warps∈{2,4} is within a few % of the best, BN=128 wins for GLM-5.1
# long chunks, and num_warps=8 is consistently 1.5–3× worse. Sweep all three
# so autotune can pick per shape; warmup absorbs the one-time cost.
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (32, 64, 128)
    for nw in (2, 4)
    for ns in (2, 4)
]

# Warmup runs against a long-K, small-M shape to mirror the chunked-prefill
# regime that dominates real serving. Per-program work is independent of M,
# but a long N gives autotune a representative timing signal so it doesn't
# pick a tile sized for launch-overhead-dominated dummy grids.
_PREFILL_WARMUP_M = 8
_PREFILL_WARMUP_N = 8192


@triton.jit
def _decode_e4m3fn(u):
    """Decode an E4M3FN byte (uint8) to fp32 using only uint/int/fp ops.

    Triton on SM80 cannot compile `tl.float8e4nv`, so we never load the
    FP8 dtype directly — we load uint8 and decode in software here. The
    expansion is ~6 ops per element, dwarfed by the surrounding matmul.

    E4M3FN: 1 sign + 4 exp (bias 7) + 3 mantissa.  No infinities.
    Subnormal (exp=0): value = (-1)^s * (mant/8) * 2^(1 - 7)
    Normal           : value = (-1)^s * (1 + mant/8) * 2^(exp - 7)
    NaN at 0x7F/0xFF is decoded numerically as ±480 — sparse-MLA inputs
    never hit this so the loss of NaN propagation is acceptable.
    """
    # Compose the fp32 bit pattern directly so the SFU `exp2` from the
    # original float-arithmetic decode is removed from the per-element path.
    # fp32 layout: [1b sign | 8b exp (bias 127) | 23b fraction].
    u32 = u.to(tl.uint32)
    sign = (u32 & 0x80) << 24  # E4M3 sign bit (b7) → fp32 sign bit (b31).
    exp_bits = (u32 >> 3) & 0x0F
    mant = u32 & 0x07

    # Normal: real_exp = exp_bits - 7, fp32 exp field = real_exp + 127
    #                                                = exp_bits + 120.
    # 3-bit mantissa shifts left to occupy bits[22:20] of the fp32 fraction.
    normal = sign | ((exp_bits + 120) << 23) | (mant << 20)

    # Subnormal: value = (mant/8) * 2^-6. Renormalise to a fp32 normal by
    # shifting until the implicit leading 1 lines up:
    #   mant=1  → 1.000 * 2^-9   → fp32 exp 118 (=127-9), fraction 0
    #   mant=2  → 1.000 * 2^-8   → fp32 exp 119 (=127-8), fraction 0
    #   mant=3  → 1.500 * 2^-8   → fp32 exp 119,          fraction bit 22
    #   mant=4  → 1.000 * 2^-7   → fp32 exp 120 (=127-7), fraction 0
    #   mant=5  → 1.250 * 2^-7   → fp32 exp 120,          fraction bit 21
    #   mant=6  → 1.500 * 2^-7   → fp32 exp 120,          bits {22}
    #   mant=7  → 1.750 * 2^-7   → fp32 exp 120,          bits {22,21}
    sub_exp = tl.where(mant >= 4, 120, tl.where(mant >= 2, 119, 118))
    sub_frac = tl.where(
        mant >= 4,
        (mant - 4) << 21,
        tl.where(mant >= 2, (mant - 2) << 22, 0),
    )
    subnormal = sign | (sub_exp << 23) | sub_frac

    # Zero (exp=0, mant=0) keeps just the sign — preserves -0 unlike the
    # old float-arithmetic decode.
    zero = sign
    bits = tl.where(exp_bits == 0, tl.where(mant == 0, zero, subnormal), normal)
    return bits.to(tl.float32, bitcast=True)


_E4M3FN_BF16_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    lut = _E4M3FN_BF16_LUT_CACHE.get(device)
    if lut is not None:
        return lut
    lut = (
        torch.arange(256, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    lut[0x7F] = 480.0
    lut[0xFF] = -480.0
    _E4M3FN_BF16_LUT_CACHE[device] = lut
    return lut


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

    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    )

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
    # Apply per-row K scale to the fp32 dot output rather than pre-scaling
    # the bf16 K tile. Saves a per-element fp32 mul + bf16 downcast and
    # keeps the scale at fp32 instead of losing precision in the round-trip.
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    k_offset = block_rk * block_size + offs_n
    valid = mask_n & (k_offset < context_len) & (k_offset <= q_offset)
    out = tl.where(valid, out, float("-inf"))

    tl.store(
        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,
        out,
        mask=mask_n,
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
        max_model_len: caller-controlled output width. The indexer passes the
            active batch's max seq length to keep the logits buffer and grid
            tight rather than the configured model max.
        clean_logits: pre-fill output with -inf so positions past `context_len`
            read as -inf. Indexer top-k consumes only `[:context_len]` per row,
            so it can set this False to skip the fill (matches DeepGEMM).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4

    # Cache layout: `indexer_k_quant_and_cache` (csrc/cache_kernels.cu) writes
    # each block as [K region | scale region] — all `block_size * head_dim`
    # fp8 K bytes first, then `block_size * 4` fp32 scale bytes. The
    # `[NB, block_size, 1, head_dim+4]` shape is just a stride trick; bytes
    # must be re-sliced flat. The kernel decodes FP8 from uint8 manually since
    # SM80 Triton can't compile `tl.float8e4nv`.
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
    grid = (B * next_n, block_tables.shape[1])
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
    # Per-program work doesn't depend on N — only the grid extent does — so
    # a single autotune config is valid across seq lengths. Keeping N in the
    # key used to re-tune from scratch on every new chunk size (e.g., 2048,
    # 4096, 6144, 8192, 9993 for a 10K prompt with chunked prefill),
    # producing ~2 minutes of first-call TTFT on top of the real work.
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
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
):
    m = tl.program_id(0)
    n_block = tl.program_id(1)

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    # Early-exit when this row's valid `[ks, ke)` range doesn't overlap with
    # this K-tile. Chunked prefill produces narrow per-row ranges (causal mask
    # within the chunk), so a meaningful fraction of CTAs would otherwise run
    # to completion with all-masked work.
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    # Bitwise on scalar tl.int1 — Triton has no short-circuit `or` at the
    # kernel level; both sides are pure scalar loads, so eager eval is free.
    if (n_start >= ke) | (n_start + BLOCK_N <= ks):
        # Wrappers may set `clean_logits=False` and skip the `-inf` pre-fill.
        # Downstream top-k reads the full row, so we must write `-inf` here
        # ourselves; the non-early-exit path's `tl.where(valid, out, -inf)`
        # below handles partially-masked tiles.
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

    q_byte = tl.load(
        q_ptr
        + m * stride_q_m
        + offs_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0,
    )
    q = _decode_e4m3fn_bf16_lut(q_byte, fp8_lut_ptr)

    k_byte = tl.load(
        k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    # Apply per-row K scale to the fp32 dot output rather than pre-scaling
    # the bf16 K tile. Saves a per-element fp32 mul + bf16 downcast and
    # keeps the scale at fp32 instead of losing precision in the round-trip.
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    valid = mask_n & (offs_n >= ks) & (offs_n < ke)
    out = tl.where(valid, out, float("-inf"))

    tl.store(
        logits_ptr + m * stride_l_m + offs_n * stride_l_n,
        out,
        mask=mask_n,
    )


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
        clean_logits: pre-fill output with -inf so masked positions read as
            -inf. Indexer top-k consumes only `[ks, ke)` per row, so it can
            set this False to skip the fill (matches DeepGEMM behavior).
    Returns:
        logits:       [M, N] float32
    """
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    if clean_logits:
        logits = torch.full((M, N), float("-inf"), dtype=torch.float32, device=q.device)
    else:
        logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Pass FP8 tensors as uint8 — kernel decodes E4M3FN bytes manually so it
    # works on SM80 where Triton can't compile the native fp8e4nv dtype.
    q_byte = q.view(torch.uint8)
    k_byte = k_fp8.view(torch.uint8)
    fp8_lut = _get_e4m3fn_bf16_lut(q.device)

    # Grid depends on the autotuned BLOCK_N.
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _fp8_mqa_logits_kernel[grid](
        q_byte,
        k_byte,
        k_scales,
        weights,
        fp8_lut,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_byte.stride(0),
        q_byte.stride(1),
        q_byte.stride(2),
        k_byte.stride(0),
        k_byte.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    return logits


def warmup_fp8_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Prime the prefill `@triton.autotune` cache for the indexer's logits
    kernel. Runs one shape matching the autotune key so that the first real
    request does not pay the inline sweep + JIT cost (~5–8 s on A100 SM80).

    N is a runtime scalar in the kernel (not a `tl.constexpr`), so one warmup
    shape compiles all prefill lengths. Use a small-M, long-N shape: per-program
    work is independent of M, but the long N gives autotune a representative
    timing signal so it picks the tile that wins for real chunked-prefill grids
    instead of a launch-overhead-dominated dummy grid.
    """
    max_block_n = max(c.kwargs["BLOCK_N"] for c in _PREFILL_AUTOTUNE_CONFIGS)
    m = _PREFILL_WARMUP_M
    n = max(_PREFILL_WARMUP_N, max_block_n)
    q = torch.empty(m, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k = torch.empty(n, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke)


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
