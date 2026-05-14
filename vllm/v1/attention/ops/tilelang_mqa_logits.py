# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang FP8 MQA logits kernels — prototype alternative to the Triton
kernels in `mqa_logits_triton.py`.

## SM80 compatibility

The upstream TileLang kernel (`fp8_lighting_indexer.py:mqa_attn_return_logits`)
uses `T.gemm` on `T.float8_e4m3fn` operands, which TileLang lowers to the
CUTLASS `SM89_16x8x32_F32E4M3E4M3F32_TN` intrinsic — only available on
sm_89+ (Ada/Hopper). On Ampere the kernel JITs but **traps at runtime**
on the first call with

    Assertion `0 && "Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN
    without CUTE_ARCH_MMA_F32_SM89_ENABLED"` failed.

We sidestep this by running the GEMM in bf16: `q.to(bf16)`, `k.to(bf16)`
in the Python wrapper before the kernel call, then the kernel runs a
standard bf16 `T.gemm` that lowers to `mma.sm80` (native on Ampere).
K-scales remain in fp32 and are still applied post-GEMM, matching
upstream's algorithm.

### Why pre-decode in Python and not in-kernel LUT?

Our Triton kernel uses an in-kernel 256-entry bf16 LUT: load uint8,
look up bf16, feed directly into `tl.dot`. We tried the analogous
TileLang pattern (uint8 staging in shared mem → LUT lookup in
`T.Parallel` → bf16 shared tile → `T.gemm`) and it was **1.4–1.8× slower
than pre-decode** on every shape we benched on A30.

The cause is architectural: `tl.dot` consumes operands from per-warp
fragments where the LUT lookup happens at fragment-fill time, fused
with the matmul issue. `T.gemm` requires bf16 already laid out in
shared memory, so the dequant pass *serializes* with the GEMM start
within each pipeline iteration. At our prefill shapes the GEMM is
compute-bound (the HBM save from in-kernel decode is irrelevant), so
adding work to the compute-bound critical path hurts more than the
memory savings help.

Pre-decode-in-Python costs an extra HBM round-trip (PyTorch's FP8→bf16
kernel writes a temporary bf16 tensor) but that kernel is bandwidth-bound
and overlaps cleanly with nothing on the timeline. PyTorch's converter
is well-tuned; the GLM-5.1 prefill chunk (M=2048, N=8192) sees
~21 µs of pre-decode out of ~1.7 ms total runtime — a 1.2% tax that
gives back 1.6 ms of fragment-vs-shared scheduling slack.

If a future port targets Hopper, the upstream FP8 GEMM works natively
and this decision is moot.

## Variants in this file

- `tilelang_fp8_mqa_logits(...)` — prefill. **Works on SM80 via bf16 wrapper-decode.**

The paged-decode variant is intentionally absent; the dispatch shim
keeps `fp8_paged_mqa_logits` on the Triton path. Porting requires a new
prim_func that reads paged KV via `block_tables` indirection, which
isn't a mechanical lift from upstream.
"""

import functools

import torch

from vllm.v1.attention.ops.tilelang_utils import (
    tilelang_num_stages as _arch_num_stages,
)

try:
    import tilelang
    from tilelang import language as T

    TILELANG_AVAILABLE = True
except ImportError:
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


def _build_prefill_kernel(
    heads: int,
    index_dim: int,
    block_N: int = 128,
    num_stages: int = 1,
    threads: int = 128,
    block_Q: int | None = None,
):
    """N-axis-tiled FP8 MQA logits prefill kernel, after upstream
    ``/root/tilelang/examples/deepseek_v32/inference/kernel.py:fp8_index_kernel``.

    The grid is 2D ``(ceildiv(M, block_Q), ceildiv(N, block_N_outer))`` so
    even when M is small (e.g. M=8 at chunked-prefill tail or spec-decode
    spans) the kernel still saturates the SMs by parallelising along the
    N axis. Per CTA: ``block_Q`` query rows × ``block_N_outer`` KV
    positions, with an inner pipeline over ``block_N_outer / block_N_inner``
    chunks so the K-tile load can overlap with the GEMM.

    Per-row [ks, ke) masking stays handled by the separate ``clean_logits``
    kernel post-pass; this kernel only needs the CTA-wide
    ``[cu_k_s_min, cu_k_e_max)`` range for early-exit on N-tiles that
    don't overlap any active row.

    Other deltas vs upstream:
    - ``dtype = bf16`` (not FP8). Upstream's ``T.gemm(float8)`` lowers to
      the sm_89-only intrinsic; we pre-decode FP8→bf16 in the Python
      wrapper and run a bf16 GEMM that fits ``mma.sm80``.
    - ``num_stages`` is arch-aware (1 on Ampere, 2 on Hopper+) via
      ``_arch_num_stages``.
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed in this environment")

    if block_Q is None:
        block_Q = max(1, 128 // heads)

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _make():
        @T.prim_func
        def kernel(
            IndexQ: T.Tensor([seq_len * heads, index_dim], dtype),  # type: ignore[valid-type]
            IndexK: T.Tensor([seq_len_kv, index_dim], dtype),  # type: ignore[valid-type]
            IndexKScale: T.Tensor([seq_len_kv], accum_dtype),  # type: ignore[valid-type]
            Logits: T.Tensor([seq_len, seq_len_kv], accum_dtype),  # type: ignore[valid-type]
            Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore[valid-type]
            CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore[valid-type]
            CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(
                T.ceildiv(seq_len, block_Q),
                T.ceildiv(seq_len_kv, block_N),
                threads=threads,
            ) as (bx_m, bx_n):
                index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
                index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
                index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
                s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
                s_reshaped = T.reshape(s, (block_N, block_Q, heads))
                logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
                weights = T.alloc_fragment([block_Q, heads], accum_dtype)

                seq_len_i = bx_m * block_Q
                k_start = bx_n * block_N

                cu_k_s_min = T.alloc_var(index_dtype)
                cu_k_e_max = T.alloc_var(index_dtype)
                cu_k_s_min = 2147483647
                cu_k_e_max = -2147483648
                for bq_i in T.serial(block_Q):
                    cu_k_s_min = T.min(
                        cu_k_s_min,
                        T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv),
                    )
                for bq_i in T.serial(block_Q):
                    cu_k_e_max = T.max(
                        cu_k_e_max,
                        T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv),
                    )

                # CTA-level early-exit: skip if this N tile doesn't overlap
                # the active K range for any row in our M tile. The
                # `clean_logits` post-pass writes -inf at masked positions
                # so leaving the output untouched here is safe.
                if (k_start + block_N > cu_k_s_min) and (k_start < cu_k_e_max):
                    T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
                    T.copy(Weights[seq_len_i, 0], weights)
                    T.copy(IndexK[k_start, 0], index_k_shared)
                    T.copy(IndexKScale[k_start], index_k_scale_fragment)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (
                            T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                        ) * index_k_scale_fragment[bn_i]

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for bq_i, bn_i in T.Parallel(block_Q, block_N):
                        Logits[
                            seq_len_i + bq_i,
                            k_start + bn_i,
                        ] = logits[bn_i, bq_i]

        return kernel

    return _make()


def _build_clean_logits_kernel(threads: int = 512, block_K: int = 4096):
    """Lift of `clean_logits_` from upstream."""
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    @tilelang.jit
    def _make():
        @T.prim_func
        def kernel(
            Logits: T.Tensor([seq_len, seq_len_kv], T.float),  # type: ignore[valid-type]
            CuSeqLenKS: T.Tensor([seq_len], T.int32),  # type: ignore[valid-type]
            CuSeqLenKE: T.Tensor([seq_len], T.int32),  # type: ignore[valid-type]
        ):
            with T.Kernel(seq_len, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                cu_k_s = CuSeqLenKS[bx]
                cu_k_e = CuSeqLenKE[bx]
                for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                    for k_i in T.serial(block_K // threads):
                        idx = n_i * block_K + k_i * threads + tx
                        if idx < cu_k_s or idx >= cu_k_e:
                            Logits[bx, idx] = -T.infinity(T.float)

        return kernel

    return _make()


@functools.lru_cache(maxsize=32)
def _cached_prefill(heads, index_dim, num_stages, block_Q):
    return _build_prefill_kernel(
        heads=heads,
        index_dim=index_dim,
        num_stages=num_stages,
        block_Q=block_Q,
    )


def _choose_block_Q(M: int, heads: int) -> int:
    """Pick the M-tile size to balance GEMM efficiency vs M parallelism.

    We use `block_Q = max(1, 128 // heads)` (= 8 for the GLM-5.1 / DSv3.2
    indexer heads=16). The N-axis grid already gives `N/block_N = 128`
    CTAs which saturates A100's 108 SMs by itself, so larger block_Q
    just amortises per-CTA overhead (mask compute, shared init, GEMM
    warmup) over more work.

    Sweep on A100 (M=64, heads=16, D=128, N=16384):

      block_Q  M-tiles  CTAs    waves     time
      1        64       8192    ~76       130.5 us
      8        8        1024    ~10        43.9 us  (3.0x faster)
      16       4        512     ~5        129.4 us  (shared mem too big)

    An earlier `if M <= 64: return 1` special case in this heuristic
    over-decomposed: each of the 8192 CTAs had so little work that
    per-CTA fixed overhead dominated the kernel.
    """
    return max(1, 128 // heads)


@functools.lru_cache(maxsize=1)
def _cached_clean_logits():
    return _build_clean_logits_kernel()


def tilelang_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """TileLang prefill MQA logits, signature-compatible with
    `fp8_mqa_logits_triton`.

    FP8 Q/K are decoded to bf16 in this wrapper before invoking the kernel
    (TileLang's `T.gemm(float8)` traps on SM80). K-scales remain fp32 and
    are still applied post-GEMM, matching upstream's algorithm. See the
    module docstring for why we don't fuse the LUT decode into the kernel.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D], k_scales [N]) — fp8_e4m3fn, float32
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: if True, run a second kernel to write -inf at masked
            positions. Indexer top-k can skip this (matches Triton semantics).
    Returns:
        logits:       [M, N] float32
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed; cannot use TileLang backend")
    k_fp8, k_scales = kv
    M, heads, D = q.shape
    N = k_fp8.shape[0]
    assert q.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert k_scales.dtype == torch.float32

    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    kernel = _cached_prefill(
        heads, D, _arch_num_stages(q.device.index), _choose_block_Q(M, heads)
    )
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)
    kernel(
        q_bf16.reshape(M * heads, D),
        k_bf16,
        k_scales.reshape(-1),
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    if clean_logits:
        clean = _cached_clean_logits()
        clean(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits
