# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse MLA attention kernel — prototype alternative to the
Triton kernel in `triton_sparse_mla_kernel.py`. Signature matches
`triton_sparse_mla_attention(q, kv, indices, sm_scale, ...)` so the bench
script can swap them at the call site.

Two-path layout mirroring the Triton kernel:

* **Single-pass** (`_build_kernel`): grid is `(seq_len*REPLICATE_H, batch,
  kv_group)` — one CTA per (token, kv_group). Used for prefill or
  batch-saturating decode where the device is already busy.
* **Split-KV** (`_build_split_kernel` + `_build_merge_kernel`): adds a
  `num_split` grid dimension that partitions the topk axis. Stage 1 writes
  a per-split fp32 `(out/e_sum, lse_log2)` tile; stage 2 runs an
  online-softmax merge across splits. Required for num_tokens=1 decode
  where the single-pass grid is `(1,1,1)` and the device is starved
  (the regression `TILELANG_FINDINGS.md` documented at 4.7× slower than
  Triton). `_choose_num_kv_splits` mirrors the Triton path.

The single-pass kernel body is a decode-adapted fork of
`/root/tilelang/examples/deepseek_v32/sparse_mla_fwd.py:sparse_mla_fwd`
with three deltas:

1. The mask check is `Indices >= 0` (filter padded-with-`-1` slots from the
   indexer) instead of the prefill kernel's causal `Indices <= max_kv_i`.
   The indexer-decode path produces strictly-valid positions or `-1` padding.
2. `num_stages` is auto-selected: 1 on sm_80/86/89 (Ampere ceiling is 164 KB
   dynamic shared per block — the upstream default 2 overflows), 2 on
   sm_90+. See `examples/deepseek_mla/AMPERE_HANDOFF.md`.
3. The wrapper reshapes vLLM's 3D tensors `[num_tokens, heads, dim]` to
   TileLang's 4D `[batch=1, seq_len=num_tokens, heads, dim]`.

The split-KV kernels reuse the same inner-loop structure as single-pass and
follow `examples/deepseek_mla/example_mla_decode_paged.py:main_split` /
`combine` for the LSE/partial layout. Stage-1 guards `sumexp==0` (all-padding
splits) and writes a finite -1e30 LSE sentinel so the merge can underflow
the empty split to zero rather than NaN-poisoning the result.

LSE return is plumbed through but ignored by the wrapper — the existing
`triton_sparse_mla_attention` doesn't expose it. We compile-cache by
`(heads, topk, num_stages, sm_scale)` for single-pass and additionally
by `num_split` for the split path.
"""

import functools

import torch

from vllm.v1.attention.ops.tilelang_utils import (
    get_buffer as _get_buffer,
)
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


_DIM_QK = 576  # 512 nope + 64 rope
_DIM_DV = 512
_DIM_DPE = 64

# Split-KV heuristic — mirrors `triton_sparse_mla_kernel._choose_num_kv_splits`
# so the TileLang path picks the same split count as Triton for the same shape.
_MIN_TOPK_PER_SPLIT = 128
_SPLIT_MAX_OCCUPANCY = 4
_KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)
# Split kernel uses BI=64 (same as single-pass). The number of BI-blocks per
# split must be ≥ 1 and topk must divide (BI * num_split) cleanly.
_SPLIT_BLOCK_I = 64
# Merge kernel DV tile: 128 lanes per CTA so each (token, head) becomes 4
# programs at D=512. Matches Triton's `_MERGE_BLOCK_DV_TILE`.
_MERGE_BLOCK_DV_TILE = 128


def _build_kernel(
    heads: int,
    topk: int,
    num_stages: int,
    sm_scale: float,
    kv_group: int = 1,
    block_I: int = 64,
    threads: int = 128,
):
    """Build a TileLang sparse-MLA-decode kernel. Cached by signature.

    `threads=128` chosen over upstream's 256 (sweep on A100; gains 1.1–1.5x
    across decode + prefill shapes). See TILELANG_FINDINGS.md.
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed in this environment")

    dim = _DIM_DV
    tail_dim = _DIM_DPE
    # log2(e) factor folded into sm_scale so the kernel can use exp2 internally.
    sm_scale_log2 = sm_scale * 1.44269504
    assert topk % block_I == 0, (
        f"topk ({topk}) must be a multiple of block_I ({block_I}); the "
        "tile loop expects a clean number of iterations."
    )

    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, (
            "kv_group != 1 with non-pow2 heads needs explicit head masking"
        )

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]

    # No `out_idx`: caller passes Output and Lse buffers explicitly so the
    # tvm_ffi adapter doesn't `torch.empty` per call inside cudagraph
    # capture. See TILELANG_FINDINGS.md.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _make():
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            KV: T.Tensor(kv_shape, dtype),  # type: ignore[valid-type]
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(o_shape, dtype),  # type: ignore[valid-type]
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
                bx,
                by,
                bz,
            ):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                O_shared = T.alloc_shared([H_per_block, D], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_i, g_i = by, bz
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                )
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                # DELTA #4 vs upstream: per-tile CTA early-exit. The
                # indexer pads short queries with trailing `-1`. The
                # naive kernel did the BI=64 KV gather even when every
                # index was `-1` (mapped to KV[0] via `T.max(idx, 0)`),
                # which dominated cold-prefill TTFT (4.6x slower at
                # pad_frac=0.99). Triton's `tl.load(mask=...)` skips
                # those loads via predicated memory ops; TileLang's
                # `T.if_then_else` does not lower to a predicated load.
                # Workaround: count valid mask bits in a serial reduce
                # and guard the whole tile body with a CTA-wide `if`.
                # Mirrors the early-exit pattern in
                # tilelang_mqa_logits.py:_build_prefill_kernel.
                # First-valid scalar: if `Indices[i_i*BI] >= 0` for a single
                # representative slot in the tile we proceed with the full
                # body. This costs one extra index load per tile but avoids
                # the vectorized-bool reduce path that TileLang's CUDA
                # codegen rejects (`Cannot convert type boolx8 to CUDA
                # type`). Correct for the production padding-at-end pattern
                # the indexer emits: trailing tiles are fully `-1`.
                first_idx_var = T.alloc_var("int32")
                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0
                    first_idx_var = Indices[b_i, s_i, g_i, i_i * BI]
                    if first_idx_var >= 0:
                        for bi_i, d_i in T.Parallel(BI, D):
                            KV_shared[bi_i, d_i] = KV[
                                b_i,
                                T.max(Indices[b_i, s_i, g_i, i_i * BI + bi_i], 0),
                                g_i,
                                d_i,
                            ]
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            K_tail_shared[bi_i, d_i] = KV[
                                b_i,
                                T.max(Indices[b_i, s_i, g_i, i_i * BI + bi_i], 0),
                                g_i,
                                D + d_i,
                            ]

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2
                            )
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale_log2
                                - m_i[h_i] * sm_scale_log2
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D):
                            acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared,
                            KV_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                # Guard divide-by-zero: with the per-tile early-exit, a
                # query whose entire topk is `-1` ends the loop with
                # sumexp[h] == 0, so 0/0 → NaN. Force the row to zero in
                # that case (matches the split-merge sentinel behaviour).
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        acc_o[h_i, d_i] / sumexp[h_i],
                        T.Cast(accum_dtype, 0),
                    )
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale_log2,
                        T.Cast(accum_dtype, -1.0e30),
                    )

                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[b_i, s_i, H0:H1, :])
                T.copy(sumexp, Lse[b_i, s_i, H0:H1])

        return main

    return _make()


@functools.lru_cache(maxsize=64)
def _cached_kernel(heads, topk, num_stages, sm_scale):
    return _build_kernel(
        heads=heads, topk=topk, num_stages=num_stages, sm_scale=sm_scale
    )


def _build_split_kernel(
    heads: int,
    topk: int,
    num_split: int,
    num_stages: int,
    sm_scale: float,
    kv_group: int = 1,
    block_I: int = _SPLIT_BLOCK_I,
    threads: int = 128,
):
    """Stage-1 kernel of the split-KV decode pipeline.

    Iterates `[split_start, split_end)` of the topk axis (one `num_split`-th of
    the BI-blocks) and writes per-split partial output (`acc/sumexp`, fp32) and
    LSE (log2 domain, fp32). Mirrors `triton_sparse_mla_kernel._sparse_mla_kernel_split`
    in semantics and `examples/deepseek_mla/example_mla_decode_paged.py:main_split`
    in shape — adapted for the `Indices >= 0`-mask indexed gather.
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed in this environment")
    if kv_group != 1:
        raise ValueError("split kernel currently assumes kv_group=1")

    dim = _DIM_DV
    tail_dim = _DIM_DPE
    sm_scale_log2 = sm_scale * 1.44269504
    # Finite sentinel for the LSE of a split that saw no valid topk entries —
    # `log2(0)+anything = -inf` would NaN-poison the stage-2 merge. -1e30
    # underflows `exp2()` to 0, so the merge weights this split out cleanly.
    neg_large_lse = -1.0e30

    BI = block_I
    assert topk % BI == 0, f"topk ({topk}) must be a multiple of block_I ({BI})"
    NI = topk // BI
    assert NI % num_split == 0, (
        f"NI ({NI}) must divide num_split ({num_split}); pick a divisor."
    )
    NI_PER_SPLIT = NI // num_split

    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1

    D = dim
    D_tail = tail_dim
    if head_kv > 64:
        assert head_kv % 64 == 0
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    partial_shape = [batch, seq_len, heads, num_split, dim]
    lse_shape = [batch, seq_len, heads, num_split]

    # No `out_idx`: caller passes Partial and Lse buffers (see single-pass
    # decorator's doc comment for the cudagraph-overhead rationale).
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _make():
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[valid-type]
            KV: T.Tensor(kv_shape, dtype),  # type: ignore[valid-type]
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore[valid-type]
            Partial: T.Tensor(partial_shape, accum_dtype),  # type: ignore[valid-type]
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(seq_len * REPLICATE_H, batch, num_split, threads=threads) as (
                bx,
                by,
                bz_split,
            ):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_i = by
                g_i = 0
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                )
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                split_start = bz_split * NI_PER_SPLIT

                # See DELTA #4 in `_build_kernel`: same per-tile early-exit
                # to skip fully-padded BI blocks during cold prefill.
                first_idx_var = T.alloc_var("int32")
                for i_i_local in T.Pipelined(NI_PER_SPLIT, num_stages=num_stages):
                    i_i = split_start + i_i_local
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0
                    first_idx_var = Indices[b_i, s_i, g_i, i_i * BI]
                    if first_idx_var >= 0:
                        for bi_i, d_i in T.Parallel(BI, D):
                            KV_shared[bi_i, d_i] = KV[
                                b_i,
                                T.max(Indices[b_i, s_i, g_i, i_i * BI + bi_i], 0),
                                g_i,
                                d_i,
                            ]
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            K_tail_shared[bi_i, d_i] = KV[
                                b_i,
                                T.max(Indices[b_i, s_i, g_i, i_i * BI + bi_i], 0),
                                g_i,
                                D + d_i,
                            ]

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2
                            )
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale_log2
                                - m_i[h_i] * sm_scale_log2
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D):
                            acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared,
                            KV_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                # Divide the partial output by its local sumexp (so the merge
                # only needs to rescale, not divide). Guard sumexp==0 (an
                # all-padding split) to keep zero rather than NaN.
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        acc_o[h_i, d_i] / sumexp[h_i],
                        T.Cast(accum_dtype, 0),
                    )
                # Final LSE in log2 domain: log2(sumexp) + m_i*sm_scale_log2.
                # Use the finite sentinel for empty splits.
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale_log2,
                        T.Cast(accum_dtype, neg_large_lse),
                    )

                # Write partial output (fp32) and LSE (fp32). Direct
                # fragment→global writes here are fine: each thread covers a
                # contiguous slice of the inner dim. Partial layout is
                # [batch, seq_len, heads, num_split, D] so the (h, dim) lanes
                # are inner-fast for a fixed split.
                for h_i, d_i in T.Parallel(H_per_block, D):
                    Partial[b_i, s_i, H0 + h_i, bz_split, d_i] = acc_o[h_i, d_i]
                for h_i in T.Parallel(H_per_block):
                    Lse[b_i, s_i, H0 + h_i, bz_split] = sumexp[h_i]

        return main

    return _make()


def _build_merge_kernel(
    heads: int,
    num_split: int,
    threads: int = 128,
    block_dv_tile: int = _MERGE_BLOCK_DV_TILE,
):
    """Stage-2 kernel of split-KV decode.

    N-way online-softmax merge of per-split partial outputs in log2 domain.
    Grid `(seq_len * num_dv_tiles, heads, batch)` so each (token, head) is
    handled by NUM_DV_TILES programs, matching `_sparse_mla_merge_kernel` in
    the Triton path (avoids the (1,1) launch starvation seen on A100).
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed in this environment")

    D = _DIM_DV
    DV_TILE = block_dv_tile
    assert D % DV_TILE == 0
    NUM_DV_TILES = D // DV_TILE

    dtype = T.bfloat16
    accum_dtype = T.float32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")

    partial_shape = [batch, seq_len, heads, num_split, D]
    lse_shape = [batch, seq_len, heads, num_split]
    out_shape = [batch, seq_len, heads, D]

    # No `out_idx`: caller passes Output buffer (see single-pass decorator's
    # doc comment for the cudagraph-overhead rationale).
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _make():
        @T.prim_func
        def merge(
            Partial: T.Tensor(partial_shape, accum_dtype),  # type: ignore[valid-type]
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore[valid-type]
            Output: T.Tensor(out_shape, dtype),  # type: ignore[valid-type]
        ):
            with T.Kernel(seq_len * NUM_DV_TILES, heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                s_i = bx // NUM_DV_TILES if NUM_DV_TILES > 1 else bx
                tile_id = bx % NUM_DV_TILES if NUM_DV_TILES > 1 else 0
                h_i = by
                b_i = bz

                o_accum = T.alloc_fragment([DV_TILE], accum_dtype)
                po_local = T.alloc_fragment([DV_TILE], accum_dtype)
                lse_max_local = T.alloc_var(accum_dtype)
                lse_logsum_local = T.alloc_var(accum_dtype)
                lse_local = T.alloc_var(accum_dtype)
                scale_local = T.alloc_var(accum_dtype)

                T.clear(o_accum)
                # First pass: find the max LSE across splits. Use a finite
                # sentinel (matches stage-1's empty-split write).
                lse_max_local = T.Cast(accum_dtype, -1.0e30)
                for k in T.serial(num_split):
                    lse_max_local = T.max(lse_max_local, Lse[b_i, s_i, h_i, k])
                # Second pass: log-sum-exp denominator in log2 domain.
                lse_logsum_local = T.Cast(accum_dtype, 0)
                for k in T.serial(num_split):
                    lse_local = Lse[b_i, s_i, h_i, k]
                    lse_logsum_local = lse_logsum_local + T.exp2(
                        lse_local - lse_max_local
                    )
                lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
                # Third pass: weighted sum of partials.
                for k in T.serial(num_split):
                    for i in T.Parallel(DV_TILE):
                        po_local[i] = Partial[b_i, s_i, h_i, k, tile_id * DV_TILE + i]
                    lse_local = Lse[b_i, s_i, h_i, k]
                    scale_local = T.exp2(lse_local - lse_logsum_local)
                    for i in T.Parallel(DV_TILE):
                        o_accum[i] = o_accum[i] + po_local[i] * scale_local
                # Write final bf16 output. T.Cast on assignment.
                for i in T.Parallel(DV_TILE):
                    Output[b_i, s_i, h_i, tile_id * DV_TILE + i] = T.Cast(
                        dtype, o_accum[i]
                    )

        return merge

    return _make()


@functools.lru_cache(maxsize=64)
def _cached_split_kernel(heads, topk, num_split, num_stages, sm_scale):
    return _build_split_kernel(
        heads=heads,
        topk=topk,
        num_split=num_split,
        num_stages=num_stages,
        sm_scale=sm_scale,
    )


@functools.lru_cache(maxsize=64)
def _cached_merge_kernel(heads, num_split):
    return _build_merge_kernel(heads=heads, num_split=num_split)


@functools.lru_cache(maxsize=8)
def _num_compute_units(device_index: int | None) -> int:
    """Cached SM count lookup. Hoisted out of the hot path so each call
    is a single dict hit instead of `from vllm... import ...` + driver
    query."""
    from vllm.utils.platform_utils import num_compute_units

    return num_compute_units(device_index)


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, topk: int, sm_count: int
) -> int:
    """Pick a power-of-2 split count that fills the device without dropping
    per-split work below `_MIN_TOPK_PER_SPLIT`. Returns 1 when the single-pass
    grid already reaches ~1/`_SPLIT_MAX_OCCUPANCY` utilization. Matches the
    Triton path's heuristic so the two kernels exercise the same regime.
    """
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = max(1, topk // _MIN_TOPK_PER_SPLIT)
    # Floor to power of 2.
    ideal = 1 << (ideal.bit_length() - 1) if ideal > 0 else 1
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)
    num_kv_splits = min(ideal, max_splits)
    # Stage-1 requires `(topk // BI) % num_split == 0`.
    NI = topk // _SPLIT_BLOCK_I
    while num_kv_splits > 1 and NI % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def tilelang_sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """TileLang implementation matching `triton_sparse_mla_attention`'s
    signature. Routes between single-pass (`num_kv_splits == 1`) and
    split-KV (`num_kv_splits > 1`) based on the device-saturation heuristic
    that mirrors `_choose_num_kv_splits` in the Triton kernel.

    Args mirror the Triton wrapper:
        q:         [num_tokens, num_heads_q, 576] bf16
        kv:        [seq_kv, 1, 576] bf16
        indices:   [num_tokens, 1, topk] int32 (use -1 for padded slots)
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single.
        sm_count:  pass a cached count to avoid per-call device queries.
    Returns:
        out:       [num_tokens, num_heads_q, 512] bf16
    """
    # Hot-path: skip asserts and `.contiguous()` (no-ops in production —
    # production callers pass already-contiguous tensors). See
    # TILELANG_FINDINGS.md for the Python-overhead profile.
    num_tokens = q.shape[0]
    num_heads_q = q.shape[1]
    topk = indices.shape[-1]
    device = q.device

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = _num_compute_units(device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens,
            max(1, (num_heads_q + 15) // 16),
            topk,
            sm_count,
        )

    # `unsqueeze(0)` is a free view; `.contiguous()` is verifiably a no-op
    # for the production call sites (q is from `torch.cat`, kv is a `.view`
    # of the cache, indices is a fresh Triton-kernel output) so we skip
    # it. Kept the unsqueeze because the kernels are JIT'd against the
    # 4D `[batch, seq_len, heads, dim]` signature.
    q_4d = q.unsqueeze(0)
    kv_4d = kv.unsqueeze(0)
    indices_4d = indices.unsqueeze(0)

    if num_kv_splits == 1:
        kernel = _cached_kernel(
            heads=num_heads_q,
            topk=topk,
            num_stages=_arch_num_stages(device.index),
            sm_scale=float(sm_scale),
        )
        out_4d = _get_buffer(
            "single_out",
            (1, num_tokens, num_heads_q, _DIM_DV),
            torch.bfloat16,
            device,
        )
        lse = _get_buffer(
            "single_lse",
            (1, num_tokens, num_heads_q),
            torch.float32,
            device,
        )
        kernel(q_4d, kv_4d, indices_4d, out_4d, lse)
        return out_4d.squeeze(0)

    split_kernel = _cached_split_kernel(
        heads=num_heads_q,
        topk=topk,
        num_split=num_kv_splits,
        num_stages=_arch_num_stages(device.index),
        sm_scale=float(sm_scale),
    )
    merge_kernel = _cached_merge_kernel(heads=num_heads_q, num_split=num_kv_splits)
    partial = _get_buffer(
        f"split_partial_n{num_kv_splits}",
        (1, num_tokens, num_heads_q, num_kv_splits, _DIM_DV),
        torch.float32,
        device,
    )
    lse = _get_buffer(
        f"split_lse_n{num_kv_splits}",
        (1, num_tokens, num_heads_q, num_kv_splits),
        torch.float32,
        device,
    )
    out_4d = _get_buffer(
        "merge_out",
        (1, num_tokens, num_heads_q, _DIM_DV),
        torch.bfloat16,
        device,
    )
    split_kernel(q_4d, kv_4d, indices_4d, partial, lse)
    merge_kernel(partial, lse, out_4d)
    return out_4d.squeeze(0)
