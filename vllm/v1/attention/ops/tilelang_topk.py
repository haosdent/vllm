# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang radix top-K — prototype alternative to the upstream CUDA
`torch.ops._C.persistent_topk` (from `csrc/topk.cu` / `csrc/persistent_topk.cuh`).

The kernel body is the upstream
`/root/tilelang/examples/deepseek_v32/topk_selector.py:tl_topk_impl` —
two-stage radix sort (8-bit quick pass + threshold-bin refinement). It
uses only `T.atomic_add`, bit reinterprets, and shared memory — no FP8 or
sm_89-specific intrinsics. Expected to compile and run on SM80.

The wrapper signature matches `torch.ops._C.persistent_topk(logits,
seq_lens, topk_indices, workspace, topk_tokens, max_seq_len)` for drop-in
substitution at the bench call site. `workspace` is ignored (TileLang
manages its own shared memory) and `max_seq_len` is implicit in `logits`.
"""

import functools

import torch

try:
    import tilelang
    from tilelang import language as T

    TILELANG_AVAILABLE = True
except ImportError:
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


def _convert_to_uint16(x):
    """Lift from upstream — radix-sort key for fp32 in 8 bits via fp16 cast."""
    hval = T.cast(x, T.float16)
    bits_uint = T.reinterpret(hval, T.uint16)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & 0xFFFF, bits_uint | 0x8000)
    return bits_uint >> 8


def _convert_to_uint32(x):
    """Lift from upstream — radix-sort key for fp32 in 32 bits, sortable."""
    bits_uint = T.reinterpret(x, T.uint32)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.cast(0xFFFFFFFF, T.uint32),
        bits_uint | T.cast(0x80000000, T.uint32),
    )
    return bits_uint


def _build_topk_kernel(topk: int):
    """Lift of `tl_topk_impl` from upstream TileLang DSV3.2 examples."""
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed in this environment")

    in_dtype = T.float32
    out_dtype = T.int32
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 4096

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    def _make():
        @T.prim_func
        def tl_topk_kernel(
            input: T.Tensor[(batch, seq_len), in_dtype],  # type: ignore[valid-type]
            index: T.Tensor[(batch, topk), out_dtype],  # type: ignore[valid-type]
            starts: T.Tensor[(batch,), out_dtype],  # type: ignore[valid-type]
            ends: T.Tensor[(batch,), out_dtype],  # type: ignore[valid-type]
        ):
            with T.Kernel(batch, threads=BLOCK_SIZE) as bx:
                tx = T.get_thread_binding()

                s_threshold_bin_id = T.alloc_shared([1], T.int32)
                s_histogram = T.alloc_shared([RADIX + 1], T.int32)
                s_num_input = T.alloc_shared([2], T.int32)
                s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

                l_threshold_bin_id = T.alloc_var(T.int32)
                l_new_topk = T.alloc_var(T.int32)
                l_num_input = T.alloc_var(T.int32)
                l_bin_id32 = T.alloc_var(T.int32)
                l_val = T.alloc_var(T.int32)
                l_start_idx = T.alloc_var(T.int32)
                l_end_idx = T.alloc_var(T.int32)
                l_start_pos = T.alloc_var(T.int32)
                l_out_pos = T.alloc_var(T.int32)
                pos = T.alloc_var(T.int32)

                l_new_topk = topk
                l_start_idx = starts[bx]
                l_end_idx = ends[bx]

                # Stage 1: 8-bit quick pass.
                T.fill(s_histogram, 0)
                T.fill(s_num_input[0], 0)
                # Initialise the threshold-bin sentinel. If `total_count` ≤
                # `topk` (e.g. dummy short sequences during cudagraph
                # profile), the threshold-finding `if hist[tx] > topk ...`
                # never fires for any tx, and `s_threshold_bin_id` would
                # otherwise read uninitialised shared memory. With the
                # sentinel = -1, the subsequent `hist[threshold + 1]` read
                # lands at `hist[0]` (the total count, which is valid) and
                # the `bin_id > -1` test puts every element on the
                # "directly write to output" path — the natural behaviour
                # when there are fewer valid elements than topk slots.
                if tx == 0:
                    s_threshold_bin_id[0] = -1
                T.sync_threads()
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    input_idx = s * BLOCK_SIZE + tx
                    if (
                        input_idx < l_end_idx
                        and input_idx >= l_start_idx
                        and input_idx < seq_len
                    ):
                        inval_int16 = _convert_to_uint16(input[bx, input_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                # Cumsum (reverse) on the RADIX histogram bins.
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val
                    T.sync_threads(3, RADIX)
                    if (
                        s_histogram[tx] > l_new_topk
                        and s_histogram[tx + 1] <= l_new_topk
                    ):
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # Collect indices above the threshold bin.
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = s * BLOCK_SIZE + tx
                    if (
                        input_idx < l_end_idx
                        and input_idx >= l_start_idx
                        and input_idx < seq_len
                    ):
                        bin_id = _convert_to_uint16(input[bx, input_idx])
                        l_bin_id32 = T.cast(bin_id, T.int32)
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(
                                s_histogram[l_bin_id32 + 1], 1, return_prev=True
                            )
                            index[bx, pos] = input_idx
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            # Defensive: drop the overflow if a pathological
                            # input puts >SMEM_INPUT_SIZE elements in the
                            # threshold bin. Realistic GLM-5.1 inputs put
                            # ~273 elements/bin at seq_len=70K, well under
                            # the 4K cap; the actual cudagraph-time illegal-
                            # address bug fixed by initialising
                            # s_threshold_bin_id below.
                            if pos < SMEM_INPUT_SIZE:
                                s_input_idx[0, pos] = input_idx

                # Stage 2: tail pass — refine the threshold bucket up to 4
                # rounds. Each round examines one more byte of the float32
                # sortable representation (`convert_to_uint32`), from the top
                # byte (round 0) to the bottom byte (round 3).
                for round in T.serial(4):
                    if l_new_topk <= 0:
                        break

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                        # Reset the threshold sentinel each round for the
                        # same reason as the stage-1 init above (rounds
                        # 1-3 may also have all-zero histograms if the
                        # previous round refined everything into a
                        # single bin or l_num_input dropped to 0).
                        s_threshold_bin_id[0] = -1
                    T.sync_threads()

                    # Clamp at SMEM_INPUT_SIZE — the stage-1 atomic_add may have
                    # overflowed the buffer on degenerate inputs (see the
                    # bounds-guarded write in stage 1).
                    l_num_input = T.min(s_num_input[r_idx], SMEM_INPUT_SIZE)
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            val = input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]
                            l_bin_id32 = T.cast(
                                (_convert_to_uint32(val) >> (24 - round * 8)) & 0xFF,
                                T.int32,
                            )
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()

                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[tx] = l_val
                        T.sync_threads(3, RADIX)
                        if (
                            s_histogram[tx] > l_new_topk
                            and s_histogram[tx + 1] <= l_new_topk
                        ):
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()

                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            val = input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]
                            l_bin_id32 = T.cast(
                                (_convert_to_uint32(val) >> (24 - round * 8)) & 0xFF,
                                T.int32,
                            )
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = (
                                    T.atomic_add(
                                        s_histogram[l_bin_id32 + 1],
                                        1,
                                        return_prev=True,
                                    )
                                    + l_start_pos
                                )
                                index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    # Last round: best-effort write any
                                    # remaining threshold-equal elements.
                                    l_out_pos = (
                                        T.atomic_add(
                                            s_histogram[l_bin_id32 + 1],
                                            1,
                                            return_prev=True,
                                        )
                                        + l_start_pos
                                    )
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = s_input_idx[
                                            r_idx, s * BLOCK_SIZE + tx
                                        ]
                                else:
                                    pos = T.atomic_add(
                                        s_num_input[r_idx ^ 1],
                                        1,
                                        return_prev=True,
                                    )
                                    if pos < SMEM_INPUT_SIZE:
                                        s_input_idx[r_idx ^ 1, pos] = s_input_idx[
                                            r_idx, s * BLOCK_SIZE + tx
                                        ]

        return tl_topk_kernel

    return _make()


@functools.lru_cache(maxsize=16)
def _cached_topk(topk: int):
    return _build_topk_kernel(topk)


def tilelang_persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in alternative to `torch.ops._C.persistent_topk`.

    Args:
        logits:        [num_rows, max_seq_len] float32 — same as upstream.
        seq_lens:      [num_rows] int32 — per-row valid range end (start = 0
            unless `starts` is provided).
        topk_indices:  [num_rows, topk_tokens] int32 — output, written in place.
        topk_tokens:   K for the top-K.
        starts:        optional [num_rows] int32 — per-row valid range start
            (defaults to all-zeros, matching the upstream `persistent_topk` API).

    Returns:
        topk_indices (the same tensor, returned for convenience).
    """
    if not TILELANG_AVAILABLE:
        raise ImportError("tilelang is not installed")

    assert logits.dtype == torch.float32
    assert logits.ndim == 2
    assert seq_lens.dtype == torch.int32
    assert topk_indices.dtype == torch.int32
    num_rows = logits.shape[0]

    if starts is None:
        starts = torch.zeros(num_rows, dtype=torch.int32, device=logits.device)

    kernel = _cached_topk(topk_tokens)
    kernel(logits, topk_indices, starts, seq_lens)
    return topk_indices
