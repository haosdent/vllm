# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import dsv4_sparse_decode_launch as policy
from vllm.v1.attention.ops.dsv4_sparse_decode_launch import (
    SparseDecodeDispatch,
    SparseDecodeLaunchConfig,
    SparseDecodeShape,
)


@pytest.mark.parametrize(
    ("is_cuda", "capability", "gfx942", "gfx950", "expected"),
    [
        (True, 79, False, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (True, 80, False, False, SparseDecodeDispatch.SM80_SPLIT_K),
        (True, 81, False, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (True, 89, False, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (True, 90, False, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (True, None, False, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (False, 80, False, False, SparseDecodeDispatch.SINGLE_PASS),
        (False, None, True, False, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (False, None, False, True, SparseDecodeDispatch.LEGACY_SPLIT_K),
        (False, None, False, False, SparseDecodeDispatch.SINGLE_PASS),
    ],
)
def test_sparse_decode_dispatch_boundaries(
    is_cuda: bool,
    capability: int | None,
    gfx942: bool,
    gfx950: bool,
    expected: SparseDecodeDispatch,
) -> None:
    assert (
        policy.classify_sparse_decode_dispatch(
            is_cuda=is_cuda,
            cuda_capability=capability,
            on_gfx942=gfx942,
            on_gfx950=gfx950,
        )
        is expected
    )


def _a100_shape(num_queries: int = 1) -> SparseDecodeShape:
    return SparseDecodeShape(
        num_queries=num_queries,
        num_heads=8,
        avg_main_len=128.0,
        avg_extra_len=512.0,
        sm_count=108,
    )


def _frozen_legacy_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float,
    avg_extra_len: float,
    block_k: int,
    sm_count: int,
) -> int:
    """The pre-SM80-policy gfx942/gfx950 implementation."""

    def partial_iters(num_splits: int) -> int:
        main_iters = (
            math.ceil(math.ceil(avg_main_len / num_splits) / block_k)
            if avg_main_len > 0
            else 0
        )
        extra_iters = (
            math.ceil(math.ceil(avg_extra_len / num_splits) / block_k)
            if avg_extra_len > 0
            else 0
        )
        return main_iters + extra_iters

    base = max(1, num_queries * heads_blocks)
    sm_count = max(1, sm_count)
    best_splits = 1
    best_cost = None
    for num_splits in range(1, 17):
        waves = (base * num_splits + sm_count - 1) // sm_count
        cost = waves * (1.0 / num_splits + 0.04)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = num_splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + sm_count - 1) // sm_count
        target_iters = partial_iters(best_splits)
        for num_splits in range(1, best_splits):
            waves = (base * num_splits + sm_count - 1) // sm_count
            if waves == target_waves and partial_iters(num_splits) == target_iters:
                return num_splits
    return best_splits


def test_legacy_gfx_split_policy_is_behaviorally_unchanged() -> None:
    for (
        num_queries,
        heads_blocks,
        lengths,
        block_k,
        sm_count,
    ) in product(
        (1, 2, 8, 24, 64, 108, 224, 512),
        (1, 4),
        ((0.0, 0.0), (128.0, 0.0), (128.0, 512.0), (511.5, 127.25)),
        (16, 32),
        (108, 256),
    ):
        avg_main_len, avg_extra_len = lengths
        expected = _frozen_legacy_num_splits(
            num_queries,
            heads_blocks,
            avg_main_len,
            avg_extra_len,
            block_k,
            sm_count,
        )
        actual = policy.legacy_sparse_decode_num_splits(
            num_queries=num_queries,
            heads_blocks=heads_blocks,
            avg_main_len=avg_main_len,
            avg_extra_len=avg_extra_len,
            block_k=block_k,
            sm_count=sm_count,
        )
        assert actual == expected


def test_sm80_search_space_and_discrete_estimate() -> None:
    shape = _a100_shape()
    candidates = policy.sm80_sparse_decode_candidates(shape)

    assert len(candidates) == 3 * 32 * 2
    assert {config.block_h for config in candidates} == {4, 8, 16}
    assert {config.block_k for config in candidates} == {32}
    assert {config.num_splits for config in candidates} == set(range(1, 33))
    assert {config.num_warps for config in candidates} == {4, 8}
    assert {config.num_stages for config in candidates} == {1}

    config = SparseDecodeLaunchConfig(
        block_h=8,
        block_k=32,
        num_splits=3,
        num_warps=8,
    )
    estimate = policy.estimate_sm80_sparse_decode_launch(shape, config)
    assert estimate.heads_blocks == 1
    assert estimate.partial_ctas == 3
    assert estimate.partial_waves == 1
    # ceil(ceil(128 / 3) / 32) + ceil(ceil(512 / 3) / 32) = 2 + 6.
    assert estimate.partial_iters == 8
    assert estimate.padded_head_iters == 64
    assert estimate.reduce_elements == 24


@pytest.mark.parametrize(
    ("num_queries", "expected_splits"),
    [(1, 16), (8, 8), (16, 6), (32, 3)],
)
def test_sm80_safe_fallback_retains_deployed_split_choice(
    num_queries: int, expected_splits: int
) -> None:
    launch = policy.safe_sm80_sparse_decode_launch(_a100_shape(num_queries))
    assert launch == SparseDecodeLaunchConfig(
        block_h=16,
        block_k=32,
        num_splits=expected_splits,
        num_warps=8,
        num_stages=1,
    )


def test_measured_selection_requires_margin() -> None:
    shape = _a100_shape(num_queries=8)
    fallback = policy.safe_sm80_sparse_decode_launch(shape)
    candidate = SparseDecodeLaunchConfig(
        block_h=8,
        block_k=32,
        num_splits=8,
        num_warps=8,
    )

    below_margin = policy.select_sm80_sparse_decode_measurements(
        shape,
        {fallback: 100.0, candidate: 96.0},
        fallback=fallback,
        min_relative_improvement=0.05,
    )
    assert below_margin.fastest == candidate
    assert below_margin.selected == fallback
    assert not below_margin.accepted
    assert below_margin.reason == "below-improvement-margin"

    accepted = policy.select_sm80_sparse_decode_measurements(
        shape,
        {fallback: 100.0, candidate: 94.0},
        fallback=fallback,
        min_relative_improvement=0.05,
    )
    assert accepted.selected == candidate
    assert accepted.accepted
    assert accepted.reason == "accepted-measured-winner"
    assert accepted.relative_improvement == pytest.approx(0.06)


def test_measured_selection_requires_valid_fallback() -> None:
    shape = _a100_shape()
    with pytest.raises(ValueError, match="fallback must have"):
        policy.select_sm80_sparse_decode_measurements(shape, {})


def test_measured_table_is_exact_shape_and_device_bound(monkeypatch) -> None:
    shape = _a100_shape(num_queries=8)
    fallback = policy.safe_sm80_sparse_decode_launch(shape)
    tuned = SparseDecodeLaunchConfig(
        block_h=4,
        block_k=32,
        num_splits=16,
        num_warps=8,
    )
    monkeypatch.setattr(policy, "SM80_MEASURED_LAUNCHES", {shape.measured_key(): tuned})

    assert policy.get_sm80_sparse_decode_launch(shape, fallback=fallback) == tuned
    assert (
        policy.get_sm80_sparse_decode_launch(
            replace(shape, sm_count=56), fallback=fallback
        )
        == fallback
    )


def test_a100_measured_table_contains_only_confirmed_shapes() -> None:
    expected = {
        (64, 8, 128, 512, 108): SparseDecodeLaunchConfig(8, 32, 3, 4),
        (128, 8, 128, 512, 108): SparseDecodeLaunchConfig(8, 32, 4, 8),
    }
    assert expected == policy.SM80_MEASURED_LAUNCHES

    # Current production-layout A/B regresses at q=1/8/32, so those shapes must
    # remain on the deployed fallback along with every other unmeasured batch.
    for num_queries in (1, 2, 4, 8, 16, 32):
        shape = _a100_shape(num_queries)
        fallback = policy.safe_sm80_sparse_decode_launch(shape)
        assert policy.get_sm80_sparse_decode_launch(shape) == fallback

    measured = _a100_shape(64)
    for miss in (
        replace(measured, num_heads=16),
        replace(measured, avg_main_len=256.0),
        replace(measured, avg_extra_len=256.0),
        replace(measured, sm_count=56),
    ):
        assert policy.get_sm80_sparse_decode_launch(miss) == (
            policy.safe_sm80_sparse_decode_launch(miss)
        )
    assert policy.get_sm80_sparse_decode_launch(
        replace(measured, avg_extra_len=511.5),
        fallback=policy.safe_sm80_sparse_decode_launch(measured),
    ) == policy.safe_sm80_sparse_decode_launch(measured)


class _KernelLaunchCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def __getitem__(self, grid: tuple[int, ...]):
        def launch(*args, **kwargs) -> None:
            self.calls.append((grid, kwargs))

        return launch


def _capture_sm80_wrapper_launch(
    monkeypatch: pytest.MonkeyPatch,
    num_queries: int,
    *,
    extra_numel: int | None = None,
    measured_launch_enabled: bool | None = True,
    dispatch: SparseDecodeDispatch = SparseDecodeDispatch.SM80_SPLIT_K,
) -> tuple[tuple[int, ...], dict[str, object]]:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    partial = _KernelLaunchCapture()
    reduce = _KernelLaunchCapture()
    monkeypatch.setattr(
        mod,
        "current_platform",
        SimpleNamespace(is_fp8_fnuz=lambda: False),
    )
    monkeypatch.setattr(mod, "_sparse_decode_dispatch", lambda: dispatch)
    monkeypatch.setattr(mod, "_decode_cu_count", lambda: 108)
    monkeypatch.setattr(
        mod,
        "get_e4m3fn_bf16_lut",
        lambda device: torch.empty(1, dtype=torch.uint16, device=device),
    )
    monkeypatch.setattr(mod, "_sparse_attn_decode_partial_kernel", partial)
    monkeypatch.setattr(mod, "_sparse_attn_decode_reduce_kernel", reduce)

    device = torch.device("meta")
    q = torch.empty((num_queries, 8, 512), dtype=torch.bfloat16, device=device)
    main_cache = torch.empty((1, 1, 584), dtype=torch.uint8, device=device)
    extra_cache = torch.empty_like(main_cache)
    main_indices = torch.empty(num_queries * 128, dtype=torch.int32, device=device)
    main_indptr = torch.empty(num_queries + 1, dtype=torch.int32, device=device)
    extra_indices = torch.empty(
        extra_numel if extra_numel is not None else num_queries * 512,
        dtype=torch.int32,
        device=device,
    )
    extra_indptr = torch.empty(num_queries + 1, dtype=torch.int32, device=device)

    kwargs = {}
    if measured_launch_enabled is not None:
        kwargs["sm80_measured_decode_launch_enabled"] = measured_launch_enabled
    mod._rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=1.0,
        attn_sink=None,
        nope_head_dim=448,
        rope_head_dim=64,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        **kwargs,
    )

    assert len(partial.calls) == 1
    assert len(reduce.calls) == 1
    return partial.calls[0]


@pytest.mark.parametrize("num_queries", [64, 128])
def test_sm80_wrapper_launches_measured_table_config(
    monkeypatch: pytest.MonkeyPatch, num_queries: int
) -> None:
    grid, launch = _capture_sm80_wrapper_launch(monkeypatch, num_queries)
    expected = policy.SM80_MEASURED_LAUNCHES[(num_queries, 8, 128, 512, 108)]

    assert grid == (
        num_queries,
        expected.num_splits,
        math.ceil(8 / expected.block_h),
    )
    assert launch["BLOCK_H"] == expected.block_h
    assert launch["BLOCK_K"] == expected.block_k
    assert launch["NUM_SPLITS"] == expected.num_splits
    assert launch["NUM_STAGES"] == expected.num_stages
    assert launch["num_warps"] == expected.num_warps


def test_sm80_wrapper_requires_full_length_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_queries = 64
    fallback = policy.safe_sm80_sparse_decode_launch(_a100_shape(num_queries))
    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch, num_queries, measured_launch_enabled=False
    )

    assert grid == (num_queries, fallback.num_splits, 1)
    assert launch["BLOCK_H"] == fallback.block_h
    assert launch["NUM_SPLITS"] == fallback.num_splits
    assert launch["num_warps"] == fallback.num_warps


def test_sm80_direct_ragged_call_without_proof_retains_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_queries = 64
    fallback = policy.safe_sm80_sparse_decode_launch(_a100_shape(num_queries))
    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch, num_queries, measured_launch_enabled=None
    )

    assert grid == (num_queries, fallback.num_splits, 1)
    assert launch["BLOCK_H"] == fallback.block_h
    assert launch["NUM_SPLITS"] == fallback.num_splits


@pytest.mark.parametrize("measured_launch_enabled", [False, True])
def test_legacy_wrapper_ignores_sm80_length_proof(
    monkeypatch: pytest.MonkeyPatch, measured_launch_enabled: bool
) -> None:
    num_queries = 64
    fallback = policy.safe_sm80_sparse_decode_launch(_a100_shape(num_queries))
    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch,
        num_queries,
        measured_launch_enabled=measured_launch_enabled,
        dispatch=SparseDecodeDispatch.LEGACY_SPLIT_K,
    )

    assert grid == (num_queries, fallback.num_splits, 1)
    assert launch["BLOCK_H"] == fallback.block_h
    assert launch["NUM_SPLITS"] == fallback.num_splits


@pytest.mark.parametrize("context_len", [131072, 262144])
@pytest.mark.parametrize("num_queries", [1, 8, 32, 64, 128])
def test_sm80_full_length_proof_accepts_target_decode_batches(
    context_len: int, num_queries: int
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    seq_lens_cpu = torch.full((num_queries,), context_len, dtype=torch.int32)

    assert mod.sm80_measured_decode_lengths_are_full(
        seq_lens_cpu, num_queries, max_query_len=1
    )


def test_sm80_uneven_or_short_rows_retain_wrapper_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    num_queries = 64
    seq_lens_cpu = torch.full((num_queries,), 131072, dtype=torch.int32)
    seq_lens_cpu[2] = 2048
    lengths_are_full = mod.sm80_measured_decode_lengths_are_full(
        seq_lens_cpu, num_queries, max_query_len=1
    )
    assert not lengths_are_full

    fallback = policy.safe_sm80_sparse_decode_launch(_a100_shape(num_queries))
    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch, num_queries, measured_launch_enabled=lengths_are_full
    )
    assert grid == (num_queries, fallback.num_splits, 1)
    assert launch["BLOCK_H"] == fallback.block_h
    assert launch["NUM_SPLITS"] == fallback.num_splits

    assert not mod.sm80_measured_decode_lengths_are_full(None, 8, 1)
    assert not mod.sm80_measured_decode_lengths_are_full(
        torch.empty(8, dtype=torch.int32, device="meta"), 8, 1
    )


def test_sm80_wrapper_fractional_q64_length_misses_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_queries = 64
    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch,
        num_queries,
        extra_numel=num_queries * 511 + num_queries // 2,
    )
    shape = replace(_a100_shape(num_queries), avg_extra_len=511.5)
    fallback = policy.safe_sm80_sparse_decode_launch(shape)

    assert grid == (num_queries, fallback.num_splits, 1)
    assert launch["BLOCK_H"] == fallback.block_h
    assert launch["NUM_SPLITS"] == fallback.num_splits
    assert launch["num_warps"] == fallback.num_warps


@pytest.mark.parametrize("context_len", [131072, 262144])
@pytest.mark.parametrize("num_queries", [1, 8, 32, 64, 128])
def test_default_full_cudagraph_capture_uses_qualified_sm80_launch(
    monkeypatch: pytest.MonkeyPatch, context_len: int, num_queries: int
) -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.config.vllm import OPTIMIZATION_LEVEL_02
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    assert (
        OPTIMIZATION_LEVEL_02["compilation_config"]["cudagraph_mode"]
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert CUDAGraphMode.FULL_AND_PIECEWISE.decode_mode() == CUDAGraphMode.FULL

    measured_launch_enabled = mod.sm80_measured_decode_launch_enabled(
        can_use_measured_decode=True,
        causal=True,
        num_decodes=num_queries,
        lengths_are_full=False,
        full_decode_cudagraph=True,
    )
    assert measured_launch_enabled
    assert not mod.sm80_measured_decode_launch_enabled(
        can_use_measured_decode=True,
        causal=True,
        num_decodes=num_queries,
        lengths_are_full=False,
        full_decode_cudagraph=False,
    )

    runtime_seq_lens = torch.full((num_queries,), context_len, dtype=torch.int32)
    assert mod.sm80_measured_decode_lengths_are_full(
        runtime_seq_lens, num_queries, max_query_len=1
    )

    grid, launch = _capture_sm80_wrapper_launch(
        monkeypatch,
        num_queries,
        measured_launch_enabled=measured_launch_enabled,
    )
    shape = _a100_shape(num_queries)
    expected = policy.SM80_MEASURED_LAUNCHES.get(
        (num_queries, 8, 128, 512, 108),
        policy.safe_sm80_sparse_decode_launch(shape),
    )
    assert grid == (
        num_queries,
        expected.num_splits,
        math.ceil(8 / expected.block_h),
    )
    assert launch["BLOCK_H"] == expected.block_h
    assert launch["NUM_SPLITS"] == expected.num_splits
    assert launch["num_warps"] == expected.num_warps
