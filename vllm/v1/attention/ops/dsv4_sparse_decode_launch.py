# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch routing for the DeepSeek-V4 sparse decode kernels.

The Triton kernels live in :mod:`rocm_aiter_mla_sparse`. This module keeps the
shared split heuristic and the small exact-shape SM80 launch table testable
without importing Triton or touching an accelerator.
"""

import enum
import math
from dataclasses import dataclass


class SparseDecodeDispatch(enum.Enum):
    """Architecture-level decode route selected before launch tuning."""

    SINGLE_PASS = "single-pass"
    LEGACY_SPLIT_K = "legacy-split-k"
    SM80_SPLIT_K = "sm80-split-k"


@dataclass(frozen=True)
class SparseDecodeLaunchConfig:
    """Compile-time and grid parameters for the split-K partial kernel."""

    block_h: int
    block_k: int
    num_splits: int
    num_warps: int
    num_stages: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("block_h", self.block_h),
            ("block_k", self.block_k),
            ("num_splits", self.num_splits),
            ("num_warps", self.num_warps),
            ("num_stages", self.num_stages),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


def classify_sparse_decode_dispatch(
    *,
    is_cuda: bool,
    cuda_capability: int | None,
    on_gfx942: bool,
    on_gfx950: bool,
) -> SparseDecodeDispatch:
    """Choose the policy family without touching a device."""

    if is_cuda and cuda_capability == 80:
        return SparseDecodeDispatch.SM80_SPLIT_K
    if is_cuda or on_gfx942 or on_gfx950:
        return SparseDecodeDispatch.LEGACY_SPLIT_K
    return SparseDecodeDispatch.SINGLE_PASS


def sparse_decode_partial_iters(
    avg_main_len: float,
    avg_extra_len: float,
    num_splits: int,
    block_k: int,
) -> int:
    """Number of discrete ``BLOCK_K`` iterations walked by one partial CTA."""

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


def legacy_sparse_decode_num_splits(
    *,
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float,
    avg_extra_len: float,
    block_k: int,
    sm_count: int,
    max_splits: int = 16,
) -> int:
    """Return the deployed CUDA/gfx942/gfx950 split-K heuristic."""

    base = max(1, num_queries * heads_blocks)
    sm_count = max(1, sm_count)
    mu = 0.04
    best_splits = 1
    best_cost: float | None = None
    for num_splits in range(1, max_splits + 1):
        waves = (base * num_splits + sm_count - 1) // sm_count
        cost = waves * (1.0 / num_splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = num_splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + sm_count - 1) // sm_count
        target_iters = sparse_decode_partial_iters(
            avg_main_len, avg_extra_len, best_splits, block_k
        )
        for num_splits in range(1, best_splits):
            waves = (base * num_splits + sm_count - 1) // sm_count
            partial_iters = sparse_decode_partial_iters(
                avg_main_len, avg_extra_len, num_splits, block_k
            )
            if waves == target_waves and partial_iters == target_iters:
                best_splits = num_splits
                break
    return best_splits


# Exact A100-SXM4-80GB (108 SM) entries that beat the production wrapper's
# deployed fallback. Unlisted shapes deliberately retain the caller's fallback.
SM80_MEASURED_LAUNCHES: dict[
    tuple[int, int, int, int, int], SparseDecodeLaunchConfig
] = {
    (64, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=3, num_warps=4
    ),
    (128, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=4, num_warps=8
    ),
}


def get_sm80_sparse_decode_launch(
    *,
    num_queries: int,
    num_heads: int,
    avg_main_len: float,
    avg_extra_len: float,
    sm_count: int,
    fallback: SparseDecodeLaunchConfig,
) -> SparseDecodeLaunchConfig:
    """Resolve an exact measured entry, otherwise return ``fallback``."""

    if (
        not math.isfinite(avg_main_len)
        or not math.isfinite(avg_extra_len)
        or not float(avg_main_len).is_integer()
        or not float(avg_extra_len).is_integer()
    ):
        return fallback
    key = (
        num_queries,
        num_heads,
        int(avg_main_len),
        int(avg_extra_len),
        sm_count,
    )
    return SM80_MEASURED_LAUNCHES.get(key, fallback)
