# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure launch-policy helpers for the DeepSeek-V4 sparse decode kernels.

The Triton kernels live in :mod:`rocm_aiter_mla_sparse`.  Keeping the policy
here makes architecture routing and benchmark selection testable without an
accelerator or a Triton launch.  SM80 has an explicit measured-config table;
until a benchmark result clears the selection margin, its entry deliberately
falls back to the already deployed ``BLOCK_H=16``/``num_warps=8`` policy.
"""

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass

SM80_BLOCK_H_CANDIDATES = (4, 8, 16)
SM80_NUM_SPLITS_CANDIDATES = tuple(range(1, 33))
SM80_NUM_WARPS_CANDIDATES = (4, 8)

SM80_SAFE_BLOCK_H = 16
SM80_SAFE_BLOCK_K = 32
SM80_SAFE_NUM_WARPS = 8
SM80_SAFE_NUM_STAGES = 1
SM80_SAFE_MAX_SPLITS = 16

# A live winner must beat the deployed fallback by this much before it can be
# emitted as an accepted policy selection.  This is deliberately above normal
# microbenchmark noise and is recorded in every benchmark artifact.
SM80_MIN_MEASURED_IMPROVEMENT = 0.05


class SparseDecodeDispatch(enum.Enum):
    """Architecture-level decode route selected before launch tuning."""

    SINGLE_PASS = "single-pass"
    LEGACY_SPLIT_K = "legacy-split-k"
    SM80_SPLIT_K = "sm80-split-k"


@dataclass(frozen=True)
class SparseDecodeShape:
    """Shape inputs that affect sparse-decode launch occupancy."""

    num_queries: int
    num_heads: int
    avg_main_len: float
    avg_extra_len: float
    sm_count: int

    def __post_init__(self) -> None:
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.sm_count <= 0:
            raise ValueError("sm_count must be positive")
        if not math.isfinite(self.avg_main_len) or self.avg_main_len < 0:
            raise ValueError("avg_main_len must be finite and non-negative")
        if not math.isfinite(self.avg_extra_len) or self.avg_extra_len < 0:
            raise ValueError("avg_extra_len must be finite and non-negative")

    def measured_key(self) -> tuple[int, int, int, int, int] | None:
        """Return an exact static-table key for fixed-length measured shapes.

        Ragged production batches can have fractional average lengths.  They
        intentionally miss the measured table and retain the safe fallback.
        """

        if (
            not float(self.avg_main_len).is_integer()
            or not float(self.avg_extra_len).is_integer()
        ):
            return None
        return (
            self.num_queries,
            self.num_heads,
            int(self.avg_main_len),
            int(self.avg_extra_len),
            self.sm_count,
        )


@dataclass(frozen=True)
class SparseDecodeLaunchConfig:
    """Compile-time and grid parameters for the split-K partial kernel."""

    block_h: int
    block_k: int
    num_splits: int
    num_warps: int
    num_stages: int = SM80_SAFE_NUM_STAGES

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

    def sort_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.block_h,
            self.block_k,
            self.num_splits,
            self.num_warps,
            self.num_stages,
        )


@dataclass(frozen=True)
class SparseDecodeLaunchEstimate:
    """Discrete, unit-free work estimate emitted beside benchmark configs."""

    heads_blocks: int
    partial_ctas: int
    partial_waves: int
    partial_iters: int
    padded_head_iters: int
    reduce_elements: int


@dataclass(frozen=True)
class SparseDecodeMeasurementSelection:
    """Result of applying the measured-latency margin gate."""

    fallback: SparseDecodeLaunchConfig
    fastest: SparseDecodeLaunchConfig
    selected: SparseDecodeLaunchConfig
    fallback_latency_us: float
    fastest_latency_us: float
    relative_improvement: float
    min_relative_improvement: float
    accepted: bool
    reason: str


def classify_sparse_decode_dispatch(
    *,
    is_cuda: bool,
    cuda_capability: int | None,
    on_gfx942: bool,
    on_gfx950: bool,
) -> SparseDecodeDispatch:
    """Choose the policy family without touching a device.

    CUDA used split-K before the SM80 policy existed, so CUDA capabilities
    other than exactly 8.0 remain on that legacy route.  Likewise gfx942 and
    gfx950 retain their existing shared heuristic byte-for-byte at the caller.
    """

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
    max_splits: int = SM80_SAFE_MAX_SPLITS,
) -> int:
    """The deployed gfx split heuristic with device count made explicit.

    The arithmetic and tie-breaking intentionally match the original helper.
    SM80 uses this as its known-safe fallback while live candidates may search
    a wider range.  gfx942/gfx950 continue to call it with ``max_splits=16``.
    """

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


def safe_sm80_sparse_decode_launch(
    shape: SparseDecodeShape,
) -> SparseDecodeLaunchConfig:
    """Return the already deployed SM80 launch as the unmeasured fallback."""

    heads_blocks = math.ceil(shape.num_heads / SM80_SAFE_BLOCK_H)
    num_splits = legacy_sparse_decode_num_splits(
        num_queries=shape.num_queries,
        heads_blocks=heads_blocks,
        avg_main_len=shape.avg_main_len,
        avg_extra_len=shape.avg_extra_len,
        block_k=SM80_SAFE_BLOCK_K,
        sm_count=shape.sm_count,
    )
    return SparseDecodeLaunchConfig(
        block_h=SM80_SAFE_BLOCK_H,
        block_k=SM80_SAFE_BLOCK_K,
        num_splits=num_splits,
        num_warps=SM80_SAFE_NUM_WARPS,
    )


def sm80_sparse_decode_candidates(
    shape: SparseDecodeShape,
) -> tuple[SparseDecodeLaunchConfig, ...]:
    """Return the canonical A100 sweep space for a measured shape."""

    del shape  # Reserved for future legality pruning; the space is fixed today.
    return tuple(
        SparseDecodeLaunchConfig(
            block_h=block_h,
            block_k=SM80_SAFE_BLOCK_K,
            num_splits=num_splits,
            num_warps=num_warps,
        )
        for block_h in SM80_BLOCK_H_CANDIDATES
        for num_splits in SM80_NUM_SPLITS_CANDIDATES
        for num_warps in SM80_NUM_WARPS_CANDIDATES
    )


def estimate_sm80_sparse_decode_launch(
    shape: SparseDecodeShape,
    config: SparseDecodeLaunchConfig,
) -> SparseDecodeLaunchEstimate:
    """Estimate launch occupancy using discrete CTAs, waves, and loop trips.

    No fitted timing coefficient is used: the fields are emitted independently
    so measured results remain the sole source of a winner.  ``padded_head_iters``
    exposes masked-head work when ``BLOCK_H`` exceeds the local head count, and
    ``reduce_elements`` exposes the split-dependent second-kernel cost.
    """

    heads_blocks = math.ceil(shape.num_heads / config.block_h)
    partial_ctas = shape.num_queries * heads_blocks * config.num_splits
    partial_waves = math.ceil(partial_ctas / shape.sm_count)
    partial_iters = sparse_decode_partial_iters(
        shape.avg_main_len,
        shape.avg_extra_len,
        config.num_splits,
        config.block_k,
    )
    return SparseDecodeLaunchEstimate(
        heads_blocks=heads_blocks,
        partial_ctas=partial_ctas,
        partial_waves=partial_waves,
        partial_iters=partial_iters,
        padded_head_iters=partial_waves * partial_iters * config.block_h,
        reduce_elements=(shape.num_queries * shape.num_heads * config.num_splits),
    )


# Exact A100-SXM4-80GB (108 SM) entries accepted by a seven-repeat interleaved
# A/B confirmation after a full canonical sweep.  Every repeat passed the
# numerical contract and each entry cleared the 5% paired-median margin and 2%
# per-arm CV gates.  Unlisted shapes deliberately retain the safe launch.
# Evidence SHA256: db84e76bac48693a784802a3f6e3e5d85842cf4bfa5a3f90edf1eb3a000a0c95
SM80_MEASURED_LAUNCHES: Mapping[
    tuple[int, int, int, int, int], SparseDecodeLaunchConfig
] = {
    (1, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=4, block_k=32, num_splits=20, num_warps=8
    ),
    (8, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=10, num_warps=8
    ),
    (32, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=6, num_warps=4
    ),
    (64, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=3, num_warps=4
    ),
    (128, 8, 128, 512, 108): SparseDecodeLaunchConfig(
        block_h=8, block_k=32, num_splits=4, num_warps=8
    ),
}


def get_sm80_sparse_decode_launch(
    shape: SparseDecodeShape,
    *,
    fallback: SparseDecodeLaunchConfig | None = None,
) -> SparseDecodeLaunchConfig:
    """Resolve an exact measured entry, otherwise return the safe fallback."""

    if fallback is None:
        fallback = safe_sm80_sparse_decode_launch(shape)
    key = shape.measured_key()
    return SM80_MEASURED_LAUNCHES.get(key, fallback) if key is not None else fallback


def select_sm80_sparse_decode_measurements(
    shape: SparseDecodeShape,
    measurements_us: Mapping[SparseDecodeLaunchConfig, float],
    *,
    fallback: SparseDecodeLaunchConfig | None = None,
    min_relative_improvement: float = SM80_MIN_MEASURED_IMPROVEMENT,
) -> SparseDecodeMeasurementSelection:
    """Select a measured winner only when it clears the fallback margin.

    Failed/non-finite candidates are ignored.  The fallback itself must have a
    valid measurement, which prevents a compile failure or missing baseline
    from silently promoting an incomparable candidate.
    """

    if not 0.0 <= min_relative_improvement < 1.0:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    if fallback is None:
        fallback = safe_sm80_sparse_decode_launch(shape)

    candidate_set = set(sm80_sparse_decode_candidates(shape))
    if fallback not in candidate_set:
        raise ValueError("fallback must belong to the canonical SM80 search space")

    fallback_latency = measurements_us.get(fallback)
    if (
        fallback_latency is None
        or not math.isfinite(fallback_latency)
        or fallback_latency <= 0
    ):
        raise ValueError("fallback must have a finite, positive measurement")

    valid = [
        (config, latency)
        for config, latency in measurements_us.items()
        if config in candidate_set and math.isfinite(latency) and latency > 0
    ]
    fastest, fastest_latency = min(
        valid,
        key=lambda item: (item[1], item[0].sort_key()),
    )
    relative_improvement = (fallback_latency - fastest_latency) / fallback_latency
    accepted = (
        fastest != fallback and relative_improvement + 1e-12 >= min_relative_improvement
    )

    if fastest == fallback:
        reason = "fallback-is-fastest"
    elif accepted:
        reason = "accepted-measured-winner"
    else:
        reason = "below-improvement-margin"

    return SparseDecodeMeasurementSelection(
        fallback=fallback,
        fastest=fastest,
        selected=fastest if accepted else fallback,
        fallback_latency_us=fallback_latency,
        fastest_latency_us=fastest_latency,
        relative_improvement=relative_improvement,
        min_relative_improvement=min_relative_improvement,
        accepted=accepted,
        reason=reason,
    )
