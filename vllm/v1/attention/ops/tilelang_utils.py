# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for TileLang sparse-MLA kernels and Phase 4 fusion ops.

Centralizes three patterns that were duplicated across kernel modules:
- `tilelang_num_stages`: arch-aware pipeline-stages count.
- `_FP32_COS_SIN_CACHE` + `ensure_fp32_cos_sin`: lazy fp32 view of the
  rotary cos/sin cache, cached against the source tensor's data_ptr.
- `_BUFFER_CACHE` + `get_buffer`: cudagraph-stable output buffer cache,
  keyed by (role, shape, dtype, device_index).
"""

import functools

import torch


@functools.lru_cache(maxsize=8)
def tilelang_num_stages(device_index: int | None = None) -> int:
    """Pipeline-stages count for TileLang prim_funcs targeting this device.
    1 on Ampere (164 KB shared cap can't fit upstream's default 2), 2 on
    Hopper+. Cached because every JIT-compile cache miss reads it.
    """
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability(device_index)
    return 2 if cap.major >= 9 else 1


_FP32_COS_SIN_CACHE: dict[int, torch.Tensor] = {}


def ensure_fp32_cos_sin(cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """Return an fp32 view of `cos_sin_cache`, casting + caching by source
    tensor identity. The TileLang RoPE kernels want fp32 cos/sin for
    precision; the source may be bf16 if the underlying RoPE class casts
    on demand (`DeepseekScalingRotaryEmbedding`). Cached so the conversion
    cost amortizes across all token calls."""
    if cos_sin_cache.dtype == torch.float32:
        return cos_sin_cache
    key = cos_sin_cache.data_ptr()
    cached = _FP32_COS_SIN_CACHE.get(key)
    if cached is None or cached.shape != cos_sin_cache.shape:
        cached = cos_sin_cache.to(torch.float32).contiguous()
        _FP32_COS_SIN_CACHE[key] = cached
    return cached


_BUFFER_CACHE: dict[tuple, torch.Tensor] = {}


def get_buffer(
    role: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Cached output buffer for a TileLang kernel call. Calls inside
    cudagraph capture allocate from the graph pool on first miss and reuse
    the same address on subsequent replays — required because PyTorch's
    `torch.empty` inside a custom-op impl would otherwise hand out fresh
    memory each call, breaking cudagraph's stable-pointer invariant.

    `shape` must already be a tuple (callers should not pass `torch.Size`,
    which forces a per-call conversion in the cache key)."""
    key = (role, shape, dtype, device.index)
    buf = _BUFFER_CACHE.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _BUFFER_CACHE[key] = buf
    return buf
