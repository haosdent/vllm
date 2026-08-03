# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 E4M3FN encode/decode helpers for Triton kernels on pre-SM89 CUDA.

Triton rejects conversions between ``fp8e4nv`` and other floating-point
types below SM89 (raw bitcasts remain legal). The helpers in this module use
a bit-exact software encoder and either an ALU or lookup-table decoder on
those devices, while preserving native conversions everywhere else.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def native_fp8_cast_supported() -> bool:
    """Return whether Triton may emit native E4M3FN conversions."""
    if not current_platform.is_cuda():
        return True
    return current_platform.supports_fp8()


# Globals referenced by @triton.jit are compile-time constants. Defaulting to
# software conversion is safe: a failed platform probe affects performance,
# not correctness or kernel compatibility.
try:
    _NATIVE_FP8_CAST = tl.constexpr(native_fp8_cast_supported())
except Exception:
    _NATIVE_FP8_CAST = tl.constexpr(False)


_E4M3FN_BF16_LUT_CACHE: dict[tuple[torch.device, float | None], torch.Tensor] = {}


def get_e4m3fn_bf16_lut(
    device: torch.device, nan_value: float | None = None
) -> torch.Tensor:
    """Return a cached 256-entry E4M3FN-to-BF16 decode table.

    ``nan_value`` replaces the positive and negative NaN encodings when a
    finite sentinel is required by a downstream reduction.
    """
    key = (device, nan_value)
    lut = _E4M3FN_BF16_LUT_CACHE.get(key)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        if nan_value is not None:
            lut[0x7F] = nan_value
            lut[0xFF] = -nan_value
        _E4M3FN_BF16_LUT_CACHE[key] = lut
    return lut


@triton.jit
def _f32_to_e4m3fn_u8(x):
    """Encode FP32 to saturating E4M3FN bytes with round-to-nearest-even."""
    fbits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign8 = ((fbits >> 24) & 0x80).to(tl.uint8)
    mag = fbits & 0x7FFFFFFF

    is_nan = mag > 0x7F800000

    exp = (mag >> 23).to(tl.int32) - 127
    mant = (mag & 0x7FFFFF) | 0x800000

    # Normal E4M3 keeps three mantissa bits. Subnormals progressively drop
    # more bits; cap the shift for FP32 subnormals and zero.
    shift = tl.minimum(20 + tl.maximum(-6 - exp, 0), 30)
    keep = (mant >> shift).to(tl.uint32)
    rest = mant & ((1 << shift) - 1)
    half = (1 << (shift - 1)).to(tl.uint32)
    round_up = (rest > half) | ((rest == half) & ((keep & 1) == 1))
    keep = keep + round_up.to(tl.uint32)

    # ``keep`` includes the implicit bit for normal values. Saturate every
    # finite overflow, including a mantissa carry, to the maximum finite code.
    val_normal = (((exp + 6) << 3).to(tl.uint32) + keep).to(tl.uint32)
    val = tl.where(exp >= -6, val_normal, keep)
    val = tl.minimum(val, 0x7E)
    val = tl.where(is_nan, 0x7F, val)
    return (val.to(tl.uint8) | sign8).to(tl.uint8)


@triton.jit
def _encode_e4m3fn_u8(x):
    """Encode E4M3FN as bytes, using native conversion when supported."""
    if _NATIVE_FP8_CAST:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        return _f32_to_e4m3fn_u8(x)


@triton.jit
def _encode_fp8_u8(x, USE_FNUZ: tl.constexpr):
    """Encode FNUZ on gfx942 and E4M3FN everywhere else."""
    if USE_FNUZ:
        return x.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    else:
        return _encode_e4m3fn_u8(x)


@triton.jit
def _e4m3fn_to_f32_alu(u):
    """Decode E4M3FN bytes exactly with integer arithmetic."""
    u32 = u.to(tl.uint32)
    sign = tl.where((u32 & 0x80) != 0, -1.0, 1.0)
    exp = ((u32 >> 3) & 0xF).to(tl.float32)
    mant = (u32 & 0x7).to(tl.float32)
    is_normal = exp > 0.0
    m = tl.where(is_normal, mant + 8.0, mant)
    e = tl.where(is_normal, exp, 1.0) - 10.0
    val = sign * m * tl.exp2(e)
    return tl.where((u32 & 0x7F) == 0x7F, float("nan"), val)


@triton.jit
def _decode_fp8_f32(u, USE_FNUZ: tl.constexpr):
    """Decode FNUZ or E4M3FN bytes to FP32 without a lookup table."""
    if USE_FNUZ:
        return u.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    elif _NATIVE_FP8_CAST:
        return u.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        return _e4m3fn_to_f32_alu(u)


@triton.jit
def _decode_fp8_lut(u, USE_FNUZ: tl.constexpr, lut_ptr):
    """Decode FP8 bytes, using ``lut_ptr`` only on pre-SM89 CUDA."""
    if USE_FNUZ:
        return u.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    elif _NATIVE_FP8_CAST:
        return u.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        return tl.load(lut_ptr + u.to(tl.uint32))
