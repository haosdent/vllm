# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness tests for the pre-SM89 E4M3FN codec."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import (
    _e4m3fn_to_f32_alu,
    _f32_to_e4m3fn_u8,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, _f32_to_e4m3fn_u8(x), mask=mask)


@triton.jit
def _decode_kernel(u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    u = tl.load(u_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, _e4m3fn_to_f32_alu(u), mask=mask)


def _encode(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.uint8)
    _encode_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out


def _decode(u: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(u, dtype=torch.float32)
    _decode_kernel[(triton.cdiv(u.numel(), 1024),)](u, out, u.numel(), BLOCK=1024)
    return out


def _assert_bytes_equal(
    actual: torch.Tensor, expected: torch.Tensor, inputs: torch.Tensor
) -> None:
    if not torch.equal(actual, expected):
        bad = (actual != expected).nonzero()[:10].flatten()
        raise AssertionError(
            f"mismatches at inputs {inputs.flatten()[bad].tolist()}: "
            f"got {actual.flatten()[bad].tolist()}, "
            f"expected {expected.flatten()[bad].tolist()}"
        )


def test_encoder_matches_torch_on_clamped_domain() -> None:
    torch.manual_seed(0)
    values = [
        (torch.rand(1_000_000, device="cuda") * 2 - 1) * 448.0,
        torch.exp2(torch.rand(1_000_000, device="cuda") * 30 - 21)
        * torch.where(torch.rand(1_000_000, device="cuda") > 0.5, 1.0, -1.0),
    ]
    x = torch.cat(values).float()
    expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), expected, x)


def test_encoder_exact_ties_and_boundaries() -> None:
    all_bytes = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite_values = (
        all_bytes[(all_bytes & 0x7F) != 0x7F].view(torch.float8_e4m3fn).float()
    )
    positive = torch.sort(finite_values[finite_values >= 0]).values
    midpoints = (positive[:-1] + positive[1:]) / 2
    values = [
        finite_values,
        midpoints,
        -midpoints,
        torch.tensor(
            [448.0, 449.0, 455.9, 456.0, 464.0, 1e30, -1e30, 0.0, -0.0],
            device="cuda",
        ),
        torch.tensor(
            [2**-9, 2**-10, 1.5 * 2**-10, 2**-11, 2**-20, -(2**-10)],
            device="cuda",
        ),
    ]
    x = torch.cat([value.float().flatten() for value in values])
    expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), expected, x)


def test_alu_decoder_all_bytes() -> None:
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    actual = _decode(u)
    expected = u.view(torch.float8_e4m3fn).float()
    assert torch.equal(actual.nan_to_num(1234.5), expected.nan_to_num(1234.5))
    assert actual[(u & 0x7F) == 0x7F].isnan().all()


def test_roundtrip_all_finite_bytes() -> None:
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = u[(u & 0x7F) != 0x7F]
    assert torch.equal(_encode(_decode(finite)), finite)
