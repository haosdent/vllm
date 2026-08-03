# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.fp8_sm80 import (
    _E4M3FN_BF16_LUT_CACHE,
    get_e4m3fn_bf16_lut,
)


def test_e4m3fn_bf16_lut_all_bytes_on_cpu() -> None:
    _E4M3FN_BF16_LUT_CACHE.clear()
    device = torch.device("cpu")
    expected = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )

    actual = get_e4m3fn_bf16_lut(device)

    assert actual.data_ptr() == get_e4m3fn_bf16_lut(device).data_ptr()
    torch.testing.assert_close(
        actual.nan_to_num(1234.0), expected.nan_to_num(1234.0), atol=0, rtol=0
    )


def test_e4m3fn_bf16_lut_replaces_signed_nan_codes() -> None:
    _E4M3FN_BF16_LUT_CACHE.clear()
    lut = get_e4m3fn_bf16_lut(torch.device("cpu"), nan_value=480.0)

    assert lut[0x7F].item() == 480.0
    assert lut[0xFF].item() == -480.0
    assert not torch.isnan(lut).any()
