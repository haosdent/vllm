# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import marlin_moe


def test_marlin_moe_block_size_default_selection(monkeypatch):
    monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

    selector = getattr(marlin_moe, "_select_marlin_moe_block_size", None)
    assert selector is not None

    assert selector(M=128, topk=8, E=64) == 32
    assert selector(M=1, topk=1, E=64) == 8


def test_marlin_moe_block_size_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "48")

    assert marlin_moe._select_marlin_moe_block_size(M=1, topk=1, E=64) == 48
    assert marlin_moe._select_batched_marlin_moe_block_size(batch_tokens_max=1) == 48


def test_marlin_moe_block_size_env_override_keeps_1byte_minimum(monkeypatch):
    monkeypatch.setenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "8")

    assert (
        marlin_moe._select_marlin_moe_block_size(
            M=1, topk=1, E=64, input_dtype=torch.uint8
        )
        == 16
    )
    assert (
        marlin_moe._select_batched_marlin_moe_block_size(
            batch_tokens_max=1, input_dtype=torch.uint8
        )
        == 16
    )


def test_marlin_moe_block_size_invalid_env(monkeypatch):
    monkeypatch.setenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "24")

    with pytest.raises(ValueError, match="VLLM_MARLIN_MOE_BLOCK_SIZE_M"):
        marlin_moe._select_marlin_moe_block_size(M=1, topk=1, E=64)
