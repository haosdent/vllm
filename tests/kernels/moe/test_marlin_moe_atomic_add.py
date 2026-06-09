# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.scalar_type import scalar_types


@pytest.mark.parametrize("use_atomic_add", [False, True])
def test_fused_marlin_moe_passes_atomic_add_env(monkeypatch, use_atomic_add):
    gemm_kwargs: list[dict[str, object]] = []

    def fake_marlin_gemm(*args, **kwargs):
        gemm_kwargs.append(kwargs)
        return args[1]

    def fake_activation(_activation, output, _input):
        output.zero_()

    monkeypatch.setattr(
        marlin_moe.envs,
        "VLLM_MARLIN_MOE_USE_ATOMIC_ADD",
        use_atomic_add,
    )
    monkeypatch.setattr(marlin_moe.ops, "moe_wna16_marlin_gemm", fake_marlin_gemm)

    hidden_size = 16
    intermediate_size = 16
    num_experts = 2
    hidden_states = torch.zeros((2, hidden_size), dtype=torch.bfloat16)
    w1 = torch.zeros((num_experts, hidden_size // 16, 1), dtype=torch.int32)
    w2 = torch.zeros(
        (num_experts, intermediate_size // 16, hidden_size * 2),
        dtype=torch.int32,
    )
    scales = torch.ones((num_experts, 1, 1), dtype=torch.bfloat16)
    topk_weights = torch.ones((hidden_states.size(0), 1), dtype=torch.float32)

    marlin_moe._fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=scales,
        w2_scale=scales,
        topk_weights=topk_weights,
        num_topk=1,
        quant_type=scalar_types.uint4,
        apply_router_weight_on_input=False,
        expert_map=None,
        block_size_m=8,
        sorted_token_ids=torch.arange(8, dtype=torch.int32),
        expert_ids=torch.zeros((1,), dtype=torch.int32),
        num_tokens_post_padded=torch.tensor(8, dtype=torch.int32),
        activation=MoEActivation.SILU,
        activation_func=fake_activation,
        workspace=torch.zeros((4,), dtype=torch.int32),
    )

    assert [kwargs["use_atomic_add"] for kwargs in gemm_kwargs] == [
        use_atomic_add,
        use_atomic_add,
    ]
