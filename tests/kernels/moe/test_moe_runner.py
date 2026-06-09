# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner


def test_maybe_reduce_final_output_truncates_padding_before_allreduce(
    monkeypatch,
):
    all_reduce_inputs = []

    def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
        all_reduce_inputs.append(x)
        return x + 1

    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", fake_all_reduce)
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            tp_size=4,
            ep_size=1,
        ),
        _fused_output_is_reduced=False,
    )
    states = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    output = moe_runner.MoERunner._maybe_reduce_final_output(
        runner,
        states,
        trunc_size=6,
    )

    assert [x.shape for x in all_reduce_inputs] == [torch.Size([2, 6])]
    torch.testing.assert_close(output, states[..., :6] + 1)


def test_maybe_reduce_final_output_returns_reduced_tensor_without_noop_slice(
    monkeypatch,
):
    reduced_output = torch.randn(2, 6)

    def fake_all_reduce(x: torch.Tensor) -> torch.Tensor:
        assert x.shape == torch.Size([2, 6])
        return reduced_output

    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", fake_all_reduce)
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            tp_size=4,
            ep_size=1,
        ),
        _fused_output_is_reduced=False,
    )
    states = torch.randn(2, 6)

    output = moe_runner.MoERunner._maybe_reduce_final_output(
        runner,
        states,
        trunc_size=6,
    )

    assert output is reduced_output
