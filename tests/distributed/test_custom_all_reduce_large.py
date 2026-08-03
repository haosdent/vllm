# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom all-reduce above the default 8 MB cap.

`VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE` raises the payload ceiling so that a
tensor-parallel PREFILL all-reduce (num_tokens x hidden x 2 B, e.g. 16 MB for a
2048-token chunk at hidden 4096) takes the custom two-shot kernel instead of
falling back to NCCL. That turns on a path which, at these sizes, has never run
before, so this covers both halves:

  * the WIRING -- the knob has to reach `CustomAllreduce`'s constructor through
    `CudaCommunicator`, which is the shipped entry point; a test that builds
    `CustomAllreduce` directly would pass while the env var did nothing.
  * the NUMERICS -- per-element random data, scored against an fp32 reduction.
    A constant fill is not sufficient: two-shot is reduce-scatter plus
    all-gather, so a chunk mapped to the wrong rank still reads back correct
    when every element holds the same value.

The actual `CustomAllreduce.custom_all_reduce` call is the anti-vacuity guard:
the run WITHOUT the knob must not engage it, while the run WITH the knob must.
Checking eligibility alone would not prove which communicator performed the
collective.

Run directly under torchrun, or via the pytest wrapper below:
    torchrun --nproc_per_node=8 tests/distributed/test_custom_all_reduce_large.py
"""

import os
import subprocess
import sys

import pytest
import torch

from vllm.distributed.device_communicators.cuda_communicator import (
    _custom_allreduce_max_size_bytes,
)
from vllm.platforms import current_platform

PAYLOAD_MB = 16
HIDDEN = 4096
DTYPE = torch.bfloat16
MiB = 1024 * 1024


def _is_a100_sm80_tp8_fully_connected() -> bool:
    """Return whether this is exactly the topology covered by this test."""
    if not current_platform.is_cuda() or current_platform.device_count() != 8:
        return False
    try:
        for device_id in range(8):
            capability = current_platform.get_device_capability(device_id)
            if capability is None or capability.to_int() != 80:
                return False
            if "A100" not in current_platform.get_device_name(device_id):
                return False
        physical_device_ids = [
            current_platform.visible_device_id_to_physical_device_id(device_id)
            for device_id in range(8)
        ]
        return current_platform.is_fully_connected(physical_device_ids)
    except Exception:
        return False


def _worker() -> int:
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    expect_custom = os.environ.get("EXPECT_CUSTOM") == "1"
    torch.accelerator.set_device_index(rank)
    # Held in a variable: inlining lets the context manager be collected and
    # every rank then dies on "Current vLLM config is not set".
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    ensure_model_parallel_initialized(world, 1)

    rows = PAYLOAD_MB * MiB // DTYPE.itemsize // HIDDEN
    torch.manual_seed(1234)
    payload = (torch.randn(rows, HIDDEN, device="cuda") * (rank + 1)).to(DTYPE)

    ref = payload.float()
    dist.all_reduce(ref)
    ca = get_tp_group().device_communicator.ca_comm
    if ca is None:
        got = tensor_model_parallel_all_reduce(payload)
        custom_calls = 0
    else:
        from unittest.mock import patch

        with patch.object(
            ca, "custom_all_reduce", wraps=ca.custom_all_reduce
        ) as custom_all_reduce:
            got = tensor_model_parallel_all_reduce(payload)
        custom_calls = custom_all_reduce.call_count
    torch.accelerator.synchronize()
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    took_custom = custom_calls == 1

    # bf16 rounding over 8 addends; the ring accumulates more of it than
    # two-shot does, so the bound has to admit the worse of the two.
    ok = took_custom == expect_custom and rel < 2e-2
    if rank == 0:
        cap = None if ca is None else ca.max_size
        print(
            f"custom_ar_calls={custom_calls} expected_custom={expect_custom} "
            f"max_size={cap} rel_err={rel:.3e} -> {'PASS' if ok else 'FAIL'}"
        )
    return 0 if ok else 1


@pytest.mark.skipif(
    not _is_a100_sm80_tp8_fully_connected(),
    reason="requires exactly 8 fully connected A100 (SM80) GPUs",
)
@pytest.mark.parametrize("knob_mb", [None, 32])
def test_large_payload_all_reduce(knob_mb: int | None):
    """Without the knob a 16 MB all-reduce must fall back to NCCL; with it,
    the custom kernel must take the payload AND stay numerically correct."""
    env = dict(os.environ)
    env["EXPECT_CUSTOM"] = "0" if knob_mb is None else "1"
    # Force both cases through the custom-AR-or-NCCL ordering under test.
    env["VLLM_USE_NCCL_SYMM_MEM"] = "0"
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    env["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
    if knob_mb is None:
        env.pop("VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE", None)
    else:
        env["VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE"] = str(knob_mb)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=8",
            os.path.abspath(__file__),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_custom_allreduce_cap_unset_preserves_default():
    assert _custom_allreduce_max_size_bytes(None) is None


def test_custom_allreduce_cap_converts_mib_to_bytes():
    assert _custom_allreduce_max_size_bytes(32) == 32 * MiB


@pytest.mark.parametrize("cap_mb", [0, -1])
def test_custom_allreduce_cap_must_be_positive(cap_mb: int):
    with pytest.raises(
        ValueError,
        match="VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE must be greater than 0",
    ):
        _custom_allreduce_max_size_bytes(cap_mb)


if __name__ == "__main__":
    sys.exit(_worker())
