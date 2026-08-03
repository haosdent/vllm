# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch._C._autograd import DeviceType

import vllm.distributed.device_communicators.custom_all_reduce as custom_all_reduce

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("device", "current_device_index", "expected_device_index"),
    [
        (torch.device("cuda:3"), 7, 3),
        (torch.device("cuda"), 7, 7),
    ],
)
def test_mnnvl_init_skips_without_multicast_support(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    current_device_index: int,
    expected_device_index: int,
):
    capability_queries = []

    class FakeSymmetricMemory:
        @staticmethod
        def has_multicast_support(device_type, device_index):
            capability_queries.append((device_type, device_index))
            return False

    def fail_empty(*args, **kwargs):
        pytest.fail("symmetric-memory allocation must be skipped")

    monkeypatch.setattr(
        custom_all_reduce,
        "torch_symm_mem",
        SimpleNamespace(_SymmetricMemory=FakeSymmetricMemory, empty=fail_empty),
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "current_device_index",
        lambda: current_device_index,
    )

    communicator = custom_all_reduce.CustomAllreduce.__new__(
        custom_all_reduce.CustomAllreduce
    )
    communicator.disabled = True
    communicator._ptr = 0
    communicator.device = device
    communicator._init_mnnvl_buffer(stage_size=1024)

    assert capability_queries == [(DeviceType.CUDA, expected_device_index)]


def test_mnnvl_init_continues_with_multicast_support(
    monkeypatch: pytest.MonkeyPatch,
):
    allocations = []

    class FakeSymmetricMemory:
        @staticmethod
        def has_multicast_support(device_type, device_index):
            return True

    def stop_after_empty(*args, **kwargs):
        allocations.append((args, kwargs))
        raise RuntimeError("stop after proving allocation was attempted")

    monkeypatch.setattr(
        custom_all_reduce,
        "torch_symm_mem",
        SimpleNamespace(_SymmetricMemory=FakeSymmetricMemory, empty=stop_after_empty),
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )

    communicator = custom_all_reduce.CustomAllreduce.__new__(
        custom_all_reduce.CustomAllreduce
    )
    communicator.disabled = True
    communicator._ptr = 0
    communicator.device = torch.device("cuda:3")
    communicator._init_mnnvl_buffer(stage_size=1024)

    assert len(allocations) == 1
