# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA attention for CUDA SM8x devices."""

from functools import lru_cache

import torch

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    FlashMLASchedMeta,
)


@lru_cache
def _warmup_ampere_mqa_logits(
    num_heads: int,
    head_dim: int,
    block_sizes: tuple[int, ...],
    device: torch.device,
) -> None:
    """Prime SM8x MQA autotuning before memory profiling captures graphs."""
    from vllm.v1.attention.ops.mqa_logits_triton import (
        warmup_fp8_mqa_logits_triton,
        warmup_fp8_paged_mqa_logits_triton,
    )

    warmup_fp8_mqa_logits_triton(num_heads, head_dim, device)
    for block_size in block_sizes:
        warmup_fp8_paged_mqa_logits_triton(num_heads, head_dim, block_size, device)


class DeepseekV4AmpereSparseSWAMetadataBuilder(
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder
):
    """Build ragged metadata without allocating unused FlashMLA schedulers."""

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        return dict.fromkeys((_LAYER_TYPE_SWAONLY, _LAYER_TYPE_C4A, _LAYER_TYPE_C128A))


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """Run the platform-neutral ragged Triton path on CUDA SM8x."""

    backend_cls = DeepseekV4AmpereMLASparseBackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.indexer is None or self.indexer.use_fp4_kv:
            return

        topk_buffer = self.indexer.topk_indices_buffer
        if topk_buffer is None:
            return

        cache_config = self.indexer.vllm_config.cache_config
        assert cache_config is not None
        configured_block_size = cache_config.block_size
        block_sizes = tuple(sorted({64, 256, configured_block_size}))
        _warmup_ampere_mqa_logits(
            self.indexer.n_head,
            self.indexer.head_dim,
            block_sizes,
            topk_buffer.device,
        )
