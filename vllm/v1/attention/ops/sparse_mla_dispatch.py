# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime dispatch for the sparse-MLA SM80/SM121 kernels.

Default is the Triton/CUDA path (what PR #38476 ships). Setting the env
flag ``VLLM_USE_TILELANG_SPARSE=1`` swaps the swappable kernels:

* sparse MLA attention: ``triton_sparse_mla_attention``
  → ``tilelang_sparse_mla_attention``
* FP8 MQA logits prefill: ``fp8_mqa_logits_triton``
  → ``tilelang_fp8_mqa_logits``
* top-K: ``torch.ops._C.persistent_topk`` → ``tilelang_persistent_topk``
* FP8 paged MQA decode: unchanged (no TileLang variant on this branch).

This is intentionally an A/B switch for benchmarking. It's not part of
the SM80/SM121 PR — see ``TILELANG_FINDINGS.md`` for the recommendation
to stay on Triton for that PR.
"""

import torch

import vllm.envs as envs
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    triton_sparse_mla_attention,
)

# Per-kernel flags. `VLLM_USE_TILELANG_SPARSE` is the master switch;
# the three per-kernel vars override it if set, otherwise inherit.
_USE_TL_MLA = envs.VLLM_USE_TILELANG_SPARSE_MLA
_USE_TL_MQA = envs.VLLM_USE_TILELANG_SPARSE_MQA
_USE_TL_TOPK = envs.VLLM_USE_TILELANG_SPARSE_TOPK


def is_tilelang_mla_enabled() -> bool:
    return _USE_TL_MLA


if _USE_TL_MQA:
    from vllm.v1.attention.ops.tilelang_mqa_logits import tilelang_fp8_mqa_logits
if _USE_TL_MLA:
    from vllm.v1.attention.ops.tilelang_sparse_mla_kernel import (
        tilelang_sparse_mla_attention,
    )
if _USE_TL_TOPK:
    from vllm.v1.attention.ops.tilelang_topk import tilelang_persistent_topk


def sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse-MLA attention dispatch. See module docstring for what each
    path does."""
    if _USE_TL_MLA:
        return tilelang_sparse_mla_attention(
            q,
            kv,
            indices,
            sm_scale,
            num_kv_splits=num_kv_splits,
            sm_count=sm_count,
        )
    return triton_sparse_mla_attention(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
        sm_count=sm_count,
    )


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """FP8 MQA logits prefill dispatch."""
    if _USE_TL_MQA:
        return tilelang_fp8_mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
        )
    return fp8_mqa_logits_triton(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
    )


def persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
    topk_tokens: int,
    max_seq_len: int,
) -> None:
    """Top-K dispatch. CUDA `torch.ops._C.persistent_topk` (default) and
    TileLang `tilelang_persistent_topk` produce the same set of indices
    after the stage-2 fix on this branch."""
    if _USE_TL_TOPK:
        # Decode path passes seq_lens as 2-D (B, next_n); flatten for TileLang
        # which expects 1-D ends. CUDA op handles both shapes internally.
        seq_lens_1d = seq_lens.flatten() if seq_lens.ndim > 1 else seq_lens
        tilelang_persistent_topk(logits, seq_lens_1d, topk_indices, topk_tokens)
        return
    torch.ops._C.persistent_topk(
        logits,
        seq_lens,
        topk_indices,
        topk_workspace,
        topk_tokens,
        max_seq_len,
    )


def warmup_indexer_kernels(
    num_heads: int, head_dim: int, block_size: int | None, device: torch.device
) -> None:
    """Prime the autotune caches for the kernel variant in use. For Triton
    this is the upstream autotune sweep; for TileLang the JIT compile of
    each shape bucket is implicit on first call (cached afterward)."""
    # The MQA-logits prefill kernel is the only one this function primes
    # (paged is always Triton). Skip the prefill Triton warmup iff TileLang
    # is taking over MQA; the paged Triton warmup runs either way.
    if not _USE_TL_MQA:
        warmup_fp8_mqa_logits_triton(
            num_heads=num_heads, head_dim=head_dim, device=device
        )
    if block_size is not None:
        warmup_fp8_paged_mqa_logits_triton(
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
        )
