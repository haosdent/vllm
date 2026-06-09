# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import ModuleType

import pytest

import vllm.utils.deep_gemm as deep_gemm


def _reset_deep_gemm_state(monkeypatch: pytest.MonkeyPatch) -> None:
    deep_gemm._import_deep_gemm.cache_clear()
    deep_gemm._import_vendored_deep_gemm.cache_clear()
    deep_gemm.is_deep_gemm_supported.cache_clear()
    deep_gemm.is_deep_gemm_e8m0_used.cache_clear()
    monkeypatch.delattr(
        deep_gemm.DeepGemmQuantScaleFMT,
        "_oracle_cache",
        raising=False,
    )

    impl_names = (
        "_cublaslt_gemm_nt_impl",
        "_fp8_gemm_nt_impl",
        "_fp8_einsum_impl",
        "_grouped_impl",
        "_grouped_masked_impl",
        "_grouped_fp4_impl",
        "_fp8_fp4_mqa_logits_impl",
        "_fp8_fp4_paged_mqa_logits_impl",
        "_get_paged_mqa_logits_metadata_impl",
        "_tf32_hc_prenorm_gemm_impl",
        "_get_mn_major_tma_aligned_tensor_impl",
        "_get_mk_alignment_for_contiguous_layout_impl",
        "_transform_sf_into_required_layout_impl",
    )
    for name in impl_names:
        monkeypatch.setattr(deep_gemm, name, None)


def test_lazy_init_uses_vendored_sparse_mla_symbols_when_external_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    _reset_deep_gemm_state(monkeypatch)

    def external_fp8_gemm_nt() -> None:
        pass

    def vendored_mqa_logits() -> None:
        pass

    def vendored_paged_mqa_logits() -> None:
        pass

    external_module = ModuleType("deep_gemm")
    external_module.fp8_gemm_nt = external_fp8_gemm_nt  # type: ignore[attr-defined]

    vendored_module = ModuleType("vllm.third_party.deep_gemm")
    vendored_module.fp8_fp4_mqa_logits = vendored_mqa_logits  # type: ignore[attr-defined]
    vendored_module.fp8_fp4_paged_mqa_logits = (  # type: ignore[attr-defined]
        vendored_paged_mqa_logits
    )

    def import_module(name: str) -> ModuleType:
        if name == "deep_gemm":
            return external_module
        if name == "vllm.third_party.deep_gemm":
            return vendored_module
        raise ImportError(name)

    monkeypatch.setattr(deep_gemm.importlib, "import_module", import_module)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)

    deep_gemm._lazy_init()

    assert deep_gemm._fp8_gemm_nt_impl is external_fp8_gemm_nt
    assert deep_gemm._fp8_fp4_mqa_logits_impl is vendored_mqa_logits
    assert deep_gemm._fp8_fp4_paged_mqa_logits_impl is vendored_paged_mqa_logits
