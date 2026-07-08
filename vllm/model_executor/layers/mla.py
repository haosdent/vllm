# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import (
    aux_stream,
    direct_register_custom_op,
    is_quantized_kv_cache,
)


# Persistent output buffers for the overlap ops when they run OUTSIDE cudagraph
# capture (eager mixed steps / piecewise-split contexts). The overlap ops are
# splitting ops, and piecewise cudagraph pieces bake their input ADDRESSES at
# capture (CUDAGraphWrapper copies nothing; addresses are only checked under
# VLLM_LOGGING_LEVEL=DEBUG) — so a splitting op feeding a replayed piece must
# return address-stable tensors, like sparse_attn_indexer's persistent topk
# buffer. Returning fresh allocations here made every <=max-capture-size mixed
# step (prefix-cache-hit tail chunks) feed stale capture-time addresses into
# the downstream piece (GSM8K cache-hit repeat 0.15 vs ~0.95). Inside FULL
# capture fresh tensors are graph-internal and fine — skipping the copy there
# keeps the decode hot path unchanged. Sized to the max capture size: larger
# eager steps run their pieces eagerly too, so fresh outputs are safe.
_dsa_overlap_out_bufs: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def _dsa_overlap_stabilize(
    self,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
):
    """Copy overlap-op outputs into persistent per-device buffers and return
    address-stable slices (see _dsa_overlap_out_bufs). Falls back to the fresh
    tensors when the step exceeds the piecewise-replayable size."""
    n = q.shape[0]
    cap = self._dsa_out_buf_tokens
    if n == 0 or n > cap:
        return q, kv_c_normed, k_pe
    dev = q.device
    bufs = _dsa_overlap_out_bufs.get(dev)
    if bufs is None:
        # q is stored flat per token: the base op returns 2D [n, H*D], the
        # prep op 3D [n, H, D] — same storage, viewed per call.
        bufs = (
            torch.empty((cap, q[0].numel()), dtype=q.dtype, device=dev),
            torch.empty(
                (cap, self.kv_lora_rank), dtype=q.dtype, device=dev
            ),
            torch.empty(
                (cap, 1, self.qk_rope_head_dim), dtype=q.dtype, device=dev
            ),
        )
        _dsa_overlap_out_bufs[dev] = bufs
    q_out = bufs[0][:n].view(q.shape)
    q_out.copy_(q)
    kv_out = None
    if kv_c_normed is not None:
        kv_out = bufs[1][:n]
        kv_out.copy_(kv_c_normed)
    kpe_out = None
    if k_pe is not None:
        kpe_out = bufs[2][:n]
        kpe_out.copy_(k_pe)
    return q_out, kv_out, kpe_out


def _dsa_attn_overlap_impl(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    q_out_dim: int,
) -> torch.Tensor:
    """Run q_b_proj (default stream) concurrently with the whole DSA indexer
    (aux stream), then return q. The indexer writes the shared topk buffer as a
    side effect. This op is OPAQUE to torch.compile and is NOT a splitting op,
    so the FULL decode cudagraph captures the multi-stream overlap once and
    replays it cheaply (mirrors the shared-experts overlap; no breakable
    cudagraph, no eager-break, no un-capture penalty)."""
    self = get_forward_context().no_compile_layers[prefix]
    idx = self.indexer
    # Fork ONLY inside cudagraph capture: capture bakes the fork/join edges into
    # the graph, so replays are race-free (validated by every decode eval).
    # EAGER executions (mixed prefill steps / uncaptured shapes — e.g. the tiny
    # tail chunks of prefix-cache-hit prompts) raced and produced garbage first
    # tokens (GSM8K cache-hit repeat 0.005-0.59 vs 0.94 with the fork off;
    # bis1/bis2 attribution serves, 2026-07-07). Eager steps are
    # prefill-dominated, so the overlap gains nothing there — run sequentially
    # (maybe_execute_in_parallel with aux_stream=None).
    capturing = torch.cuda.is_current_stream_capturing()
    stream = aux_stream() if capturing else None
    q, _ = maybe_execute_in_parallel(
        lambda: self.q_b_proj(q_c)[0],
        lambda: idx.forward_via_function(
            hidden_states, q_c, positions, self.indexer_rope_emb
        ),
        self._dsa_overlap_events[0],
        self._dsa_overlap_events[1],
        stream,
    )
    if not capturing:
        # Address-stable output for the downstream piecewise piece; see
        # _dsa_overlap_out_bufs. FULL-capture returns stay fresh (graph-
        # internal), keeping the decode hot path copy-free.
        q, _, _ = _dsa_overlap_stabilize(self, q)
    return q


def _dsa_attn_overlap_fake(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    q_out_dim: int,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], q_out_dim), dtype=hidden_states.dtype
    )


direct_register_custom_op(
    op_name="dsa_attn_overlap",
    op_func=_dsa_attn_overlap_impl,
    mutates_args=["hidden_states"],
    fake_impl=_dsa_attn_overlap_fake,
)


def _dsa_attn_overlap_prep_impl(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    kv_lora: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    qk_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Captured multi-stream overlap (VLLM_GLM_ATTN_PREP_OVERLAP): the DEFAULT
    stream runs q_b_proj AND the topk-independent attention prep (kv_a_layernorm
    + the MLA rope on q and k_pe); the aux stream runs the whole DSA indexer.
    q_b_proj alone is shorter than the indexer, so the default stream used to
    stall; filling it with the kv_a_layernorm + rope that previously ran
    serialized AFTER this op hides more of the indexer. Returns
    (q[3d, roped], kv_c_normed, k_pe[3d, roped]) so the caller skips the serial
    prep. Same opaque/captured-cudagraph properties as dsa_attn_overlap (NOT a
    splitting op, NOT eager-break). ONE fork / ONE join."""
    fc = get_forward_context()
    self = fc.no_compile_layers[prefix]
    idx = self.indexer
    # Fork only under capture — see _dsa_attn_overlap_impl (the eager fork
    # raced on uncaptured shapes; sequential there, overlap kept in graphs).
    capturing = torch.cuda.is_current_stream_capturing()
    stream = aux_stream() if capturing else None

    def _main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_b_proj(q_c)[0].view(-1, num_heads, qk_head_dim)
        if self.attn_prep_mode >= 2:
            # ONE fused Triton kernel (split + kv_a_layernorm + interleaved
            # rope), fusion-parity with the baseline inductor kernel. Call the
            # PLAIN function, NOT the registered op, to avoid nesting custom
            # ops. Returns fresh kv_c_normed + k_pe (no aliasing); q roped
            # in place.
            from vllm.models.deepseek_v4.common.ops.fused_mla_rope import (
                fused_mla_split_rmsnorm_rope,
            )

            kv_c_normed, k_pe_out = fused_mla_split_rmsnorm_rope(
                positions,
                q,
                kv_lora,
                self.kv_a_layernorm.weight.data,
                self.kv_a_layernorm.variance_epsilon,
                self.rotary_emb.cos_sin_cache,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
            )
            if self.attn_prep_mode >= 3:
                # Hoist the topk-INDEPENDENT KV-cache write into the overlap so
                # it runs concurrent with the indexer instead of serialized
                # after it (MLAAttention skips its own write via
                # skip_kv_cache_update). The attention op stays ordered after
                # this write through its data dependency on kv_c_normed.
                attn = self.mla_attn
                slot_mapping = fc.slot_mapping.get(attn.layer_name)
                if slot_mapping is not None:
                    attn.impl.do_kv_cache_update(
                        kv_c_normed,
                        k_pe_out,
                        attn.kv_cache,
                        slot_mapping,
                        attn.kv_cache_dtype,
                        attn._k_scale,
                    )
                if (
                    self.attn_prep_mode == 4
                    and attn.use_precomputed_mqa
                    and slot_mapping is not None
                    and q.shape[0] <= attn.mqa_buf_capacity
                ):
                    # 4a: hoist the topk-INDEPENDENT W_UK-absorb bmm into the
                    # overlap, writing directly into the attn layer's PERSISTENT
                    # mqa buffer — zero graph-pool allocation on the main stream,
                    # so nothing allocates concurrently with the aux-stream
                    # indexer during capture (that concurrent main+aux pool
                    # allocation aliased a live tensor and corrupted output).
                    # concat+quant stay eager in forward_impl, which reads the
                    # buffer across the eager-break. Gate on slot_mapping (present
                    # in every real capture incl. PIECEWISE, absent in the profile
                    # run), matching the KV-write above. Skipped when tokens
                    # exceed the buffer (eager prefill recomputes inline).
                    attn.precompute_mqa_bmm_into_buffer(q)
            return q, kv_c_normed, k_pe_out
        # mode 1: EAGER prep (kept for A/B; regresses due to inductor-fusion
        # loss). Fresh k_pe copy: the in-place rope must not mutate/alias the
        # kv_lora input across the custom-op boundary.
        kv_c = kv_lora[..., :kv_lora_rank]
        k_pe = kv_lora[..., kv_lora_rank:].unsqueeze(1).contiguous()
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe_out = k_pe
        if self.rotary_emb is not None:
            q[..., qk_nope_head_dim:], k_pe_out = self.rotary_emb(
                positions, q[..., qk_nope_head_dim:], k_pe
            )
        return q, kv_c_normed, k_pe_out

    (q, kv_c_normed, k_pe_out), _ = maybe_execute_in_parallel(
        _main,
        lambda: idx.forward_via_function(
            hidden_states, q_c, positions, self.indexer_rope_emb
        ),
        self._dsa_overlap_events[0],
        self._dsa_overlap_events[1],
        stream,
    )
    if not capturing:
        # Address-stable outputs for the downstream piecewise piece; see
        # _dsa_overlap_out_bufs. FULL-capture returns stay fresh (graph-
        # internal), keeping the decode hot path copy-free.
        q, kv_c_normed, k_pe_out = _dsa_overlap_stabilize(
            self, q, kv_c_normed, k_pe_out
        )
    return q, kv_c_normed, k_pe_out


def _dsa_attn_overlap_prep_fake(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    kv_lora: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    qk_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = hidden_states.shape[0]
    q = hidden_states.new_empty((n, num_heads, qk_head_dim), dtype=hidden_states.dtype)
    kv_c_normed = kv_lora.new_empty((n, kv_lora_rank), dtype=kv_lora.dtype)
    k_pe = kv_lora.new_empty((n, 1, qk_rope_head_dim), dtype=kv_lora.dtype)
    return q, kv_c_normed, k_pe


direct_register_custom_op(
    op_name="dsa_attn_overlap_prep",
    op_func=_dsa_attn_overlap_prep_impl,
    mutates_args=["hidden_states"],
    fake_impl=_dsa_attn_overlap_prep_fake,
)


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

        # Captured multi-stream overlap of q_b_proj || the DSA indexer
        # (opt-in via VLLM_GLM_DSA_V4_ATTN). Implemented as a captured opaque
        # op (NOT a splitting op, NOT eager-break) so the FULL decode cudagraph
        # captures the overlap and replays it cheaply.
        self.use_glm_dsa_v4 = (
            envs.VLLM_GLM_DSA_V4_ATTN
            and self.is_sparse
            and not self.skip_topk
            and self.indexer is not None
            and getattr(self.indexer, "use_fused_indexer_k", False)
            and self.q_b_proj is not None
            and self.q_lora_rank is not None
            and current_platform.is_cuda_alike()
        )
        # Extend the captured q_b_proj||indexer overlap: also run the MLA prep
        # (kv_a_layernorm + rope) on the DEFAULT stream, concurrent with the
        # indexer on the aux stream, moving it off the post-op serial path.
        # mode 1 = EAGER prep (loses inductor fusion, regresses); mode 2 = one
        # fused_mla_split_rmsnorm_rope Triton kernel (fusion-parity, intended);
        # mode 3 = mode 2 + hoist the topk-independent KV-cache write into the
        # overlap (fills more of the indexer stall); mode 4 = mode 3 + hoist the
        # topk-independent decode-mqa W_UK-absorb bmm (requires
        # VLLM_GLM_ATTN_PREP_MQA_BMM_ONLY=1, "4a"; without it mode 4 == mode 3).
        # Supersedes decode-overlap; off (0) => PR#6 path unchanged.
        self.attn_prep_mode = (
            int(envs.VLLM_GLM_ATTN_PREP_OVERLAP) if self.use_glm_dsa_v4 else 0
        )
        if self.attn_prep_mode >= 2:
            # The fused kernel is specific to GLM's interleaved DeepseekScaling
            # rope (is_neox_style=False, rotary_dim == qk_rope_head_dim); fall
            # back to eager prep if that gate is not met.
            fused_ok = (
                self.rotary_emb is not None
                and not getattr(self.rotary_emb, "is_neox_style", True)
                and getattr(self.rotary_emb, "rotary_dim", None)
                == self.qk_rope_head_dim
                and hasattr(self.rotary_emb, "cos_sin_cache")
            )
            if not fused_ok:
                self.attn_prep_mode = 1
        self.use_glm_attn_prep_overlap = self.attn_prep_mode >= 1
        if self.attn_prep_mode == 4:
            # 4a (VLLM_GLM_ATTN_PREP_MQA_BMM_ONLY=1): allocate the persistent
            # bmm-output buffer the prep op writes and forward_impl reads across
            # the eager-break. Without the flag the hoist stays off (setup is a
            # no-op) and mode 4 behaves as mode 3.
            self.mla_attn.setup_precomputed_mqa(
                bmm_only=envs.VLLM_GLM_ATTN_PREP_MQA_BMM_ONLY
            )
        if self.use_glm_dsa_v4:
            self._dsa_overlap_events = [torch.cuda.Event(), torch.cuda.Event()]
            # Pre-create the global aux stream now: the fork only engages inside
            # cudagraph capture, and lazily creating a CUDA stream mid-capture
            # is illegal.
            aux_stream()
            # Piecewise-replayable steps never exceed the max capture size;
            # larger eager steps keep fresh outputs (see _dsa_overlap_stabilize).
            self._dsa_out_buf_tokens = (
                get_current_vllm_config().compilation_config.max_cudagraph_capture_size
                or 0
            )
            static_fwd_ctx = (
                get_current_vllm_config().compilation_config.static_forward_context
            )
            if prefix in static_fwd_ctx:
                raise ValueError(f"Duplicate layer name: {prefix}")
            static_fwd_ctx[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        prep_done = False

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            kv_c_normed = None
            if self.use_glm_attn_prep_overlap:
                # q_b_proj + kv_a_layernorm + MLA rope (default stream) ||
                # whole DSA indexer (aux stream), captured by the FULL decode
                # cudagraph. Returns q(3d, roped), kv_c_normed, k_pe(3d, roped),
                # so the serial prep below is skipped.
                q, kv_c_normed, k_pe = torch.ops.vllm.dsa_attn_overlap_prep(
                    hidden_states,
                    q_c,
                    kv_lora,
                    positions,
                    self.prefix,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    self.qk_nope_head_dim,
                    self.num_heads,
                    self.qk_head_dim,
                )
                prep_done = True
            elif self.use_glm_dsa_v4:
                # q_b_proj || whole DSA indexer, captured by the FULL decode
                # cudagraph (the indexer writes the topk buffer as a side
                # effect, so the separate self.indexer(...) call below is
                # skipped).
                q = torch.ops.vllm.dsa_attn_overlap(
                    hidden_states,
                    q_c,
                    positions,
                    self.prefix,
                    self.num_heads * self.qk_head_dim,
                )
            else:
                q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]
            kv_c_normed = None

        if not prep_done:
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            if kv_c_normed is None:
                kv_c_normed = self.kv_a_layernorm(kv_c)

            q = q.view(-1, self.num_heads, self.qk_head_dim)
            # Add head dim of 1 to k_pe
            k_pe = k_pe.unsqueeze(1)

            if self.rotary_emb is not None:
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )

        if (
            self.indexer
            and self.is_sparse
            and not self.skip_topk
            and not self.use_glm_dsa_v4
        ):
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            # mode 3 already wrote the KV cache inside the overlap op.
            skip_kv_cache_update=(self.attn_prep_mode >= 3),
        )

        return self.o_proj(attn_out)[0]
