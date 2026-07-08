# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MLA decode preprocessing: split kv_lora -> (kv_c, k_pe),
kv_a_layernorm(kv_c), and GPT-J interleaved RoPE on q_pe and k_pe, in one pass.

This replaces the torch.compile glue that the unfused MLA forward emits for the
GLM-DSA decode path (kv_lora.split + kv_a_layernorm + rotary_emb on
q[..., qk_nope:] and k_pe). Selected by VLLM_GLM_ATTN_PREP_OVERLAP >= 2 on the
GLM-DSA decode path (see MultiHeadLatentAttentionWrapper).

Numerics are matched to the unfused reference:
  - kv_a_layernorm matches csrc/.../layernorm_kernels.cu:
        out = (scalar_t)(x * rsqrt(mean(x^2) + eps)) * weight
    i.e. (x * rrms) is rounded to the I/O dtype BEFORE the weight multiply.
  - RoPE is GPT-J INTERLEAVED (is_neox_style=False) over the last rope_dim dims;
    cos/sin come from a (max_pos, rope_dim) fp32 cos_sin_cache laid out as
    [cos(rope_dim//2) | sin(rope_dim//2)] (DeepseekScalingRotaryEmbedding cache).
    out[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
    out[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round(x, IS_BF16: tl.constexpr):
    # Round an fp32 value to bf16/fp16 and back to fp32 (emulates a torch
    # low-precision intermediate). Kept in fp32 registers between ops.
    if IS_BF16:
        return x.to(tl.bfloat16).to(tl.float32)
    return x.to(tl.float16).to(tl.float32)


@triton.jit
def _interleaved_rope(x_even, x_odd, cos, sin, IS_BF16: tl.constexpr):
    """GPT-J interleaved RoPE matching ApplyRotaryEmb.forward_static (the
    forward_native path) in bf16/fp16 arithmetic: cos/sin and x are first
    rounded to the output dtype, then each product and the add/sub are rounded
    to the output dtype, exactly as torch evaluates `x1*cos - x2*sin` on
    low-precision tensors. Returns (r_even, r_odd) rounded to the output dtype
    (still represented as fp32 values; the caller stores them as out dtype)."""
    # Round inputs to the low-precision dtype (cos.to(x.dtype) in the reference)
    # then round each product and the add/sub to mirror torch's per-op rounding.
    xe = _round(x_even, IS_BF16)
    xo = _round(x_odd, IS_BF16)
    c = _round(cos, IS_BF16)
    s = _round(sin, IS_BF16)
    r_even = _round(_round(xe * c, IS_BF16) - _round(xo * s, IS_BF16), IS_BF16)
    r_odd = _round(_round(xo * c, IS_BF16) + _round(xe * s, IS_BF16), IS_BF16)
    return r_even, r_odd


@triton.jit
def _fused_mla_rope_kernel(
    pos_ptr,
    # q (in/out, in-place rope on the last ROPE_DIM dims of each head)
    q_ptr,
    q_stride0,
    q_stride1,
    # kv_lora (input, split into kv_c[:KV_LORA] and k_pe[KV_LORA:])
    kv_lora_ptr,
    kv_lora_stride0,
    # kv_c_normed output (bf16/fp16), (T, KV_LORA)
    kv_c_out_ptr,
    kv_c_out_stride0,
    kv_a_weight_ptr,
    kv_a_eps,
    # k_pe output, (T, 1, ROPE_DIM)
    k_pe_out_ptr,
    k_pe_out_stride0,
    # cos_sin cache (fp32), (max_pos, ROPE_DIM) = [cos(HALF) | sin(HALF)]
    cos_sin_ptr,
    cos_sin_stride0,
    NUM_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_LORA: tl.constexpr,
    KV_LORA_BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    HALF: tl.constexpr = ROPE_DIM // 2
    tok_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    half_off = tl.arange(0, HALF)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride0 + half_off).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride0 + half_off + HALF).to(tl.float32)

    if task < NUM_HEADS:
        # ---- q head RoPE (in-place on the last ROPE_DIM dims) ----
        head_idx = task
        rot_base = q_ptr + tok_idx * q_stride0 + head_idx * q_stride1 + NOPE_DIM
        x_even = tl.load(rot_base + half_off * 2).to(tl.float32)
        x_odd = tl.load(rot_base + half_off * 2 + 1).to(tl.float32)
        r_even, r_odd = _interleaved_rope(x_even, x_odd, cos, sin, IS_BF16)
        tl.store(rot_base + half_off * 2, r_even.to(rot_base.dtype.element_ty))
        tl.store(rot_base + half_off * 2 + 1, r_odd.to(rot_base.dtype.element_ty))
    else:
        # ---- kv_a_layernorm(kv_c) + k_pe RoPE ----
        kv_base = kv_lora_ptr + tok_idx * kv_lora_stride0
        kv_block = tl.arange(0, KV_LORA_BLOCK)
        kv_mask = kv_block < KV_LORA
        x = tl.load(kv_base + kv_block, mask=kv_mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / KV_LORA
        rrms = tl.rsqrt(variance + kv_a_eps)
        w = tl.load(kv_a_weight_ptr + kv_block, mask=kv_mask, other=0.0).to(tl.float32)
        # Match the rms_norm reference: round (x*rrms) to the output dtype
        # BEFORE *w (csrc/layernorm_kernels.cu; verified bit-exact vs ir.rms_norm).
        y = _round(x * rrms, IS_BF16) * w
        tl.store(
            kv_c_out_ptr + tok_idx * kv_c_out_stride0 + kv_block,
            y.to(kv_c_out_ptr.dtype.element_ty),
            mask=kv_mask,
        )

        # k_pe = kv_lora[..., KV_LORA:], interleaved RoPE, write to k_pe_out.
        kpe_in = kv_base + KV_LORA
        x_even = tl.load(kpe_in + half_off * 2).to(tl.float32)
        x_odd = tl.load(kpe_in + half_off * 2 + 1).to(tl.float32)
        r_even, r_odd = _interleaved_rope(x_even, x_odd, cos, sin, IS_BF16)
        kpe_out = k_pe_out_ptr + tok_idx * k_pe_out_stride0
        tl.store(kpe_out + half_off * 2, r_even.to(k_pe_out_ptr.dtype.element_ty))
        tl.store(kpe_out + half_off * 2 + 1, r_odd.to(k_pe_out_ptr.dtype.element_ty))


def fused_mla_split_rmsnorm_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    kv_lora: torch.Tensor,
    kv_a_weight: torch.Tensor,
    kv_a_eps: float,
    cos_sin_cache: torch.Tensor,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MLA decode preprocess.

    Inputs:
        positions   : (T,) int
        q           : (T, num_heads, qk_head_dim) — RoPE applied IN-PLACE to the
                      last qk_rope_head_dim dims of each head.
        kv_lora     : (T, kv_lora_rank + qk_rope_head_dim)
        kv_a_weight : (kv_lora_rank,) RMSNorm weight
        cos_sin_cache: (max_pos, qk_rope_head_dim) fp32, [cos|sin] halves
    Returns:
        kv_c_normed : (T, kv_lora_rank), same dtype as kv_lora
        k_pe        : (T, 1, qk_rope_head_dim), same dtype as kv_lora
    (q is mutated in place and is the same tensor the caller passed in.)
    """
    assert positions.ndim == 1
    assert q.ndim == 3
    assert kv_lora.ndim == 2
    assert q.stride(-1) == 1 and kv_lora.stride(-1) == 1
    assert kv_a_weight.is_contiguous()

    num_tokens = positions.shape[0]
    num_heads = q.shape[1]
    qk_head_dim = q.shape[2]
    assert qk_head_dim == qk_nope_head_dim + qk_rope_head_dim
    assert kv_lora.shape[1] == kv_lora_rank + qk_rope_head_dim

    kv_c_normed = torch.empty(
        (num_tokens, kv_lora_rank), dtype=kv_lora.dtype, device=kv_lora.device
    )
    k_pe = torch.empty(
        (num_tokens, 1, qk_rope_head_dim), dtype=kv_lora.dtype, device=kv_lora.device
    )
    if num_tokens == 0:
        return kv_c_normed, k_pe

    assert kv_lora.dtype in (torch.bfloat16, torch.float16), (
        f"fused MLA rope supports bf16/fp16 only, got {kv_lora.dtype}"
    )
    is_bf16 = kv_lora.dtype == torch.bfloat16
    kv_lora_block = triton.next_power_of_2(kv_lora_rank)
    _fused_mla_rope_kernel[(num_tokens, num_heads + 1)](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        kv_lora,
        kv_lora.stride(0),
        kv_c_normed,
        kv_c_normed.stride(0),
        kv_a_weight,
        kv_a_eps,
        k_pe,
        k_pe.stride(0),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        NUM_HEADS=num_heads,
        NOPE_DIM=qk_nope_head_dim,
        ROPE_DIM=qk_rope_head_dim,
        KV_LORA=kv_lora_rank,
        KV_LORA_BLOCK=kv_lora_block,
        IS_BF16=is_bf16,
        num_warps=4,
    )
    return kv_c_normed, k_pe
