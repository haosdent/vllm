# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)


def _make_logits(
    lengths: torch.Tensor,
    max_len: int,
    *,
    sorted_asc: bool,
) -> torch.Tensor:
    num_rows = lengths.numel()
    logits = torch.full(
        (num_rows, max_len), float("-inf"), dtype=torch.float32, device="cuda"
    )
    for row, length in enumerate(lengths.tolist()):
        if length == 0:
            continue
        if sorted_asc:
            logits[row, :length] = torch.arange(
                length, dtype=torch.float32, device="cuda"
            )
        else:
            logits[row, :length] = torch.randn(
                length, dtype=torch.float32, device="cuda"
            )
    return logits


def _make_block_table(
    num_rows: int,
    num_blocks: int,
) -> torch.Tensor:
    return torch.arange(
        num_rows * num_blocks, dtype=torch.int32, device="cuda"
    ).reshape(num_rows, num_blocks)


def _run_case(
    *,
    name: str,
    top_k: int,
    max_len: int,
    lengths: torch.Tensor,
    block_size: int,
    block_table_blocks: int | None = None,
    sorted_asc: bool = False,
) -> None:
    num_rows = lengths.numel()
    assert num_rows > 32

    logits = _make_logits(lengths, max_len, sorted_asc=sorted_asc)
    req_id = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    num_blocks = (
        math.ceil(max_len / block_size)
        if block_table_blocks is None
        else block_table_blocks
    )
    block_table = _make_block_table(num_rows, num_blocks)

    output = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    valid_count = torch.empty((num_rows,), dtype=torch.int32, device="cuda")
    local_indices = torch.empty_like(output)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")

    torch.ops._C.persistent_topk(
        logits,
        lengths,
        local_indices,
        workspace,
        top_k,
        max_len,
    )
    torch.ops._C.persistent_topk_global(
        logits,
        lengths,
        output,
        valid_count,
        block_table,
        req_id,
        top_k,
        max_len,
        block_size,
    )
    expected_output, expected_valid_count = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        local_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=top_k,
        return_valid_counts=True,
    )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(
        valid_count,
        (output >= 0).sum(dim=1).to(torch.int32),
        rtol=0,
        atol=0,
        msg=f"{name}: output-derived valid_count mismatch",
    )
    torch.testing.assert_close(
        valid_count,
        expected_valid_count,
        rtol=0,
        atol=0,
        msg=f"{name}: triton valid_count mismatch",
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_persistent_topk_global_valid_count_parity() -> None:
    current_platform.import_kernels()

    num_rows = 40
    block_size = 64

    for top_k in (512, 1024, 2048):
        trivial_pattern = torch.tensor(
            [0, 1, top_k // 2, top_k], dtype=torch.int32, device="cuda"
        )
        _run_case(
            name=f"trivial_k{top_k}",
            top_k=top_k,
            max_len=top_k,
            lengths=trivial_pattern.repeat(num_rows // trivial_pattern.numel()),
            block_size=block_size,
            sorted_asc=True,
        )

        main_len = top_k + 256
        _run_case(
            name=f"main_k{top_k}",
            top_k=top_k,
            max_len=main_len,
            lengths=torch.full((num_rows,), main_len, dtype=torch.int32, device="cuda"),
            block_size=block_size,
            sorted_asc=True,
        )

    _run_case(
        name="oob_post_remap",
        top_k=512,
        max_len=640,
        lengths=torch.full((num_rows,), 640, dtype=torch.int32, device="cuda"),
        block_size=block_size,
        block_table_blocks=6,
        sorted_asc=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_persistent_buf_regrow_graph_safety() -> None:
    """Serve-free repro of the >512-running GSM8K illegal-memory-access. FULL
    decode cudagraphs bake the dsa_get_persistent_bufs buffer ADDRESSES into the
    captured fold kernel (req_id read, valid_count write) and the captured
    attention (seq_lens read). An eager step with num_rows above the current
    capacity regrows the buffers; if the old generation were freed, the
    allocator would reuse its memory and every later replay would read garbage
    req_id -> out-of-bounds block_table gather -> illegal access. The fix
    retains retired generations forever, keeping replays self-consistent
    (req_id is a static arange; valid_count is written+read within one replay).

    Sequence: capture at 48 rows -> healthy replay -> regrow to 616 -> spray
    same-size garbage allocations -> replay must stay clean and write the
    correct valid_count into the RETIRED buffer."""
    current_platform.import_kernels()
    from vllm.v1.attention.backends.mla import indexer as idx

    dev = torch.device("cuda")
    num_rows, top_k, max_len, block_size = 48, 512, 768, 64
    num_blocks = max_len // block_size

    logits = torch.randn(num_rows, max_len, dtype=torch.float32, device=dev)
    lengths = torch.full((num_rows,), max_len, dtype=torch.int32, device=dev)
    block_table = _make_block_table(num_rows, num_blocks)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)

    # Fresh state (other tests may have touched the module dicts).
    idx._dsa_req_id_buf.clear()
    idx._dsa_valid_count_buf.clear()
    idx._dsa_retired_bufs.clear()

    req_id, vc = idx.dsa_get_persistent_bufs(dev, num_rows)
    old_req_ptr = idx._dsa_req_id_buf[dev].data_ptr()
    old_vc_ptr = idx._dsa_valid_count_buf[dev].data_ptr()

    def run_fold():
        torch.ops._C.persistent_topk_global(
            logits, lengths, out, vc, block_table, req_id,
            top_k, max_len, block_size,
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_fold()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fold()

    vc.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert (vc == top_k).all(), "healthy replay broken"

    # The >512-rows eager step: regrow. Drop our views first (the serve holds
    # none across steps), then try hard to make the allocator reuse the old
    # blocks — with the keep-alive fix it must NOT be able to.
    del req_id, vc
    idx.dsa_get_persistent_bufs(dev, 616)
    junk = []
    for _ in range(256):
        t = torch.empty(512, dtype=torch.int32, device=dev)
        assert t.data_ptr() not in (old_req_ptr, old_vc_ptr), (
            "retired persistent buffer was freed and reused — captured graphs "
            "now read garbage req_id (the illegal-memory-access bug)"
        )
        t.fill_(0x7FFFFFF0)
        junk.append(t)

    g.replay()
    torch.cuda.synchronize()
    retired_vc = idx._dsa_retired_bufs[1]
    assert retired_vc.data_ptr() == old_vc_ptr
    assert (retired_vc[:num_rows] == top_k).all(), (
        f"replayed valid_count wrong: {retired_vc[:8].tolist()}"
    )
    print("persistent-buffer regrow graph-safety: PASS (retired gen kept alive; "
          "replay clean and self-consistent after regrow + allocation spray)")


def test_fold_decision_mixed_step_collision() -> None:
    """Serve-free repro of the Lever-A fold-DECISION coupling (the miss of the
    kernel-only parity above). The kernel is correct; the bug was the CHANNEL that
    tells the attention whether the indexer folded. Exercises the gate helper (the
    single source of truth) and the per-step publish/read semantics, including the
    MIXED-STEP COLLISION the old persistent count-keyed side-table got wrong.
    Pure Python, no CUDA."""
    import os

    # This test is about the fold decision, so exercise it with the fold enabled
    # (the gate's first term is the VLLM_GLM_TOPK_GLOBAL_FOLD env).
    os.environ["VLLM_GLM_TOPK_GLOBAL_FOLD"] = "1"
    from vllm.v1.attention.backends.mla.indexer import (
        _DSA_FOLD_FC_KEY,
        should_fold_topk_global,
    )

    # The gate is a pure function of per-step inputs. Clean pure-decode step in
    # the FilteredTopK regime folds; every collision-prone / unsupported step
    # type must NOT fold (so the attention keeps the always-correct convert).
    assert should_fold_topk_global(
        topk_tokens=512, num_rows=64, next_n=1,
        requires_padding=False, has_prefill=False,
    )
    for bad in (
        dict(has_prefill=True),        # mixed step (prefill present)
        dict(next_n=2),                # native spec decode
        dict(requires_padding=True),   # padded decode
        dict(num_rows=32),             # small batch -> per-row kernel, no fold
        dict(topk_tokens=256),         # not a FilteredTopK size
    ):
        kw = dict(
            topk_tokens=512, num_rows=64, next_n=1,
            requires_padding=False, has_prefill=False,
        )
        kw.update(bad)
        assert not should_fold_topk_global(**kw), f"must not fold: {bad}"

    # MIXED-STEP COLLISION: a mixed step whose token count equals a prior
    # pure-decode fold step's size N. Step 1 (pure decode at N) folds -> True;
    # step 2 (mixed, num_actual_toks == N) must NOT fold -> False.
    N = 64
    dec1 = should_fold_topk_global(
        topk_tokens=512, num_rows=N, next_n=1,
        requires_padding=False, has_prefill=False,
    )
    dec2 = should_fold_topk_global(
        topk_tokens=512, num_rows=N, next_n=1,
        requires_padding=False, has_prefill=True,
    )
    assert dec1 is True and dec2 is False

    # OLD channel (persistent dict keyed by count): step 1 writes [N]=True; step 2
    # reads the SAME key and gets the STALE True -> would wrongly skip the convert
    # and feed LOCAL indices to the gather (the out-of-bounds crash).
    old_side_table: dict[int, bool] = {}
    old_side_table[N] = dec1
    assert old_side_table.get(N, False) is True, (
        "sanity: the OLD count-keyed side-table reads the stale True (the bug)"
    )

    # NEW channel (per-step forward_context.additional_kwargs): each step gets a
    # FRESH dict; the indexer publishes THIS step's decision; the attention reads
    # it, defaulting an ABSENT key to False. No count key, no persistence -> the
    # collision cannot happen.
    fc_step2 = {_DSA_FOLD_FC_KEY: dec2}
    assert fc_step2.get(_DSA_FOLD_FC_KEY, False) is False, (
        "the FIX reads step 2's real False -> convert runs"
    )
    assert {}.get(_DSA_FOLD_FC_KEY, False) is False, "absent key is the safe default"

    print(
        "fold-decision mixed-step collision repro: PASS "
        "(gate correct; OLD side-table reads stale True; fc channel reads fresh False)"
    )


if __name__ == "__main__":
    # Decision-layer repro first (pure Python, no CUDA).
    test_fold_decision_mixed_step_collision()
    if not current_platform.is_cuda():
        raise SystemExit("persistent_topk_global parity requires CUDA")
    test_persistent_topk_global_valid_count_parity()
    test_persistent_buf_regrow_graph_safety()
