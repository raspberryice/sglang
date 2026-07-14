"""GPU engine smoke-test for top-p mask replay capture (Phase 0 gate).

Unlike test_top_p_replay_logprob.py (pure-CPU unit test of the helpers), this
launches a real SGLang offline Engine, sends a top_p<1.0 request with
``return_top_p_token_ids``, and asserts the kept nucleus round-trips through
meta_info end to end:

    meta_info["top_p_token_ids"]      base64 int32 flat kept-id array
    meta_info["top_p_token_offsets"]  base64 int32 ragged offsets, len == n_tokens+1

Validated invariants:
  * offsets[0] == 0, offsets[-1] == len(token_ids), len(offsets) == n_out + 1
  * every response token's kept set is non-empty
  * the sampled token is a member of its own kept set (force-keep guarantee)

REQUIRES A GPU. Set TEST_TOP_P_MODEL to a small local model path, e.g.:

    TEST_TOP_P_MODEL=/shared/models/Qwen2.5-0.5B-Instruct \
        python test/srt/test_top_p_replay_engine_smoke.py

Skips (exit 0) if no GPU or no model path is provided, so it is safe to wire
into CI on CPU-only shards without failing.
"""

import base64
import os
import sys

import numpy as np


def _decode_int32(b64: str) -> list:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32).tolist()


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available.")
        return 0

    model_path = os.environ.get("TEST_TOP_P_MODEL")
    if not model_path:
        print("SKIP: set TEST_TOP_P_MODEL to a local model path to run this smoke-test.")
        return 0

    import sglang as sgl

    top_p = 0.95
    engine = sgl.Engine(model_path=model_path, tp_size=1, mem_fraction_static=0.6)
    try:
        sampling_params = {
            "temperature": 1.0,
            "top_p": top_p,
            "max_new_tokens": 32,
        }
        out = engine.generate(
            "Write one sentence about the ocean.",
            sampling_params,
            return_logprob=True,
            # slime rollout sets this via custom_params to opt the request in.
            custom_params={"return_top_p_token_ids": True},
        )
        meta = out["meta_info"]

        assert "top_p_token_ids" in meta, f"top_p_token_ids missing from meta_info: {list(meta)}"
        assert "top_p_token_offsets" in meta, "top_p_token_offsets missing from meta_info"

        token_ids = _decode_int32(meta["top_p_token_ids"])
        offsets = _decode_int32(meta["top_p_token_offsets"])

        n_out = len(meta["output_token_logprobs"])
        assert offsets[0] == 0, f"offsets[0] != 0: {offsets[0]}"
        assert offsets[-1] == len(token_ids), (
            f"offsets[-1]={offsets[-1]} != len(token_ids)={len(token_ids)}"
        )
        assert len(offsets) == n_out + 1, (
            f"len(offsets)={len(offsets)} != n_out+1={n_out + 1}"
        )

        # Each response token has a non-empty kept set, and the sampled token
        # (meta output_token_logprobs[i][1]) is in its own kept set.
        sampled_ids = [tok for _, tok, *_ in meta["output_token_logprobs"]]
        for i in range(n_out):
            kept = set(token_ids[offsets[i] : offsets[i + 1]])
            assert kept, f"token {i} has an empty kept set"
            assert sampled_ids[i] in kept, (
                f"sampled token {sampled_ids[i]} not in its own kept set at pos {i}"
            )

        mean_kept = len(token_ids) / max(n_out, 1)
        print(
            f"PASS: {n_out} response tokens, mean nucleus size {mean_kept:.1f} "
            f"(top_p={top_p}); offsets + membership invariants hold."
        )
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
