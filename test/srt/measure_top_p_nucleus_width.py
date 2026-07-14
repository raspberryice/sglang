"""Measure the top-p nucleus-width distribution on real AOJ prompts.

Answers the representation question for top-p mask replay (ragged ids+offsets
vs fixed-width [T,k]): what is the per-response-token kept-nucleus size on our
actual codeforces/AOJ workload, at the checkpoint we train?

This is the smoke-test harness extended: a BARE sglang Engine (no strands agent
loop / judge / sandbox — just Engine.generate on prompt strings), fed real AOJ
prompts with the chat template applied, sampling at top_p (default 0.95). For
every response token we read the kept-nucleus size off
meta_info["top_p_token_offsets"] and report the distribution + how often it
exceeds candidate top-k caps.

Caveat: the real training path wraps prompts in a codeforces tool-surface
template and runs multi-turn via the agent bridge; this bare-Engine run feeds
the raw prompt text single-turn. So the token distribution is representative of
code-reasoning generation but is NOT the exact agentic distribution — read the
result as an order-of-magnitude on nucleus width, not an exact production number.

REQUIRES A GPU. Env:
  TEST_TOP_P_MODEL   path to the HF checkpoint (required)
  AOJ_PROMPTS_JSONL  path to a JSONL with an "prompt" field (required)
  N_PROMPTS          how many prompts to sample (default 40)
  TOP_P              nucleus p (default 0.95)
  TOP_K              top-k cap to also apply at sampling (default -1 = none)
  MAX_NEW_TOKENS     per-prompt generation length (default 512)

Example:
  TEST_TOP_P_MODEL=/shared/home/slliz/models/Qwen3.6-35B-A3B \
  AOJ_PROMPTS_JSONL=/shared/home/slliz/data/rl_training/codeforces/aoj/codeforces-train-full.rmbug.q36-4to6.cpp.aoj_sandbox.jsonl \
  N_PROMPTS=40 MAX_NEW_TOKENS=512 \
    python test/srt/measure_top_p_nucleus_width.py
"""

import base64
import json
import os
import sys

import numpy as np


def _decode_int32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


def _load_prompts(path: str, n: int) -> list:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = d.get("prompt")
            if p is None:
                continue
            prompts.append(p)
            if len(prompts) >= n:
                break
    return prompts


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[idx]


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device.")
        return 0

    model_path = os.environ.get("TEST_TOP_P_MODEL")
    prompts_path = os.environ.get("AOJ_PROMPTS_JSONL")
    if not model_path or not prompts_path:
        print("SKIP: set TEST_TOP_P_MODEL and AOJ_PROMPTS_JSONL.")
        return 0

    n_prompts = int(os.environ.get("N_PROMPTS", "40"))
    top_p = float(os.environ.get("TOP_P", "0.95"))
    top_k = int(os.environ.get("TOP_K", "-1"))
    max_new = int(os.environ.get("MAX_NEW_TOKENS", "512"))

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    raw_prompts = _load_prompts(prompts_path, n_prompts)
    if not raw_prompts:
        print(f"ERROR: no prompts with a 'prompt' field in {prompts_path}")
        return 1

    # Apply the chat template so inputs match training (prompt is a user turn).
    def to_text(p):
        if isinstance(p, list):  # already a messages list
            msgs = p
        else:
            msgs = [{"role": "user", "content": str(p)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    texts = [to_text(p) for p in raw_prompts]
    print(f"Loaded {len(texts)} AOJ prompts from {prompts_path}")

    sampling_params = {
        "temperature": 1.0,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new,
        "custom_params": {"return_top_p_token_ids": True},
    }

    engine = sgl.Engine(model_path=model_path, tp_size=8, mem_fraction_static=0.8)
    try:
        outs = engine.generate(texts, sampling_params, return_logprob=True)
        if isinstance(outs, dict):
            outs = [outs]

        widths = []  # per-response-token nucleus size across all prompts
        n_missing = 0
        for out in outs:
            meta = out["meta_info"]
            if "top_p_token_offsets" not in meta:
                n_missing += 1
                continue
            offs = _decode_int32(meta["top_p_token_offsets"])
            # per-token widths = diff of the ragged offsets
            w = np.diff(offs)
            widths.extend(int(x) for x in w)

        if not widths:
            print(f"ERROR: no top-p data captured (n_missing={n_missing}/{len(outs)}).")
            return 1

        widths_sorted = sorted(widths)
        arr = np.array(widths, dtype=np.int64)
        total_ids = int(arr.sum())
        n_tokens = len(widths)

        print("\n===== TOP-P NUCLEUS WIDTH (per response token) =====")
        print(f"model         : {model_path}")
        print(f"prompts       : {len(outs)}  (missing top-p: {n_missing})")
        print(f"top_p={top_p}  top_k={top_k}  max_new_tokens={max_new}")
        print(f"response tokens measured : {n_tokens}")
        print(f"mean nucleus  : {arr.mean():.2f}")
        print(f"p50 / p90 / p95 / p99 / max : "
              f"{_percentile(widths_sorted, 0.50)} / {_percentile(widths_sorted, 0.90)} / "
              f"{_percentile(widths_sorted, 0.95)} / {_percentile(widths_sorted, 0.99)} / {widths_sorted[-1]}")
        print(f"total kept ids : {total_ids}  (ragged int32 bytes = {total_ids * 4 / 1e6:.2f} MB for this sample)")

        print("\n--- fixed-width [T,k] cost vs exceedance (if we capped at top_k=k) ---")
        for k in (8, 16, 20, 32, 50, 100, 200):
            over = int((arr > k).sum())
            dense_bytes = n_tokens * k * 4
            print(f"  k={k:>3}: {100.0 * over / n_tokens:5.1f}% of tokens exceed k "
                  f"| dense [T,k] = {dense_bytes / 1e6:6.2f} MB "
                  f"({dense_bytes / max(total_ids * 4, 1):.1f}x the ragged size)")
        print("=====================================================")
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
