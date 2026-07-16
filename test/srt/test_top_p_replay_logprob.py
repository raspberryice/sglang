"""CPU-only correctness tests for the top-p mask replay helpers.

These cover the pure functions added for top-p mask replay (slime top-p replay,
ports THUDM/slime#2102):

- ``_top_p_filter_rows``          — which rows actually got truncated
- ``_top_p_keep_mask_sorted``     — the nucleus keep-mask (rank/top-p/min-p)
- ``get_top_p_token_ids_from_probs`` — the kept vocab ids per row
- ``renorm_logprob_over_top_p``   — logprob renormalized over the nucleus

The functions are pure torch and run on CPU; no GPU or running engine needed.
Each is checked against an independent reference implementation written directly
from the nucleus definition, plus the boundary/edge cases that the trainer-side
mask must match exactly.

Run:  python -m pytest test/srt/test_top_p_replay_logprob.py -q
"""

import unittest

import torch

from sglang.srt.layers.utils.logprob import (
    _top_p_keep_mask_bounded,
    _top_p_keep_mask_sorted,
    _top_p_filter_rows,
    get_top_p_token_ids_from_probs,
    renorm_logprob_over_top_p,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def _reference_nucleus(row_probs, top_k, top_p, min_p, need_top_p, need_min_p):
    """Independent per-row reference: the set of kept vocab ids.

    Mirrors the nucleus definition (rank < top_k, EXCLUSIVE cumulative prob
    within top_p, prob >= top1 * min_p) written from scratch so it does not share
    code with the implementation under test.
    """
    V = row_probs.shape[-1]
    order = sorted(range(V), key=lambda j: (-float(row_probs[j]), j))
    kept = []
    cum_before = 0.0
    top1 = float(row_probs[order[0]])
    for rank, j in enumerate(order):
        p = float(row_probs[j])
        ok = rank < top_k
        if need_top_p:
            ok = ok and (cum_before <= top_p)
        if need_min_p:
            ok = ok and (p >= top1 * min_p)
        if ok:
            kept.append(j)
        cum_before += p
    return set(kept)


def _make_probs(batch, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=g)
    return torch.softmax(logits, dim=-1)


class TestTopPReplayLogprob(unittest.TestCase):
    def _params(self, batch, top_ps, top_ks, min_ps):
        return (
            torch.tensor(top_ks, dtype=torch.long),
            torch.tensor(top_ps, dtype=torch.float32),
            torch.tensor(min_ps, dtype=torch.float32),
        )

    def test_ids_match_reference_topp_only(self):
        probs = _make_probs(6, 200, seed=1)
        top_ps = [0.8, 0.9, 0.95, 0.99, 1.0, 0.5]
        top_ks = [TOP_K_ALL] * 6
        min_ps = [0.0] * 6
        top_ks_t, top_ps_t, min_ps_t = self._params(6, top_ps, top_ks, min_ps)
        request_mask = torch.ones(6, dtype=torch.bool)

        ids = get_top_p_token_ids_from_probs(
            probs=probs,
            top_ks=top_ks_t,
            top_ps=top_ps_t,
            min_ps=min_ps_t,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            request_mask=request_mask,
        )
        self.assertIsNotNone(ids)
        for i in range(6):
            ref = _reference_nucleus(
                probs[i], top_k=1 << 60, top_p=top_ps[i], min_p=0.0,
                need_top_p=True, need_min_p=False,
            )
            if top_ps[i] == 1.0:
                # row not truncated by top_p alone and top_k is ALL -> no filter
                self.assertIsNone(ids[i])
            else:
                got = set(int(x) for x in ids[i].tolist())
                self.assertEqual(got, ref, f"row {i} top_p={top_ps[i]}")

    def test_ids_match_reference_topk_and_minp(self):
        probs = _make_probs(5, 128, seed=2)
        top_ps = [0.95, 0.9, 1.0, 0.9, 0.8]
        top_ks = [20, 50, 10, TOP_K_ALL, 30]
        min_ps = [0.0, 0.05, 0.0, 0.05, 0.02]
        top_ks_t, top_ps_t, min_ps_t = self._params(5, top_ps, top_ks, min_ps)
        request_mask = torch.ones(5, dtype=torch.bool)

        ids = get_top_p_token_ids_from_probs(
            probs=probs,
            top_ks=top_ks_t,
            top_ps=top_ps_t,
            min_ps=min_ps_t,
            need_top_p_sampling=True,
            need_min_p_sampling=True,
            request_mask=request_mask,
        )
        self.assertIsNotNone(ids)
        for i in range(5):
            ref = _reference_nucleus(
                probs[i], top_k=top_ks[i], top_p=top_ps[i], min_p=min_ps[i],
                need_top_p=True, need_min_p=True,
            )
            got = set(int(x) for x in ids[i].tolist())
            self.assertEqual(got, ref, f"row {i}")

    def test_filter_rows_and_request_mask(self):
        top_ks = torch.tensor([TOP_K_ALL, 20, TOP_K_ALL], dtype=torch.long)
        top_ps = torch.tensor([1.0, 1.0, 0.9], dtype=torch.float32)
        min_ps = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        request_mask = torch.tensor([True, True, False], dtype=torch.bool)
        rows = _top_p_filter_rows(
            top_ks, top_ps, min_ps,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
        )
        # row0: no filter (top_k ALL, top_p 1.0) -> False
        # row1: top_k filter -> True
        # row2: top_p filter but request_mask False -> False
        self.assertEqual(rows.tolist(), [False, True, False])

    def test_renorm_equals_logsoftmax_over_kept(self):
        probs = _make_probs(4, 100, seed=3)
        top_ps = [0.9, 0.95, 0.8, 1.0]
        top_ks = [TOP_K_ALL] * 4
        min_ps = [0.0] * 4
        top_ks_t, top_ps_t, min_ps_t = self._params(4, top_ps, top_ks, min_ps)
        request_mask = torch.ones(4, dtype=torch.bool)

        logp = renorm_logprob_over_top_p(
            probs=probs,
            top_ks=top_ks_t,
            top_ps=top_ps_t,
            min_ps=min_ps_t,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            request_mask=request_mask,
        )
        self.assertIsNotNone(logp)
        for i in range(4):
            ref = _reference_nucleus(
                probs[i], top_k=1 << 60, top_p=top_ps[i], min_p=0.0,
                need_top_p=True, need_min_p=False,
            )
            if top_ps[i] == 1.0:
                # non-filter row: returns log(probs) unchanged
                torch.testing.assert_close(logp[i], torch.log(probs[i]))
                continue
            kept_idx = torch.tensor(sorted(ref), dtype=torch.long)
            denom = probs[i, kept_idx].sum()
            # kept tokens: log(p / denom); dropped tokens: -inf
            for j in range(probs.shape[1]):
                if j in ref:
                    expect = torch.log(probs[i, j] / denom)
                    self.assertTrue(
                        torch.allclose(logp[i, j], expect, atol=1e-5),
                        f"row {i} kept token {j}",
                    )
                else:
                    self.assertTrue(
                        torch.isinf(logp[i, j]) and logp[i, j] < 0,
                        f"row {i} dropped token {j} should be -inf",
                    )

    def test_force_keep_makes_target_finite(self):
        probs = _make_probs(1, 64, seed=4)
        # pick a low-prob token that is guaranteed outside the nucleus
        target = int(torch.argmin(probs[0]).item())
        top_ks_t, top_ps_t, min_ps_t = self._params(1, [0.5], [TOP_K_ALL], [0.0])
        request_mask = torch.ones(1, dtype=torch.bool)

        # without force-keep: target is dropped -> -inf
        logp_plain = renorm_logprob_over_top_p(
            probs=probs, top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
        )
        self.assertTrue(torch.isinf(logp_plain[0, target]))

        # with force-keep: denominator is nucleus ∪ {target}, target finite
        logp_forced = renorm_logprob_over_top_p(
            probs=probs, top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
            force_keep_token_ids=torch.tensor([target], dtype=torch.long),
        )
        self.assertTrue(torch.isfinite(logp_forced[0, target]))

        # reference: nucleus ∪ {target}
        ref = _reference_nucleus(
            probs[0], top_k=1 << 60, top_p=0.5, min_p=0.0,
            need_top_p=True, need_min_p=False,
        )
        ref.add(target)
        kept_idx = torch.tensor(sorted(ref), dtype=torch.long)
        denom = probs[0, kept_idx].sum()
        torch.testing.assert_close(
            logp_forced[0, target], torch.log(probs[0, target] / denom)
        )

    def test_no_filter_returns_none(self):
        probs = _make_probs(3, 50, seed=5)
        top_ks_t, top_ps_t, min_ps_t = self._params(
            3, [1.0, 1.0, 1.0], [TOP_K_ALL] * 3, [0.0, 0.0, 0.0]
        )
        request_mask = torch.ones(3, dtype=torch.bool)
        ids = get_top_p_token_ids_from_probs(
            probs=probs, top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
        )
        self.assertIsNone(ids)
        logp = renorm_logprob_over_top_p(
            probs=probs, top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
        )
        self.assertIsNone(logp)

    def _kept_ids_set(self, keep, probs_idx, row):
        return set(int(x) for x in probs_idx[row][keep[row]].tolist())

    def test_bounded_topk_equals_full_sort(self):
        """The bounded `torch.topk` mask matches the full-sort mask exactly.

        The decode-time speedup (PR #27408 structure): gathering only the top-K
        candidates instead of a full-vocab sort must produce the identical kept
        nucleus whenever the nucleus fits within K. vocab 200, K default 128 (>=
        every nucleus at these top_p/top_k/min_p), random non-tied probs.
        """
        probs = _make_probs(6, 200, seed=11)
        top_ps = [0.8, 0.9, 0.95, 0.99, 0.7, 0.5]
        top_ks = [TOP_K_ALL, 50, TOP_K_ALL, 100, 30, TOP_K_ALL]
        min_ps = [0.0, 0.0, 0.05, 0.0, 0.02, 0.0]
        top_ks_t, top_ps_t, min_ps_t = self._params(6, top_ps, top_ks, min_ps)
        common = dict(
            top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=True,
        )
        keep_full, idx_full = _top_p_keep_mask_sorted(probs, **common)
        keep_bnd, idx_bnd = _top_p_keep_mask_bounded(probs, **common)
        for i in range(6):
            self.assertEqual(
                self._kept_ids_set(keep_bnd, idx_bnd, i),
                self._kept_ids_set(keep_full, idx_full, i),
                f"row {i}: bounded topk nucleus != full-sort nucleus",
            )

    def test_bounded_falls_back_when_nucleus_exceeds_k(self):
        """A nucleus wider than K is recovered via the full-sort fallback.

        Near-uniform probs over vocab 300 with a small explicit `mask_top_k=16`
        forces the K-bound guard to fire; the recomputed nucleus must still equal
        the full-sort nucleus (no silent truncation).
        """
        probs = torch.full((1, 300), 1.0 / 300.0)
        top_ks_t, top_ps_t, min_ps_t = self._params(1, [0.9], [TOP_K_ALL], [0.0])
        common = dict(
            top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
        )
        keep_full, idx_full = _top_p_keep_mask_sorted(probs, **common)
        keep_bnd, idx_bnd = _top_p_keep_mask_bounded(probs, mask_top_k=16, **common)
        full_set = self._kept_ids_set(keep_full, idx_full, 0)
        bnd_set = self._kept_ids_set(keep_bnd, idx_bnd, 0)
        # The nucleus is far wider than 16, so the guard must have engaged.
        self.assertGreater(len(full_set), 16)
        self.assertEqual(bnd_set, full_set, "K-bound fallback dropped nucleus tokens")

    def test_sampled_token_in_own_nucleus(self):
        """The token the sampler would draw is always inside the recorded set.

        Sanity check that the kept set is the region the sampler samples from:
        the argmax (top-1) token is always kept for any top_p in (0, 1].
        """
        probs = _make_probs(4, 100, seed=6)
        top_ps = [0.1, 0.5, 0.9, 0.95]
        top_ks_t, top_ps_t, min_ps_t = self._params(
            4, top_ps, [TOP_K_ALL] * 4, [0.0] * 4
        )
        request_mask = torch.ones(4, dtype=torch.bool)
        ids = get_top_p_token_ids_from_probs(
            probs=probs, top_ks=top_ks_t, top_ps=top_ps_t, min_ps=min_ps_t,
            need_top_p_sampling=True, need_min_p_sampling=False,
            request_mask=request_mask,
        )
        for i in range(4):
            top1 = int(torch.argmax(probs[i]).item())
            self.assertIn(top1, set(int(x) for x in ids[i].tolist()))


if __name__ == "__main__":
    unittest.main()
