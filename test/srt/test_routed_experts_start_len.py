"""Unit tests for `RoutedExpertsCapturer.get_routed_experts(..., start_len=...)`.

The full integration test in `test/registered/rl/test_return_routed_experts.py`
covers the end-to-end path through SGLang's HTTP server. This unit test
isolates the slicing contract — what `start_len` means, that defaults preserve
the historical behavior, and that defensive clamping handles edge cases —
without needing a GPU model launch.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsCapturerReal


def _make_capturer(num_tokens: int, num_layers: int, top_k: int) -> _RoutedExpertsCapturerReal:
    """Construct a capturer instance with a deterministic host buffer.

    Bypasses __init__ to avoid CUDA/distributed setup. Each row of the host
    buffer encodes its absolute position in low-order bits so we can verify the
    slice exactly: `buffer[i, layer, k] == i * 1000 + layer * 10 + k`.
    """
    cap = _RoutedExpertsCapturerReal.__new__(_RoutedExpertsCapturerReal)
    host_cache = SimpleNamespace()
    host_cache.buffer = torch.zeros(num_tokens, num_layers, top_k, dtype=torch.int32)
    for i in range(num_tokens):
        for layer in range(num_layers):
            for k in range(top_k):
                host_cache.buffer[i, layer, k] = i * 1000 + layer * 10 + k
    cap.host_cache = host_cache
    return cap


def _make_req_to_token_pool(req_pool_idx: int, token_indices: list[int]) -> SimpleNamespace:
    """Mock ReqToTokenPool whose `req_to_token[req_pool_idx]` slices to `token_indices`."""
    pool = SimpleNamespace()
    # We only need indexing by `req_pool_idx`, so a 2-D tensor where row `req_pool_idx` is the mapping.
    rows = max(req_pool_idx + 1, 1)
    pool.req_to_token = torch.zeros((rows, len(token_indices)), dtype=torch.int64)
    pool.req_to_token[req_pool_idx] = torch.tensor(token_indices, dtype=torch.int64)
    return pool


class TestRoutedExpertsStartLen(unittest.TestCase):
    NUM_TOKENS = 32
    NUM_LAYERS = 4
    TOP_K = 2
    REQ_POOL_IDX = 0

    def setUp(self) -> None:
        self.capturer = _make_capturer(self.NUM_TOKENS, self.NUM_LAYERS, self.TOP_K)

    def _call(self, seqlen: int, start_len: int = 0) -> torch.Tensor:
        # Map trajectory positions [0, seqlen) to host-buffer rows [0, seqlen).
        token_indices = list(range(seqlen))
        pool = _make_req_to_token_pool(self.REQ_POOL_IDX, token_indices)
        return self.capturer.get_routed_experts(
            req_pool_idx=self.REQ_POOL_IDX,
            seqlen=seqlen,
            req_to_token_pool=pool,
            start_len=start_len,
        )

    def test_default_start_len_zero_preserves_full_slice(self):
        """start_len=0 must return routing for [0, seqlen-1) — the historical contract."""
        seqlen = 10
        out = self._call(seqlen, start_len=0)
        self.assertEqual(out.shape, (seqlen - 1, self.NUM_LAYERS, self.TOP_K))
        # Position 0's row should be index 0 (low bits encode the position).
        self.assertEqual(int(out[0, 0, 0]), 0 * 1000 + 0 * 10 + 0)
        # Last position should be seqlen - 2.
        self.assertEqual(int(out[-1, 0, 0]), (seqlen - 2) * 1000)

    def test_start_len_trims_prefix(self):
        """start_len=K returns routing for [K, seqlen-1) — length seqlen - 1 - K."""
        seqlen = 10
        for k in (1, 3, 5, 7):
            with self.subTest(start_len=k):
                out = self._call(seqlen, start_len=k)
                expected_len = seqlen - 1 - k
                self.assertEqual(out.shape, (expected_len, self.NUM_LAYERS, self.TOP_K))
                # First row should encode position K (the trim boundary).
                self.assertEqual(int(out[0, 0, 0]), k * 1000)
                # Last row should be position seqlen - 2.
                self.assertEqual(int(out[-1, 0, 0]), (seqlen - 2) * 1000)

    def test_start_len_equals_seqlen_minus_one_returns_empty(self):
        """When start_len == seqlen - 1, no positions remain — empty tensor."""
        seqlen = 10
        out = self._call(seqlen, start_len=seqlen - 1)
        self.assertEqual(out.shape, (0, self.NUM_LAYERS, self.TOP_K))

    def test_start_len_overflow_clamps_to_empty(self):
        """Stale / mis-set start_len > seqlen-1 should not crash; produce empty."""
        seqlen = 10
        out = self._call(seqlen, start_len=seqlen + 5)
        self.assertEqual(out.shape, (0, self.NUM_LAYERS, self.TOP_K))

    def test_negative_start_len_clamps_to_zero(self):
        """A negative start_len shouldn't reverse-slice or raise — treat as 0."""
        seqlen = 10
        out_neg = self._call(seqlen, start_len=-5)
        out_zero = self._call(seqlen, start_len=0)
        self.assertEqual(out_neg.shape, out_zero.shape)
        self.assertTrue(torch.equal(out_neg, out_zero))

    def test_concat_partial_slices_reconstructs_full(self):
        """Two slices stitched at a boundary == the full slice. This is the property
        multi-turn agent rollouts depend on: each turn requests routing only for its
        new tokens, and the bridge concatenates them.
        """
        seqlen = 12
        full = self._call(seqlen, start_len=0)
        for boundary in (1, 4, 7, seqlen - 2):
            with self.subTest(boundary=boundary):
                head = self._call(boundary + 1, start_len=0)
                tail = self._call(seqlen, start_len=boundary)
                stitched = torch.cat([head, tail], dim=0)
                self.assertEqual(stitched.shape, full.shape)
                self.assertTrue(torch.equal(stitched, full))


if __name__ == "__main__":
    unittest.main()
