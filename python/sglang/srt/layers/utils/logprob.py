from __future__ import annotations

import dataclasses
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_params import TOP_K_ALL

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyOutput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class InputLogprobsResult:
    input_token_logprobs: torch.Tensor
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None


def compute_temp_top_p_normalized_logprobs(
    last_logits: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_p: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    compute logprobs for the output token from the given logits.

    Returns:
        torch.Tensor: logprobs from logits
    """
    if top_p is None:
        top_p = logits_metadata.top_p
    if temperature is None:
        temperature = logits_metadata.temperature

    # Scale logits if temperature scaling is enabled
    if logits_metadata.temp_scaled_logprobs:
        last_logits = last_logits / temperature

    # Normalize logprobs if top_p normalization is enabled
    # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
    if logits_metadata.top_p_normalized_logprobs and (top_p != 1.0).any():
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = torch.softmax(last_logits, dim=-1)
        del last_logits
        probs = top_p_normalize_probs_torch(probs, top_p)
        return torch.log(probs)
    else:
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    return get_top_logprobs_raw(
        all_logprobs,
        logits_metadata.top_logprobs_nums,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def _top_p_filter_rows(
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
) -> torch.Tensor:
    """Rows that were requested AND actually have a top-k/top-p/min-p filter."""
    row_has_filter = top_ks != TOP_K_ALL
    if need_top_p_sampling:
        row_has_filter = row_has_filter | (top_ps != 1.0)
    if need_min_p_sampling:
        row_has_filter = row_has_filter | (min_ps > 0)
    return request_mask & row_has_filter


# Default gather width for the top-p nucleus capture. `get_top_p_token_ids_from_probs`
# only needs the KEPT nucleus (a prefix of the prob-sorted order), which is tiny in
# practice (measured width p99 ~11 / max ~35 on Qwen3.6 codeforces rollouts), so we
# bound the descending gather to the top-K with `torch.topk` instead of a full-vocab
# `probs.sort` (~150k). This is the decode-time speedup (see
# notes/training_logs/ioi/2026-07-16_qwen3_6_stage2_topp_mask_replay_observations.md):
# it is decoupled from the sampling `top_k` on purpose, so it needs NO change to the
# rollout recipe / `--rollout-top-k` and does NOT alter the sampling distribution — it
# only bounds how many candidates the capture inspects. `_top_p_keep_mask_bounded`
# falls back to the full sort for any row whose nucleus would exceed K (guarded), so
# correctness never depends on K being large enough.
MASK_TOPK_DEFAULT = 128


def _top_p_keep_mask_sorted(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean nucleus keep-mask in descending-prob order, plus the sort indices.

    Reproduces SGLang's sampler truncation (rank < top_k, cumulative prob within
    top_p, prob >= top1 * min_p) so replay sees the exact set the sampler keeps.
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(probs_sort.shape[-1], device=probs_sort.device).view(1, -1)
    keep = ranks < top_ks.view(-1, 1)
    if need_top_p_sampling:
        keep &= (torch.cumsum(probs_sort, dim=-1) - probs_sort) <= top_ps.view(-1, 1)
    if need_min_p_sampling:
        keep &= probs_sort >= (probs_sort[:, 0] * min_ps).view(-1, 1)
    return keep, probs_idx


def _keep_mask_from_sorted(
    probs_sort: torch.Tensor,
    positions: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
) -> torch.Tensor:
    """Nucleus keep-mask over an already descending-sorted prob tensor.

    `positions` is the descending rank of each column (``arange(width)``); it is the
    truncated width for the `torch.topk` path and the full vocab for the sort path.
    Identical truncation rule as `_top_p_keep_mask_sorted` (rank < top_k, cumulative
    prob within top_p, prob >= top1 * min_p).
    """
    keep = positions < top_ks.view(-1, 1)
    if need_top_p_sampling:
        keep &= (torch.cumsum(probs_sort, dim=-1) - probs_sort) <= top_ps.view(-1, 1)
    if need_min_p_sampling:
        keep &= probs_sort >= (probs_sort[:, 0] * min_ps).view(-1, 1)
    return keep


def _top_p_keep_mask_bounded(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    mask_top_k: int = MASK_TOPK_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nucleus keep-mask + descending token ids, gathering only the top-`mask_top_k`.

    Mirrors upstream sglang's `_compute_sampling_mask_from_probs` (PR #27408): use
    `torch.topk(probs, k=mask_top_k)` instead of a full-vocab `probs.sort`, since the
    kept nucleus is a small prefix of the descending order. Guards the K-bound: any
    row whose nucleus reaches the K-th (last-gathered) column — i.e. the true nucleus
    may extend past K — is recomputed with the full sort so the capture is never
    silently truncated. Returns `(keep, probs_idx)` in the SAME descending-order
    layout as `_top_p_keep_mask_sorted`, but width `K` (or full vocab on fallback).
    """
    vocab_size = probs.shape[-1]
    k = int(mask_top_k)
    if k <= 0 or k >= vocab_size:
        # Nothing to bound — use the full sort directly.
        return _top_p_keep_mask_sorted(
            probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
        )

    probs_sort, probs_idx = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
    positions = torch.arange(k, device=probs.device).view(1, -1)
    keep = _keep_mask_from_sorted(
        probs_sort, positions, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )

    # Silent-truncation guard: if the last gathered column is still kept for a row,
    # that row's nucleus might extend beyond K. Recompute ONLY those rows with the
    # full-vocab sort and splice them in. Cheap in the common case (0 rows), correct
    # in the rare wide-nucleus case. Single batched `.any()` sync (not per-row).
    clipped = keep[:, -1]
    if bool(clipped.any().item()):
        full_keep, full_idx = _top_p_keep_mask_sorted(
            probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
        )
        logger.warning(
            "top-p nucleus capture: %d row(s) hit the mask_top_k=%d gather bound; "
            "recomputing them with the full sort (nucleus wider than K).",
            int(clipped.sum().item()),
            k,
        )
        # Pad the K-width tensors out to full vocab so we can index-assign clipped rows.
        pad = vocab_size - k
        keep = torch.nn.functional.pad(keep, (0, pad), value=False)
        probs_idx = torch.nn.functional.pad(probs_idx, (0, pad), value=0)
        keep[clipped] = full_keep[clipped]
        probs_idx[clipped] = full_idx[clipped]
    return keep, probs_idx


def renorm_logprob_over_top_p(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
    force_keep_token_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    rows = _top_p_filter_rows(
        top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
    )
    if not bool(rows.any().item()):
        return None

    # Bounded gather instead of a full-vocab sort (same speedup as
    # `get_top_p_token_ids_from_probs`; this renorm runs on the same per-decode-step
    # hot path via the sampler). Falls back to full sort per row if a nucleus exceeds
    # K (guarded in `_top_p_keep_mask_bounded`).
    keep, probs_idx = _top_p_keep_mask_bounded(
        probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )
    # Build the vocab-order keep-mask by SETTING only the kept positions True (via a
    # nonzero gather), never writing False. A plain `scatter_(probs_idx, keep)` would
    # be unsafe here: `_top_p_keep_mask_bounded` pads unused columns with index 0, so a
    # padded False could overwrite a genuinely-kept token at vocab id 0.
    keep_vocab = torch.zeros_like(probs, dtype=torch.bool)
    _fr, _fc = keep.nonzero(as_tuple=True)
    keep_vocab[_fr, probs_idx[_fr, _fc]] = True

    if force_keep_token_ids is not None:
        # Force-keep the sampled/accepted token so its renormalized logprob is
        # finite even when SGLang's sampling kernel (e.g. flashinfer) keeps a
        # boundary token that this torch nucleus drops. This matches the trainer,
        # which also force-keeps the target token before renormalizing, so the
        # rollout and training denominators are both ``nucleus ∪ {token}``.
        # Non-filter rows are overwritten by the ``torch.where`` below, so
        # force-keeping every row is harmless and avoids a row gather.
        row_idx = torch.arange(keep_vocab.shape[0], device=keep_vocab.device)
        keep_vocab[row_idx, force_keep_token_ids] = True

    kept_probs = probs * keep_vocab
    kept_probs = kept_probs / kept_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(rows.view(-1, 1), torch.log(kept_probs), torch.log(probs))


def get_top_p_token_ids_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
) -> Optional[List[Optional[torch.Tensor]]]:
    rows = _top_p_filter_rows(
        top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
    )
    if not bool(rows.any().item()):
        return None

    # Bounded gather instead of a full-vocab sort (PR #27408 structure): the kept
    # nucleus is a tiny prefix of the descending order, so `torch.topk(k=128)` avoids
    # sorting ~150k entries every decode step. Falls back to full sort per row if a
    # nucleus would exceed K (guarded inside `_top_p_keep_mask_bounded`).
    keep, probs_idx = _top_p_keep_mask_bounded(
        probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )
    # Zero out rows we don't emit (not requested / no active filter) so the vectorized
    # gather below never produces ids for them, then split the flat id stream back into
    # per-row lists with a SINGLE batched device->host copy (no per-row `.item()` sync,
    # which would serialize decode — the old hot path). Mirrors upstream
    # `_attach_sampling_mask_to_output`.
    keep = keep & rows.view(-1, 1)
    flat_rows, flat_cols = keep.nonzero(as_tuple=True)
    flat_ids = probs_idx[flat_rows, flat_cols].to(torch.int32)
    row_lengths = keep.sum(dim=-1, dtype=torch.int64)

    flat_ids_cpu = flat_ids.cpu()
    row_lengths_cpu = row_lengths.cpu().tolist()
    rows_cpu = rows.cpu().tolist()

    out: List[Optional[torch.Tensor]] = []
    cursor = 0
    for i in range(probs.shape[0]):
        n = int(row_lengths_cpu[i])
        if rows_cpu[i]:
            out.append(flat_ids_cpu[cursor : cursor + n])
        else:
            out.append(None)
        cursor += n
    return out


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                logprobs.device, non_blocking=True
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, no_copy_to_cpu=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_k_nums: List[int],
    pruned_lens: List[int],
    input_top_logprobs_val: List,
    input_top_logprobs_idx: List,
    split_pruned_len: int,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_top_logprobs_val: List to store top-k logprob values
        input_top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    max_k = max(logits_metadata.top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_top_logprobs_val[-1].extend(val)
                input_top_logprobs_idx[-1].extend(idx)
            else:
                input_top_logprobs_val.append(val)
                input_top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    input_token_ids_logprobs_val: List,
    input_token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing token IDs and pruned length info
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_token_ids_logprobs_val: List to store token logprob values
        input_token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """

    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                val.append(logprobs[pt + j, token_ids].tolist())
                idx.append(token_ids)

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_token_ids_logprobs_val[-1].extend(val)
                input_token_ids_logprobs_idx[-1].extend(idx)
            else:
                input_token_ids_logprobs_val.append(val)
                input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def add_output_logprobs_for_spec_v1(
    batch: ScheduleBatch,
    res: Union[EagleVerifyOutput, NgramVerifyInput],
    logits_output: Optional[LogitsProcessorOutput] = None,
):
    # Extract args
    if logits_output is None:
        logits_output = res.logits_output

    if hasattr(res, "accept_length_per_req_cpu"):
        accept_length_per_req_cpu = res.accept_length_per_req_cpu
    else:
        # FIXME: Get a NgramVerifyOutput class and use that instead of this hack.
        accept_length_per_req_cpu = res.accept_length.tolist()

    top_logprobs_nums = batch.top_logprobs_nums
    token_ids_logprobs = batch.token_ids_logprobs
    accepted_indices = res.accepted_indices
    assert len(accepted_indices) == len(logits_output.next_token_logits)

    temperatures = batch.sampling_info.temperatures
    num_draft_tokens = batch.spec_info.draft_token_num
    # acceptance indices are the indices in a "flattened" batch.
    # dividing it to num_draft_tokens will yield the actual batch index.
    temperatures = temperatures[accepted_indices // num_draft_tokens]
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
    else:
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits / temperatures, dim=-1
        )
    batch_next_token_ids = res.verified_id
    num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

    # We should repeat top_logprobs_nums to match num_tokens_per_req.
    top_logprobs_nums_repeat_interleaved = [
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    token_ids_logprobs_repeat_interleaved = [
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    # Extract logprobs
    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)
    if should_top_logprobs:
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            logprobs,
            top_logprobs_nums_repeat_interleaved,
        )

    if should_token_ids_logprobs:
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            logprobs,
            token_ids_logprobs_repeat_interleaved,
        )

    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
        batch_next_token_ids,
    ]

    # Add output logprobs to the request
    pt = 0
    next_token_logprobs = logits_output.next_token_logprobs.tolist()
    verified_ids = batch_next_token_ids.tolist()
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx
    token_ids_logprobs_val = logits_output.next_token_token_ids_logprobs_val
    token_ids_logprobs_idx = logits_output.next_token_token_ids_logprobs_idx
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
        for _ in range(num_tokens):
            if req.return_logprob:
                req.output_token_logprobs_val.append(next_token_logprobs[pt])
                req.output_token_logprobs_idx.append(verified_ids[pt])
                if req.top_logprobs_num > 0:
                    assert (
                        should_top_logprobs
                    ), "Inconsistent state: should_top_logprobs is False"
                    req.output_top_logprobs_val.append(token_top_logprobs_val[pt])
                    req.output_top_logprobs_idx.append(token_top_logprobs_idx[pt])
                if req.token_ids_logprob is not None and should_token_ids_logprobs:
                    req.output_token_ids_logprobs_val.append(token_ids_logprobs_val[pt])
                    req.output_token_ids_logprobs_idx.append(token_ids_logprobs_idx[pt])
            pt += 1
