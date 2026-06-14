import atexit
import logging
import os
import signal
import zlib
from abc import ABC
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size
                    * get_global_server_args().dp_size,
                    max_running_requests,
                ),
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        # Also enable the real capturer if the router-logits dump path is set,
        # so analysis runs don't need --enable-return-routed-experts as well.
        if enable or os.environ.get("SLIME_DUMP_ROUTER_LOGITS_DIR"):
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ):
        raise NotImplementedError

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

        # DeepEP a2a path: each attn-TP rank only sees its scattered slice of
        # topk_ids. All-gather across attn-TP at capture time so device_cache
        # holds the full batch and the existing _sync_fwd / D2H paths work
        # unchanged. Pre-allocate the gather target.
        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = get_attention_tp_size() if is_dp_attention_enabled() else 1
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )

        # Async D->H copy state. Eliminates the per-step cudaStreamSynchronize
        # in _sync_fwd_experts_buffer_DtoH that breaks SGLang's overlap scheduler.
        dev_buf = self.device_cache.buffer
        topk = self.num_experts_per_tok

        # GPU staging buffer - same shape as device_cache.buffer; never overwritten
        # by capture(), so it's safe to read on the copy stream after the snapshot.
        self._staging_buffer = torch.zeros_like(dev_buf)

        # CPU pinned staging for routing data (topk slice only - no fused shared
        # experts, matching the host_cache layout).
        self._pinned_staging = torch.zeros(
            (dev_buf.shape[0], dev_buf.shape[1], topk),
            dtype=dev_buf.dtype,
            device="cpu",
            pin_memory=True,
        )

        # CPU pinned buffer for out_cache_loc indices. int64 to absorb any dtype
        # of forward_batch.out_cache_loc on the GPU side.
        max_batch = dev_buf.shape[0]
        self._pinned_loc = torch.zeros(
            max_batch, dtype=torch.int64, device="cpu", pin_memory=True
        )

        # Dedicated copy stream + event for the async D->H pipeline.
        self._copy_stream = torch.cuda.Stream(device=dev_buf.device)
        self._copy_event = torch.cuda.Event()

        # Pending scatter state. 0 means nothing pending.
        self._pending_n = 0

        staging_mb = (
            self._staging_buffer.nelement() * self._staging_buffer.element_size()
        ) / _MB
        pinned_mb = (
            self._pinned_staging.nelement() * self._pinned_staging.element_size()
        ) / _MB
        logger.info(
            "Routing-replay async D->H: GPU staging %.2f MB, CPU pinned staging %.2f MB",
            staging_mb,
            pinned_mb,
        )

        # Optional: also record full per-layer router_logits (fp16) for offline
        # routing-stability analysis. Enabled by setting
        # SLIME_DUMP_ROUTER_LOGITS_DIR. Reuses the existing copy-stream pipeline
        # so capture stays cuda-graph-safe.
        self._logits_device_cache = None
        self._logits_host_buf = None
        self._logits_dump_path = None
        self._logits_chunk_idx = 0
        dump_dir = os.environ.get("SLIME_DUMP_ROUTER_LOGITS_DIR")
        if dump_dir:
            num_experts = getattr(
                model_config.hf_text_config, "num_experts", None
            ) or getattr(model_config.hf_text_config, "n_routed_experts", None)
            if num_experts is None:
                logger.warning(
                    "SLIME_DUMP_ROUTER_LOGITS_DIR=%s set but model_config exposes "
                    "neither num_experts nor n_routed_experts; router-logits capture "
                    "disabled.",
                    dump_dir,
                )
            else:
                max_batch = dev_buf.shape[0]
                host_capacity = int(
                    os.environ.get("SLIME_DUMP_ROUTER_LOGITS_CAPACITY", "20000")
                )
                self._logits_num_experts = int(num_experts)
                self._logits_device_cache = torch.zeros(
                    (max_batch, self.num_hidden_layers, self._logits_num_experts),
                    dtype=torch.float16,
                    device=device,
                )
                self._logits_staging = torch.zeros_like(self._logits_device_cache)
                self._logits_pinned = torch.zeros(
                    (max_batch, self.num_hidden_layers, self._logits_num_experts),
                    dtype=torch.float16,
                    device="cpu",
                    pin_memory=True,
                )
                self._logits_host_buf = torch.zeros(
                    (host_capacity, self.num_hidden_layers, self._logits_num_experts),
                    dtype=torch.float16,
                    device="cpu",
                )
                self._logits_host_pos = 0
                self._logits_capacity = host_capacity
                rank = int(
                    os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
                )
                os.makedirs(dump_dir, exist_ok=True)
                self._logits_dump_path = os.path.join(
                    dump_dir,
                    f"router_logits_rank{rank}_pid{os.getpid()}.pt",
                )
                dev_mb = (
                    self._logits_device_cache.nelement()
                    * self._logits_device_cache.element_size()
                ) / _MB
                host_mb = (
                    self._logits_host_buf.nelement()
                    * self._logits_host_buf.element_size()
                ) / _MB
                logger.info(
                    "Router-logits capture enabled: max_batch=%d, layers=%d, "
                    "experts=%d, host_capacity=%d, dev=%.1f MB host=%.1f MB, "
                    "dump_path=%s",
                    max_batch,
                    self.num_hidden_layers,
                    self._logits_num_experts,
                    host_capacity,
                    dev_mb,
                    host_mb,
                    self._logits_dump_path,
                )
                atexit.register(self.dump_router_logits)

                # SIGTERM (Ray job stop / process.terminate) does not run atexit
                # handlers by default. Install one that flushes the pending
                # window before falling through to the default action.
                def _term_dump(signum, frame, self=self):
                    try:
                        self.dump_router_logits()
                    finally:
                        signal.signal(signum, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)

                try:
                    signal.signal(signal.SIGTERM, _term_dump)
                except (ValueError, OSError):
                    # Not main thread (e.g. some sglang worker contexts) —
                    # atexit alone will have to suffice. The auto-dump on
                    # capacity-fill is the other safety net.
                    pass
    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        # When DeepEP is enabled, capture() already does all_gather, so
        # device_cache.buffer contains data from all DP ranks. Skip per-rank
        # slicing in that case.
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        n_tok = local_end_pos - local_start_pos
        topk = self.num_experts_per_tok

        # Flush previous pending scatter so pinned buffers can be reused.
        self._flush_pending_scatter()

        # Capture the active stream BEFORE entering the copy-stream context. In
        # overlap-scheduler mode this is forward_stream; without overlap it is the
        # default stream. The copy stream needs to wait on this so the GPU→GPU
        # snapshot below completes before being read.
        active_stream = torch.cuda.current_stream(self.device_cache.buffer.device)

        # 1) GPU→GPU snapshot on the active stream — fast, no sync.
        self._staging_buffer[:n_tok].copy_(
            self.device_cache.buffer[local_start_pos:local_end_pos]
        )
        if self._logits_device_cache is not None:
            # Mirror the topk_ids snapshot for the parallel router_logits buffer.
            # Note: DeepEP path doesn't all-gather logits today — we capture only
            # the local attn-TP slice. For analysis at scale this is acceptable
            # (every rank sees a representative sample of tokens).
            self._logits_staging[:n_tok].copy_(
                self._logits_device_cache[local_start_pos:local_end_pos]
            )

        # 2) On copy stream: async copies to pinned CPU buffers.
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(active_stream)
            self._pinned_staging[:n_tok, :, :topk].copy_(
                self._staging_buffer[:n_tok, :, :topk], non_blocking=True
            )
            self._pinned_loc[:n_tok].copy_(
                forward_batch.out_cache_loc, non_blocking=True
            )
            if self._logits_device_cache is not None:
                self._logits_pinned[:n_tok].copy_(
                    self._logits_staging[:n_tok], non_blocking=True
                )

        # 3) Record event — no sync, returns immediately.
        self._copy_event.record(self._copy_stream)

        # 4) Mark pending; scatter happens at next flush.
        self._pending_n = n_tok

    def _flush_pending_scatter(self):
        """Synchronize pending async copy and scatter pinned data into host_cache.

        Called at the start of the next _sync_fwd_experts_buffer_DtoH (so pinned
        buffers can be reused) and at the start of get_routed_experts (so the
        host_cache read sees the latest data).
        """
        if self._pending_n == 0:
            return
        self._copy_event.synchronize()
        n = self._pending_n
        topk = self.num_experts_per_tok
        loc = self._pinned_loc[:n]
        self.host_cache.buffer[loc] = self._pinned_staging[:n, :, :topk]
        if self._logits_device_cache is not None:
            # Append to the host ring buffer; when it fills, dump a chunk to
            # disk and continue with the remainder so we don't drop tokens on
            # long runs.
            offset = 0
            while offset < n:
                remaining = self._logits_capacity - self._logits_host_pos
                take = min(n - offset, remaining)
                if take > 0:
                    self._logits_host_buf[
                        self._logits_host_pos : self._logits_host_pos + take
                    ] = self._pinned_logits_view(offset, take)
                    self._logits_host_pos += take
                    offset += take
                if self._logits_host_pos >= self._logits_capacity:
                    # Dump-and-reset; pos goes back to 0 in dump_router_logits.
                    self.dump_router_logits()
        self._pending_n = 0

    def _pinned_logits_view(self, offset: int, take: int) -> torch.Tensor:
        return self._logits_pinned[offset : offset + take]

    def capture_router_logits(
        self, layer_id: int, router_logits: torch.Tensor
    ) -> None:
        """Record raw router logits (pre-softmax) for offline analysis.

        Mirrors :py:meth:`capture` for topk_ids. The capture point in topk.py
        invokes us before logical-to-physical id remap, so columns are aligned
        with the model's logical expert indices 0..num_experts-1.
        """
        if self._logits_device_cache is None:
            return
        batch = router_logits.shape[0]
        # Pre-allocated fp16 device buffer; write the layer's row in place.
        self._logits_device_cache[:batch, layer_id, :] = router_logits.to(
            torch.float16
        )

    def dump_router_logits(self) -> Optional[str]:
        """Flush the host ring buffer to disk. Idempotent / safe at atexit."""
        if self._logits_host_buf is None or self._logits_dump_path is None:
            return None
        # Make sure any pending async copies have landed.
        self._flush_pending_scatter()
        n = self._logits_host_pos
        if n == 0:
            return None
        payload = {
            "router_logits": self._logits_host_buf[:n].clone(),
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self._logits_num_experts,
            "captured_tokens": n,
            "capacity": self._logits_capacity,
        }
        # Chunked output: a new file per dump call. Avoids overwriting on
        # repeated triggers (atexit + future on-demand flushes).
        path = self._logits_dump_path
        if self._logits_chunk_idx > 0:
            base, ext = os.path.splitext(path)
            path = f"{base}_chunk{self._logits_chunk_idx}{ext}"
        torch.save(payload, path)
        logger.info(
            "Router logits dumped: %d tokens x %d layers x %d experts -> %s",
            n,
            self.num_hidden_layers,
            self._logits_num_experts,
            path,
        )
        # Reset for the next chunk window.
        self._logits_host_pos = 0
        self._logits_chunk_idx += 1
        return path

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        if get_moe_a2a_backend().is_deepep():
            local_topk_ids = topk_ids
            topk_ids = self.gather_buffer[
                : local_topk_ids.size(0) * get_attention_tp_size()
            ]
            attn_tp_all_gather_into_tensor(topk_ids, local_topk_ids)
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ):
        # Ensure the most recent async D→H copy has landed in host_cache before reading.
        self._flush_pending_scatter()

        end = seqlen - 1
        # Defensive clamp — caller guarantees 0 <= start_len <= seqlen, but a
        # stale prefix length (e.g. session reuse mishap) shouldn't slice past
        # the end. Slicing past end yields an empty tensor naturally.
        start_len = max(0, min(start_len, end))
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len:end].cpu().clone()
        )
        return self.get_host_cache().buffer[cache_pool_idx]

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        self._sync_fwd_experts_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def capture_router_logits(
        self, layer_id: int, router_logits: torch.Tensor
    ) -> None:
        pass

    def dump_router_logits(self) -> Optional[str]:
        return None

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ):
        pass

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        pass

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_from_meta_info(data):
    # Routing experts are encoded as zlib-compressed int16 in base64.
    # See tokenizer_manager (v0.5.10 path) for the encoding side.
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        zlib.decompress(pybase64.b64decode(routed_experts_base64.encode("utf-8"))),
        dtype=np.int16,
    ).astype(np.int32)
    return routed_experts
