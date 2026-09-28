"""Flat-model GPU residency: global IDs are slots, with no CPU ID planner.

The caller must select this pool only while every live row fits its GPU budget.
Host/topology changes require flush(), invalidate(), then reset_scores(new_size).
"""
import numpy as np
import torch

from utils.resident_pool_v2 import IndexedPacket


class DirectResidentPool:
    def __init__(self, host, host_scores, capacity, *, live_size=None, device="cuda",
                 transfer_rows=65536, pin_staging=True, ops=None, **kwargs):
        if (host.device.type != "cpu" or host.dtype != torch.float32 or host.ndim != 2
                or host.shape[1] % 3 or host_scores.device.type != "cpu"
                or host_scores.dtype != torch.float32 or host_scores.shape != (len(host),)):
            raise ValueError("expected CPU FP32 model/Adam rows and scores")
        if ops is None or transfer_rows <= 0:
            raise ValueError("direct residency requires native ops and positive transfer_rows")
        self.host, self.host_scores = host, host_scores
        self.device = torch.device(device)
        self.capacity = min(int(capacity), len(host))
        self.live_size = self.capacity if live_size is None else int(live_size)
        if not 0 <= self.live_size <= self.capacity:
            raise ValueError("all live rows must fit the direct GPU budget")
        self.transfer_rows = int(transfer_rows)
        self.ops = ops
        self.state = self.scores = self.active = None
        self.used = self.epoch = 0
        # Aggregate dirty flag, not a CPU per-ID map. IDs never leave the GPU.
        self.dirty = np.zeros(1, dtype=np.bool_)
        self.to_slot = np.empty(0, dtype=np.int64)
        self.stats = dict.fromkeys(("requested_rows", "hit_rows", "uploaded_rows",
            "downloaded_rows", "evicted_rows", "packet_gathers", "same_packet_hits",
            "overflow_steps", "eviction_scan_rows", "prefetched_rows", "prefetch_hit_rows",
            "prefetch_cancelled_rows", "prefetch_skipped_rows", "indexed_packet_gathers",
            "avoided_moment_gather_bytes"), 0)

    @torch.no_grad()
    def _ensure_store(self):
        if self.state is not None:
            return
        if self.live_size > self.capacity:
            raise ValueError("direct residency exceeded its budget; switch to streaming")
        self.state = torch.empty((self.live_size, self.host.shape[1]), device=self.device,
                                 dtype=torch.float32)
        self.scores = torch.empty(self.live_size, device=self.device, dtype=torch.float32)
        for start in range(0, self.live_size, self.transfer_rows):
            end = min(start + self.transfer_rows, self.live_size)
            self.state[start:end].copy_(self.host[start:end])
            self.scores[start:end].copy_(self.host_scores[start:end])
        self.used = self.live_size
        self.stats["uploaded_rows"] += self.live_size

    @torch.no_grad()
    def _commit_active(self):
        packet = self.active
        if packet is not None and packet.dirty:
            # Packet scores started from the current global maxima. No other
            # writer can update them while this packet is active.
            self.scores.index_copy_(0, packet.slots, packet.scores)
            self.dirty[0] = True
            packet.dirty = False

    @torch.no_grad()
    def acquire(self, ids, validate=True):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Gaussian IDs must be a one-dimensional integer array")
        ids = ids.detach().to(dtype=torch.long).contiguous()
        if validate and ids.numel():
            if bool(((ids < 0) | (ids >= self.live_size)).any()):
                raise IndexError("Gaussian ID outside live model")
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError("active cut contains duplicate Gaussian IDs")
        self._commit_active()
        self.active = None
        self._ensure_store()
        raw = self.ops.gather_parameters(self.state, ids)
        self.active = IndexedPacket(ids, ids, raw, self.scores.index_select(0, ids),
                                    store=self.state, ops=self.ops)
        self.stats["requested_rows"] += len(ids)
        self.stats["hit_rows"] += len(ids)
        self.stats["packet_gathers"] += 1
        self.stats["indexed_packet_gathers"] += 1
        self.stats["avoided_moment_gather_bytes"] += len(ids) * (self.host.shape[1] // 3) * 8
        return self.active

    @torch.no_grad()
    def flush(self):
        self._commit_active()
        if self.state is None or not self.dirty.any():
            return
        for start in range(0, self.live_size, self.transfer_rows):
            end = min(start + self.transfer_rows, self.live_size)
            self.host[start:end].copy_(self.state[start:end])
            self.host_scores[start:end].copy_(self.scores[start:end])
        self.stats["downloaded_rows"] += self.live_size
        self.dirty.fill(False)

    def invalidate(self):
        if self.dirty.any() or (self.active is not None and self.active.dirty):
            raise RuntimeError("flush direct pool before invalidating")
        self.active = self.state = self.scores = None
        self.used = 0
        self.epoch += 1

    def reset_scores(self, size):
        if self.used or self.active is not None:
            raise RuntimeError("invalidate before resetting scores/topology")
        if not 0 <= size <= len(self.host):
            raise ValueError("invalid live model size")
        self.live_size = int(size)
        self.host_scores[:size].zero_()

    def resize_empty(self, capacity):
        if self.used or self.active is not None or self.dirty.any():
            raise RuntimeError("resize requires an invalidated pool")
        if capacity < self.live_size:
            raise ValueError("switch to streaming when the model exceeds its budget")
        self.capacity = min(int(capacity), len(self.host))

    def prefetch(self, ids, epoch=None):
        return 0

    def close(self):
        self.flush()
        self.invalidate()
