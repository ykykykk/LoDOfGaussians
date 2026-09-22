"""Fixed-slot Gaussian cache with a compact, write-back active render packet.

The host store uses ALoD's [parameters | Adam m | Adam v] row layout.
Only requested rows are packed for gsplat; cold resident rows never move.
This module deliberately has no CUDA extension or scene import requirement.
"""
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import torch


@dataclass
class ActivePacket:
    ids: np.ndarray
    slots: Optional[torch.Tensor]
    state: torch.Tensor
    scores: torch.Tensor
    dirty: bool = False

    @property
    def width(self) -> int:
        return self.state.shape[1] // 3

    def parameters(self) -> torch.Tensor:
        # A leaf view, not a gather whose backward allocates a full-pool gradient.
        return self.state[:, :self.width].detach().requires_grad_(True)

    @torch.no_grad()
    def accumulate_scores(self, indices: torch.Tensor, scores: torch.Tensor) -> None:
        # One camera normally yields unique packed IDs. amax also handles repeats.
        self.scores.scatter_reduce_(0, indices.long(), scores.detach(), reduce="amax", include_self=True)
        self.dirty = True

    @torch.no_grad()
    def adam_step(self, grad: torch.Tensor, rates: torch.Tensor, iteration: int,
                  frozen_prefix: int = 0) -> None:
        """ALoD OurAdam._single_tensor_adam2, packed across parameter groups.

        iteration is the original zero-based fine-training counter; upstream
        constructs tensor(iteration) and increments it before bias correction.
        Cold rows and missing views do NOT decay moments or advance a local step.
        """
        d = self.width
        if grad.shape != (len(self.ids), d) or rates.shape != (d,):
            raise ValueError("Adam gradient/rate dimensions do not match the packet")
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if frozen_prefix:
            frozen = torch.as_tensor(self.ids < frozen_prefix, device=grad.device)
            grad = grad.clone()
            grad[frozen] = 0
        p, m, v = self.state.split(d, dim=1)
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        step = iteration + 1
        denom = v.sqrt().div_(math.sqrt(1.0 - 0.999 ** step)).add_(1e-8)
        p.sub_(m.div(denom).mul_(rates / (1.0 - 0.9 ** step)))
        self.dirty = True


def capacity_for_budget(requested: int, width: int, free_bytes: int,
                        pool_gib: float, headroom_gib: float) -> int:
    """Budget the resident store, not the renderer's unpredictable peak.

    The caller retains a configurable headroom for images, render/backward and
    the active packet. This is an allocation cap, not an OOM guarantee.
    """
    if requested <= 0 or width <= 0 or pool_gib <= 0 or headroom_gib < 0:
        raise ValueError("invalid resident memory budget")
    available = max(0, int(free_bytes - headroom_gib * 2**30))
    budget = min(int(pool_gib * 2**30), available)
    return min(requested, budget // (4 * (3 * width + 1)))


class ResidentPool:
    """CPU ID planner + fixed CUDA/CPU slots + lazy write-back render packet.

    Public acquire() validates uniqueness. The internal selector can opt out
    after establishing a unique cut. Host mutation requires flush()+invalidate().
    A cut larger than the pool uses a direct active packet, never point dropping.
    The pool is released in that mode so it does not compete with the render.
    """
    def __init__(self, host: torch.Tensor, host_scores: torch.Tensor, capacity: int,
                 device="cuda", transfer_rows: int = 65536, pin_staging: bool = True):
        if host.device.type != "cpu" or host_scores.device.type != "cpu":
            raise ValueError("resident backing store must be on CPU")
        if host.dtype != torch.float32 or host.ndim != 2 or host.shape[1] % 3:
            raise ValueError("expected FP32 [N, parameters+Adam_m+Adam_v] host rows")
        if host_scores.shape != (len(host),) or host_scores.dtype != torch.float32:
            raise ValueError("expected FP32 host scores [N]")
        if capacity < 0 or transfer_rows <= 0:
            raise ValueError("invalid capacity/transfer_rows")
        self.host, self.host_scores = host, host_scores
        self.device = torch.device(device)
        self.capacity = min(int(capacity), len(host))
        self.transfer_rows = int(transfer_rows)
        self.pin_staging = bool(pin_staging and self.device.type == "cuda")
        self.to_slot = np.full(len(host), -1, dtype=np.int64)
        self.to_id = np.full(self.capacity, -1, dtype=np.int64)
        self.last_used = np.zeros(self.capacity, dtype=np.int64)
        self.dirty = np.zeros(self.capacity, dtype=np.bool_)
        self.used = 0
        self.clock = 0
        self.active = None
        self.state = self.scores = None
        self._staging = None
        self._staging_done = None
        self._last_cuda_ids = None
        self._eviction_cursor = 0
        self.stats = dict(requested_rows=0, hit_rows=0, uploaded_rows=0,
                          downloaded_rows=0, evicted_rows=0, packet_gathers=0,
                          same_packet_hits=0, overflow_steps=0, eviction_scan_rows=0)

    def _ensure_store(self):
        if self.state is None:
            self.state = torch.empty((self.capacity, self.host.shape[1]), dtype=torch.float32, device=self.device)
            self.scores = torch.empty(self.capacity, dtype=torch.float32, device=self.device)

    def _index(self, a: np.ndarray, device=None) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.long, device=device or self.device)

    def _upload_block(self, rows: torch.Tensor) -> torch.Tensor:
        if not self.pin_staging:
            return rows.to(self.device)
        # Pin a bounded transfer slab, NEVER the entire out-of-core scene.
        # Waiting before reuse is essential: non_blocking alone doesn't own
        # pinned source memory or protect it from CPU overwrites.
        if self._staging_done is not None:
            self._staging_done.synchronize()
        if self._staging is None:
            self._staging = torch.empty((self.transfer_rows, rows.shape[1]), dtype=torch.float32, pin_memory=True)
        slab = self._staging[:len(rows)]
        slab.copy_(rows)
        result = slab.to(self.device, non_blocking=True)
        self._staging_done = torch.cuda.Event()
        self._staging_done.record(torch.cuda.current_stream(self.device))
        return result

    @torch.no_grad()
    def _upload(self, ids: np.ndarray, slots: Optional[np.ndarray], target=None):
        for start in range(0, len(ids), self.transfer_rows):
            end = min(start + self.transfer_rows, len(ids))
            hidx = self._index(ids[start:end], "cpu")
            rows = self._upload_block(self.host.index_select(0, hidx))
            score = self.host_scores.index_select(0, hidx).to(self.device)
            if slots is None:
                target[0][start:end].copy_(rows)
                target[1][start:end].copy_(score)
            else:
                idx = self._index(slots[start:end])
                self.state.index_copy_(0, idx, rows)
                self.scores.index_copy_(0, idx, score)
        self.stats["uploaded_rows"] += len(ids)

    @torch.no_grad()
    def _download(self, ids: np.ndarray, rows: torch.Tensor, scores: torch.Tensor):
        # These copies are intentionally blocking before host indexing consumes
        # them. An async dirty write-back needs separate ownership/events.
        for start in range(0, len(ids), self.transfer_rows):
            end = min(start + self.transfer_rows, len(ids))
            idx = self._index(ids[start:end], "cpu")
            self.host.index_copy_(0, idx, rows[start:end].cpu())
            self.host_scores.index_copy_(0, idx, scores[start:end].cpu())
        self.stats["downloaded_rows"] += len(ids)

    @torch.no_grad()
    def _write_slots(self, slots: np.ndarray):
        # Chunk BEFORE gather; a save/eviction must not duplicate the whole pool.
        for start in range(0, len(slots), self.transfer_rows):
            chosen = slots[start:start + self.transfer_rows]
            idx = self._index(chosen)
            self._download(self.to_id[chosen], self.state.index_select(0, idx), self.scores.index_select(0, idx))
            self.dirty[chosen] = False

    @torch.no_grad()
    def _commit_active(self):
        p = self.active
        if p is None or not p.dirty:
            return
        if p.slots is None:
            self._download(p.ids, p.state, p.scores)
        else:
            self.state.index_copy_(0, p.slots, p.state)
            self.scores.index_copy_(0, p.slots, p.scores)
            self.dirty[self.to_slot[p.ids]] = True
        p.dirty = False

    def _reuse_active(self):
        self.stats["same_packet_hits"] += 1
        self.stats["hit_rows"] += len(self.active.ids)
        if self.active.slots is not None:
            self.last_used[self.to_slot[self.active.ids]] = self.clock
        else:
            self.stats["overflow_steps"] += 1
        return self.active

    @torch.no_grad()
    def acquire(self, ids, validate: bool = True) -> ActivePacket:
        cuda_ids = None
        if torch.is_tensor(ids):
            if ids.ndim != 1 or ids.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
                raise ValueError("Gaussian IDs must be a one-dimensional integer array")
            if ids.device.type == "cuda":
                cuda_ids = ids.detach()
                if (self.active is not None and self._last_cuda_ids is not None
                        and torch.equal(cuda_ids, self._last_cuda_ids)):
                    self.clock += 1
                    self.stats["requested_rows"] += len(ids)
                    return self._reuse_active()
            # Preserve int32 during transfer; expand CPU indexing only afterward.
            ids = ids.detach().cpu().numpy()
        packet = self._acquire_cpu(np.asarray(ids), validate)
        self._last_cuda_ids = cuda_ids.clone() if cuda_ids is not None else None
        return packet

    def _choose_evictions(self, needed):
        # Segmented clock/LRU approximation: scan small ring windows instead
        # of sorting/partitioning the entire multi-million-row cache per miss.
        result, scanned = [], 0
        while needed and scanned < self.used:
            count = min(max(4096, needed * 2), self.used - scanned)
            indices = (np.arange(count, dtype=np.int64) + self._eviction_cursor) % self.used
            self._eviction_cursor = int((self._eviction_cursor + count) % self.used)
            scanned += count
            candidates = indices[(self.to_id[indices] >= 0) & (self.last_used[indices] != self.clock)]
            take = min(needed, len(candidates))
            if take:
                if take < len(candidates):
                    candidates = candidates[np.argpartition(self.last_used[candidates], take - 1)[:take]]
                result.append(candidates)
                needed -= take
        self.stats["eviction_scan_rows"] += scanned
        if needed:
            raise RuntimeError("resident planner has insufficient evictable slots")
        return np.concatenate(result)

    def _acquire_cpu(self, ids, validate):
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Gaussian IDs must be a one-dimensional integer array")
        ids = ids.astype(np.int64, copy=False)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.host)):
            raise IndexError("Gaussian ID is outside the backing store")
        if validate and len(np.unique(ids)) != len(ids):
            raise ValueError("active cut contains duplicate Gaussian IDs")
        self.clock += 1
        self.stats["requested_rows"] += len(ids)
        if self.active is not None and np.array_equal(ids, self.active.ids):
            return self._reuse_active()
        self._commit_active()
        self.active = None
        if len(ids) > self.capacity:
            # Fallback is exact active-set streaming, not a coarser LoD cut.
            self.flush()
            self.invalidate()
            self.state = self.scores = None
            rows = torch.empty((len(ids), self.host.shape[1]), dtype=torch.float32, device=self.device)
            score = torch.empty(len(ids), dtype=torch.float32, device=self.device)
            self._upload(ids, None, (rows, score))
            self.active = ActivePacket(ids.copy(), None, rows, score)
            self.stats["overflow_steps"] += 1
            return self.active
        self._ensure_store()
        slots = self.to_slot[ids].copy()
        hit = slots >= 0
        self.last_used[slots[hit]] = self.clock
        self.stats["hit_rows"] += int(hit.sum())
        missing = ids[~hit]
        needed = len(missing)
        if needed:
            spare = min(needed, self.capacity - self.used)
            assigned = np.arange(self.used, self.used + spare, dtype=np.int64)
            self.used += spare
            shortage = needed - spare
            if shortage:
                evicted = self._choose_evictions(shortage)
                self._write_slots(evicted[self.dirty[evicted]])
                self.to_slot[self.to_id[evicted]] = -1
                self.stats["evicted_rows"] += len(evicted)
                assigned = np.concatenate((assigned, evicted))
            # Publish the mapping only after all rows have uploaded successfully.
            self._upload(missing, assigned)
            self.to_slot[missing] = assigned
            self.to_id[assigned] = missing
            self.dirty[assigned] = False
            slots[~hit] = assigned
        self.last_used[slots] = self.clock
        dslots = self._index(slots)
        self.active = ActivePacket(ids.copy(), dslots, self.state.index_select(0, dslots),
                                   self.scores.index_select(0, dslots))
        self.stats["packet_gathers"] += 1
        return self.active

    @torch.no_grad()
    def flush(self):
        self._commit_active()
        chosen = np.flatnonzero(self.dirty[:self.used])
        if len(chosen):
            self._write_slots(chosen)

    @torch.no_grad()
    def invalidate(self):
        """Call after flush and before/after an external topology mutation."""
        if (self.active is not None and self.active.dirty) or self.dirty[:self.used].any():
            raise RuntimeError("flush dirty Gaussian state before invalidation")
        self.active = None
        self._last_cuda_ids = None
        live = self.to_id[:self.used]
        self.to_slot[live[live >= 0]] = -1
        self.to_id[:self.used] = -1
        self.dirty[:self.used] = False
        self.last_used[:self.used] = 0
        self.used = 0

    @torch.no_grad()
    def reset_scores(self, count: int):
        """Reset a densification window after its scores have been consumed."""
        self.host_scores[:count] = 0
        if self.scores is not None:
            self.scores[:self.used] = 0
        if self.active is not None:
            self.active.scores.zero_()

    def resize_empty(self, capacity):
        """Resize only at a flushed topology barrier; never migrate dirty slots."""
        if self.used or self.active is not None or self.dirty.any():
            raise RuntimeError("flush and invalidate before resizing the resident pool")
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = min(int(capacity), len(self.host))
        self.state = self.scores = None
        self.to_id = np.full(self.capacity, -1, dtype=np.int64)
        self.last_used = np.zeros(self.capacity, dtype=np.int64)
        self.dirty = np.zeros(self.capacity, dtype=np.bool_)
        self._eviction_cursor = 0

    def close(self):
        self.flush()
        self.invalidate()
        if self._staging_done is not None:
            self._staging_done.synchronize()
        self.state = self.scores = self._staging = self._staging_done = None
