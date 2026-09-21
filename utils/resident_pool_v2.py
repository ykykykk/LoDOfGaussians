"""Resident v2: slot-indexed Adam and bounded, versioned next-view prefetch.

Speculation only uses free slots or clean cold slots. Active/dirty rows are
never evicted to prefetch; misses remain ordinary exact demand loads. CPU
copies own their pinned source until their CUDA event completes. All external
host/topology mutations must go through flush + invalidate, as in resident v1.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from utils.resident_pool import ActivePacket, ResidentPool


@dataclass
class IndexedPacket(ActivePacket):
    store: object = None
    ops: object = None
    _frozen: object = None
    _frozen_prefix: int = -1

    @property
    def width(self):
        return self.state.shape[1]

    def parameters(self):
        return self.state.detach().requires_grad_(True)

    @torch.no_grad()
    def adam_step(self, grad, rates, iteration, frozen_prefix=0):
        if grad.shape != self.state.shape or rates.shape != (self.width,) or iteration < 0:
            raise ValueError("invalid indexed Adam gradient/rates/iteration")
        if self._frozen is None or self._frozen_prefix != frozen_prefix:
            self._frozen = torch.as_tensor(self.ids < frozen_prefix, device=grad.device)
            self._frozen_prefix = frozen_prefix
        step = iteration + 1
        self.ops.indexed_adam(self.store, self.slots, self.state, grad.contiguous(), rates.contiguous(),
                              self._frozen, 1 - 0.9**step, math.sqrt(1 - 0.999**step))
        self.dirty = True


class StreamingResidentPool(ResidentPool):
    def __init__(self, *args, ops=None, prefetch_rows=131072, **kwargs):
        super().__init__(*args, **kwargs)
        if prefetch_rows < 0:
            raise ValueError("prefetch_rows must be nonnegative")
        self.ops = ops
        # At most two slabs, irrespective of scene size or requested view size.
        self.prefetch_rows = min(int(prefetch_rows), 2 * self.transfer_rows)
        self.epoch = 0
        self._copy_stream = None
        self._pending = []
        self._prefetched = set()
        self.stats.update(prefetched_rows=0, prefetch_hit_rows=0,
                          prefetch_cancelled_rows=0, prefetch_skipped_rows=0,
                          indexed_packet_gathers=0, avoided_moment_gather_bytes=0)

    def settle_prefetch(self, host=False):
        retained = []
        for event, source in self._pending:
            if host:
                event.synchronize()
            else:
                torch.cuda.current_stream(self.device).wait_event(event)
            if not host and not event.query():
                retained.append((event, source))
        self._pending = retained

    @torch.no_grad()
    def acquire(self, ids, validate=True):
        self.settle_prefetch()
        return super().acquire(ids, validate)

    @torch.no_grad()
    def _commit_active(self):
        packet = self.active
        if isinstance(packet, IndexedPacket):
            if packet.dirty:
                # Parameter/m/v writes were already made in-place by the CUDA
                # optimizer. Only scores need a scatter here (also on split steps).
                self.scores.index_copy_(0, packet.slots, packet.scores)
                self.dirty[self.to_slot[packet.ids]] = True
                packet.dirty = False
        else:
            super()._commit_active()

    def _acquire_cpu(self, ids, validate):
        # Track actual use of speculative rows even on the reference path.
        if ids.ndim == 1 and ids.dtype.kind in "iu" and self._prefetched:
            used = self._prefetched.intersection(ids.tolist())
            self.stats["prefetch_hit_rows"] += len(used)
            self._prefetched.difference_update(used)
        if self.ops is None or len(ids) > self.capacity:
            return super()._acquire_cpu(ids, validate)
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
        self._ensure_store()
        slots = self.to_slot[ids].copy()
        hit = slots >= 0
        self.last_used[slots[hit]] = self.clock
        self.stats["hit_rows"] += int(hit.sum())
        missing = ids[~hit]
        if len(missing):
            spare = min(len(missing), self.capacity - self.used)
            assigned = np.arange(self.used, self.used + spare, dtype=np.int64)
            self.used += spare
            shortage = len(missing) - spare
            if shortage:
                evicted = self._choose_evictions(shortage)
                self._write_slots(evicted[self.dirty[evicted]])
                self._prefetched.difference_update(self.to_id[evicted].tolist())
                self.to_slot[self.to_id[evicted]] = -1
                self.stats["evicted_rows"] += len(evicted)
                assigned = np.concatenate((assigned, evicted))
            self._upload(missing, assigned)
            self.to_slot[missing] = assigned
            self.to_id[assigned] = missing
            self.dirty[assigned] = False
            slots[~hit] = assigned
        self.last_used[slots] = self.clock
        dslots = self._index(slots)
        raw = self.ops.gather_parameters(self.state, dslots)
        self.active = IndexedPacket(ids.copy(), dslots, raw, self.scores.index_select(0, dslots),
                                    store=self.state, ops=self.ops)
        self.stats["packet_gathers"] += 1
        self.stats["indexed_packet_gathers"] += 1
        self.stats["avoided_moment_gather_bytes"] += len(ids) * (self.host.shape[1] // 3) * 8
        return self.active

    def _prefetch_plan(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("prefetch IDs must be a one-dimensional integer array")
        ids = ids.astype(np.int64, copy=False)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.host)):
            raise IndexError("prefetch ID outside backing store")
        requested_slots = self.to_slot[ids]
        # Reserve next-view hits too: speculation must not evict a row that the
        # same predicted view already has, just to bring another one in.
        self.last_used[requested_slots[requested_slots >= 0]] = self.clock
        missing = ids[requested_slots < 0]
        request_count = len(missing)
        # Deduplicate only the bounded speculative prefix, not millions of
        # already-resident IDs. Demand loading will still handle the full cut.
        missing = missing[:2 * self.prefetch_rows]
        _, first = np.unique(missing, return_index=True)
        missing = missing[np.sort(first)][:self.prefetch_rows]
        free = min(len(missing), self.capacity - self.used)
        assigned = np.arange(self.used, self.used + free, dtype=np.int64)
        shortage = len(missing) - free
        evicted = np.empty(0, dtype=np.int64)
        if shortage and self.used:
            # Speculative work is bounded. Never force dirty D2H writeback or a
            # full-cache scan just to anticipate a future view.
            count = min(self.used, max(4096, shortage * 2))
            scan = (np.arange(count) + self._eviction_cursor) % self.used
            clean = scan[(~self.dirty[scan]) & (self.last_used[scan] != self.clock)]
            evicted = clean[:shortage]
            assigned = np.concatenate((assigned, evicted))
        missing = missing[:len(assigned)]
        self.stats["prefetch_skipped_rows"] += request_count - len(missing)
        return missing, assigned, free, evicted

    @torch.no_grad()
    def prefetch(self, ids, epoch=None):
        if epoch is not None and epoch != self.epoch:
            return 0
        if not self.prefetch_rows or self.state is None or self.active is None or self.active.slots is None:
            return 0
        # Don't stall the host to recycle a slab that a previous transfer owns.
        self._pending = [(e, s) for e, s in self._pending if not e.query()]
        if self._pending:
            return 0
        if torch.is_tensor(ids):
            if (ids.is_cuda and self._last_cuda_ids is not None
                    and torch.equal(ids, self._last_cuda_ids)):
                return 0  # Same active cut: no host ID readback or miss planning.
            ids = ids.detach().cpu().numpy()
        missing, assigned, free, evicted = self._prefetch_plan(ids)
        if not len(missing):
            return 0
        if self.device.type == "cuda" and self.pin_staging:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=self.device)
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            self.state.record_stream(self._copy_stream)
            self.scores.record_stream(self._copy_stream)
            for start in range(0, len(missing), self.transfer_rows):
                stop = min(start + self.transfer_rows, len(missing))
                hidx = self._index(missing[start:stop], "cpu")
                slab = torch.empty((stop - start, self.host.shape[1] + 1), dtype=torch.float32, pin_memory=True)
                slab[:, :-1].copy_(self.host.index_select(0, hidx))
                slab[:, -1].copy_(self.host_scores.index_select(0, hidx))
                slot_source = torch.empty(stop-start, dtype=torch.int64, pin_memory=True)
                slot_source.copy_(torch.from_numpy(assigned[start:stop]))
                with torch.cuda.stream(self._copy_stream):
                    rows = slab.to(self.device, non_blocking=True)
                    # _index(..., cuda) performs a blocking upload; do not use
                    # it after queuing the speculative copy in this stream.
                    dslots = slot_source.to(self.device, non_blocking=True)
                    self.state.index_copy_(0, dslots, rows[:, :-1])
                    self.scores.index_copy_(0, dslots, rows[:, -1])
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                self._pending.append((event, (slab, slot_source)))
            self.stats["uploaded_rows"] += len(missing)
        else:
            # CPU tests and pin_staging=false have identical cache semantics,
            # but explicitly do not claim copy/compute overlap.
            self._upload(missing, assigned)
        # Pending mappings are visible only through acquire(), which waits for
        # their events. Slot planning excludes the current active set.
        if len(evicted):
            self._prefetched.difference_update(self.to_id[evicted].tolist())
            self.to_slot[self.to_id[evicted]] = -1
            self.stats["evicted_rows"] += len(evicted)
        self.used += free
        self.to_slot[missing] = assigned
        self.to_id[assigned] = missing
        self.last_used[assigned] = self.clock
        self.dirty[assigned] = False
        self._prefetched.update(missing.tolist())
        self.stats["prefetched_rows"] += len(missing)
        return len(missing)

    @torch.no_grad()
    def flush(self):
        self.settle_prefetch(host=True)
        super().flush()

    @torch.no_grad()
    def invalidate(self):
        self.settle_prefetch(host=True)
        super().invalidate()
        self.stats["prefetch_cancelled_rows"] += len(self._prefetched)
        self._prefetched.clear()
        self.epoch += 1

    def close(self):
        super().close()
        self._copy_stream = None
