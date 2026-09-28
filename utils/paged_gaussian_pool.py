"""Bounded whole-block GPU cache for parent-free, disk-backed Gaussian training.

Only block metadata scales with the scene. Oversized requests fail before cache
mutation; callers must subdivide the image, never discard visible Gaussians.
"""
from collections import OrderedDict
import math
import numpy as np
import torch
from utils.resident_pool_v2 import IndexedPacket


class CapacityError(RuntimeError):
    def __init__(self, required_rows, capacity_rows):
        self.required_rows, self.capacity_rows = required_rows, capacity_rows
        super().__init__(f'View requires {required_rows} cache rows; capacity is {capacity_rows}')


class PagedPacket(IndexedPacket):
    @torch.no_grad()
    def adam_step(self, grad, rates, iteration, frozen_prefix=0):
        # Background membership is per block, not a global contiguous ID prefix.
        if grad.shape != self.state.shape or rates.shape != (self.width,) or iteration < 0:
            raise ValueError('Invalid paged Adam inputs')
        step = iteration + 1
        if self.ops is not None:
            self.ops.indexed_adam(self.store, self.slots, self.state, grad.contiguous(),
                rates.contiguous(), self._frozen, 1 - .9**step, math.sqrt(1 - .999**step))
        else:
            rows = self.store[self.slots]
            p, m, v = rows.split(self.width, 1)
            frozen_rows = rows[self._frozen].clone()
            grad = grad.clone()
            grad[self._frozen] = 0
            m.mul_(.9).add_(grad, alpha=.1)
            v.mul_(.999).addcmul_(grad, grad, value=.001)
            p.sub_(m * (rates / (1 - .9**step)) / (v.sqrt() / math.sqrt(1 - .999**step) + 1e-8))
            rows[self._frozen] = frozen_rows
            self.store[self.slots] = rows
            self.state.copy_(p)
        self.dirty = True


class PagedGaussianPool:
    def __init__(self, store, capacity_rows, ops=None, device='cuda'):
        self.store, self.ops, self.device = store, ops, torch.device(device)
        self.block_rows = int(store.block_rows)
        self.pages = int(capacity_rows) // self.block_rows
        if self.pages < 1:
            raise ValueError('Cache must hold at least one complete block')
        self.capacity = self.pages * self.block_rows
        self.capacity_rows = self.capacity
        self.state = torch.empty((self.capacity, 69), device=self.device, dtype=torch.float32)
        self.scores = torch.empty(self.capacity, device=self.device, dtype=torch.float32)
        self.resident = OrderedDict()
        self.free = list(reversed(range(self.pages)))
        self.dirty = set()
        self.score_dirty = set()
        self.active = None
        self._writes = 0
        self.stats = dict(hit_blocks=0, uploaded_rows=0, downloaded_rows=0, evictions=0,
                          requested_blocks=0, uploaded_blocks=0)
        self.refresh_metadata()

    def refresh_metadata(self, preserve_resident=False):
        if self.resident and not preserve_resident:
            raise RuntimeError('Clear cache before refreshing topology')
        # Dirty page bounds are newer than disk metadata. Keep them across append.
        previous = {bid: self.bounds[i].clone() for bid, i in self.index.items()} if preserve_resident else {}
        self.metadata = {b['id']: b for b in self.store.blocks}
        self.block_ids = list(self.metadata)
        self.index = {bid: i for i, bid in enumerate(self.block_ids)}
        b = [list(self.metadata[bid]['bounds_min']) + list(self.metadata[bid]['bounds_max']) for bid in self.block_ids]
        self.bounds = torch.tensor(b, dtype=torch.float32, device=self.device).reshape(-1, 6)
        self.background = torch.tensor([bool(self.metadata[bid].get('skybox', False)) for bid in self.block_ids], device=self.device)
        for bid in self.dirty:
            if bid in previous:
                self.bounds[self.index[bid]].copy_(previous[bid])

    @staticmethod
    def _planes(camera, device):
        m = camera.full_proj_transform.to(device).T
        p = torch.stack((m[3]+m[0], m[3]-m[0], m[3]+m[1], m[3]-m[1]))
        return (p / p[:, :3].norm(dim=1, keepdim=True).clamp_min(1e-20)).contiguous()

    @torch.no_grad()
    def candidate_blocks(self, camera):
        self.commit_active()
        p = self._planes(camera, self.device)
        center = (self.bounds[:, :3] + self.bounds[:, 3:]) * .5
        half = (self.bounds[:, 3:] - self.bounds[:, :3]) * .5
        visible = ((center @ p[:, :3].T + half @ p[:, :3].abs().T + p[:, 3]) >= 0).all(1)
        visible |= self.background
        return [self.block_ids[i] for i in visible.nonzero().flatten().cpu().tolist()]

    def _slice(self, bid):
        start = self.resident[bid] * self.block_rows
        return slice(start, start + int(self.metadata[bid]['count']))

    @torch.no_grad()
    def _write(self, bid):
        if bid not in self.dirty and bid not in self.score_dirty:
            return
        s = self._slice(bid)
        if bid not in self.dirty:
            self.store.write_scores(bid, self.scores[s].cpu().numpy())
            self.score_dirty.discard(bid)
            return
        rows = self.state[s].cpu().numpy()
        scores = self.scores[s].cpu().numpy()
        self.store.write(bid, rows, scores=scores)
        self.stats['downloaded_rows'] += len(rows)
        self.dirty.remove(bid)
        self.score_dirty.discard(bid)
        self._writes += 1
        if self._writes % 64 == 0:
            self.store.garbage_collect()

    @torch.no_grad()
    def ensure(self, block_ids):
        requested = list(dict.fromkeys(block_ids))
        for bid in requested:
            if bid not in self.metadata:
                raise KeyError(bid)
            if not 0 < int(self.metadata[bid]['count']) <= self.block_rows:
                raise ValueError('Invalid block count')
        if len(requested) > self.pages:
            raise CapacityError(len(requested) * self.block_rows, self.capacity)
        self.commit_active()
        keep = set(requested)
        self.stats['requested_blocks'] += len(requested)
        for bid in requested:
            if bid in self.resident:
                self.resident.move_to_end(bid)
                self.stats['hit_blocks'] += 1
                continue
            if not self.free:
                old = next(b for b in self.resident if b not in keep)
                self._write(old)
                self.free.append(self.resident.pop(old))
                self.stats['evictions'] += 1
            page = self.free.pop()
            # Own the host copy; readonly mappings must not outlive asynchronous transfers.
            rows = torch.from_numpy(np.array(self.store.read(bid), dtype=np.float32, copy=True))
            scores = torch.from_numpy(np.array(self.store.read_scores(bid), dtype=np.float32, copy=True))
            if rows.shape != (int(self.metadata[bid]['count']), 69) or scores.shape != (len(rows),):
                self.free.append(page)
                raise ValueError('Invalid block storage shape')
            start = page * self.block_rows
            self.state[start:start+len(rows)].copy_(rows)
            self.scores[start:start+len(rows)].copy_(scores)
            self.resident[bid] = page
            self.stats['uploaded_rows'] += len(rows)
            self.stats['uploaded_blocks'] += 1
        return requested

    @torch.no_grad()
    def acquire(self, block_ids, camera=None):
        requested = self.ensure(block_ids)
        if camera is not None and self.ops is not None and hasattr(self.ops, 'paged_visible'):
            counts = [0] * self.pages
            mapping = [-1] * self.pages
            selected = [False] * self.pages
            sky = [False] * self.pages
            for bid in requested:
                page = self.resident[bid]
                counts[page] = int(self.metadata[bid]['count'])
                mapping[page] = self.index[bid]
                selected[page] = True
                sky[page] = bool(self.metadata[bid].get('skybox', False))
            counts = torch.tensor(counts, dtype=torch.long, device=self.device)
            mapping = torch.tensor(mapping, dtype=torch.long, device=self.device)
            selected = torch.tensor(selected, dtype=torch.bool, device=self.device)
            sky = torch.tensor(sky, dtype=torch.bool, device=self.device)
            mask = self.ops.paged_visible(self.state, counts, selected, sky,
                                          self.block_rows, self._planes(camera, self.device))
            slots = mask.nonzero().flatten()
            raw = self.ops.gather_parameters(self.state, slots)
            packet = PagedPacket(ids=slots, slots=slots, state=raw, scores=self.scores[slots],
                store=self.state, ops=self.ops, _frozen=sky[slots // self.block_rows], _frozen_prefix=0)
            packet.block_ids = requested
            packet.page_metadata = (counts, mapping, selected)
            self.active = packet
            return packet
        parts, frozen = [], []
        p = None if camera is None else self._planes(camera, self.device)
        for bid in requested:
            s = self._slice(bid)
            ids = torch.arange(s.start, s.stop, device=self.device, dtype=torch.long)
            sky = bool(self.metadata[bid].get('skybox', False))
            if p is not None and not sky:
                raw = self.state[s, :6]
                b = torch.cat((raw[:, :3], raw[:, 3:6].amax(1, keepdim=True).exp()*3), 1)
                mask = self.ops.flat_visible(b, p) if self.ops is not None else ((b[:, :3] @ p[:, :3].T + p[:, 3] + b[:, 3:]) >= 0).all(1)
                ids = ids[mask]
            parts.append(ids)
            frozen.append(torch.full((len(ids),), sky, dtype=torch.bool, device=self.device))
        slots = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=self.device)
        freeze = torch.cat(frozen) if frozen else torch.empty(0, dtype=torch.bool, device=self.device)
        raw = self.ops.gather_parameters(self.state, slots) if self.ops is not None else self.state[slots, :23].contiguous()
        packet = PagedPacket(ids=slots, slots=slots, state=raw, scores=self.scores[slots],
                             store=self.state, ops=self.ops, _frozen=freeze, _frozen_prefix=0)
        packet.block_ids = requested
        self.active = packet
        return packet

    @torch.no_grad()
    def commit_active(self):
        packet = self.active
        if packet is not None and packet.dirty:
            self.scores.index_copy_(0, packet.slots, packet.scores)
            native_bounds = hasattr(packet, 'page_metadata') and hasattr(self.ops, 'recompute_page_bounds')
            if native_bounds:
                counts, mapping, selected = packet.page_metadata
                self.ops.recompute_page_bounds(self.state, counts, mapping, selected,
                                                self.block_rows, self.bounds)
            for bid in packet.block_ids:
                if self.metadata[bid].get('skybox', False):
                    continue
                self.dirty.add(bid)
                if native_bounds:
                    continue
                raw = self.state[self._slice(bid), :6]
                r = raw[:, 3:6].amax(1, keepdim=True).exp()*3
                self.bounds[self.index[bid], :3] = (raw[:, :3]-r).amin(0)
                self.bounds[self.index[bid], 3:] = (raw[:, :3]+r).amax(0)
            packet.dirty = False
        self.active = None

    def finish_step(self, packet=None):
        if packet is not None and packet is not self.active:
            raise ValueError('Packet is no longer active')
        self.commit_active()

    def flush(self):
        self.commit_active()
        for bid in list(self.dirty | self.score_dirty):
            self._write(bid)

    @torch.no_grad()
    def densify(self, opt):
        """Grow from current pages, retaining cache and deferring resident writes.

        As with Adam, flush before checkpointing the underlying store. Failure
        aborts this training transaction; the published manifest is unchanged.
        """
        from utils.block_densification import densify_blocks
        self.commit_active()
        # Growth groups child pages spatially; use live bounds for that ordering.
        dirty_ids = list(self.dirty)
        if dirty_ids:
            indices = torch.tensor([self.index[bid] for bid in dirty_ids], device=self.device)
            bounds = self.bounds.index_select(0, indices).cpu().numpy()
            for bid, bound in zip(dirty_ids, bounds):
                self.store.update_bounds(bid, bound[:3], bound[3:])
        result = densify_blocks(_GrowthStore(self), opt)
        self.refresh_metadata(preserve_resident=True)
        return result

    def clear(self):
        self.flush()
        self.resident.clear()
        self.free = list(reversed(range(self.pages)))
        self.refresh_metadata()

    invalidate = clear


class _GrowthStore:
    """Block-bounded view of the latest CPU/disk and resident parameter state."""
    def __init__(self, pool):
        self.pool = pool
        self.store = pool.store
        self.blocks = self.store.blocks
        self.block_rows = pool.block_rows

    def read(self, bid):
        if bid in self.pool.resident:
            rows = self.pool.state[self.pool._slice(bid)].cpu().numpy().copy()
            self.pool.stats['downloaded_rows'] += len(rows)
            return rows
        return self.store.read(bid)

    def read_scores(self, bid):
        if bid in self.pool.resident:
            return self.pool.scores[self.pool._slice(bid)].cpu().numpy().copy()
        return self.store.read_scores(bid)

    def write_scores(self, bid, scores):
        if bid not in self.pool.resident:
            return self.store.write_scores(bid, scores)
        value = np.asarray(scores)
        if value.shape != (self.blocks[bid]['count'],) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected finite float32 scores [N]')
        self.pool.scores[self.pool._slice(bid)].copy_(torch.from_numpy(value))
        self.pool.score_dirty.add(bid)

    def write(self, bid, rows, scores=None):
        if bid not in self.pool.resident:
            return self.store.write(bid, rows, scores=scores)
        value = rows.detach().cpu().numpy() if isinstance(rows, torch.Tensor) else np.asarray(rows)
        if value.ndim != 2 or value.shape[1] != 69 or not 0 < len(value) <= self.block_rows or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected block-bounded finite float32 [N,69] rows')
        radius = 3 * np.exp(value[:, 3:6].astype(np.float64)).max(axis=1)
        low = (value[:, :3] - radius[:, None]).min(axis=0)
        high = (value[:, :3] + radius[:, None]).max(axis=0)
        self.store.update_bounds(bid, low, high)
        old_count = self.blocks[bid]['count']
        self.blocks[bid]['count'] = len(value)
        s = self.pool._slice(bid)
        self.pool.state[s].copy_(torch.from_numpy(value))
        self.pool.stats['uploaded_rows'] += len(value)
        if scores is not None:
            self.write_scores(bid, scores)
        elif old_count != len(value):
            self.pool.scores[s].zero_()
        self.pool.bounds[self.pool.index[bid]].copy_(torch.as_tensor(np.concatenate((low, high)), device=self.pool.device))
        self.pool.dirty.add(bid)

    def append(self, rows, skybox=False, scores=None):
        return self.store.append(rows, skybox=skybox, scores=scores)
