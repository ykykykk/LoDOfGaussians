"""Bounded decoded-camera caching, one-view CPU lookahead and CUDA upload.

The worker thread only fetches CPU data. All CUDA submissions happen on the
training thread. Camera objects are copied before device/pinning mutations.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
import torch

TENSORS = ("K_train", "original_image", "alpha_mask", "invdepthmap", "depth_mask",
           "world_view_transform", "projection_matrix", "full_proj_transform",
           "full_proj_transform_inverse", "camera_center")
METADATA = ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")


def camera_bytes(camera):
    seen, size = set(), 0
    for name in TENSORS:
        value = getattr(camera, name, None)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen:
                size += storage.nbytes()
                seen.add(key)
    return size


class CachedCameras(torch.utils.data.Dataset):
    """Per-worker LRU, with the total configured budget divided by workers."""
    def __init__(self, dataset, max_bytes):
        self.dataset = dataset
        self.max_bytes = max(0, int(max_bytes))
        self.cache = OrderedDict()
        self.used_bytes = 0
        self.hits = self.misses = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index in self.cache:
            self.hits += 1
            camera, size = self.cache.pop(index)
            self.cache[index] = (camera, size)
            return copy(camera)
        self.misses += 1
        camera = self.dataset[index]
        if any(isinstance(getattr(camera, n, None), torch.Tensor)
               and getattr(camera, n).device.type != "cpu" for n in TENSORS):
            raise ValueError("camera decoding/cache must stay on CPU")
        size = camera_bytes(camera)
        if 0 < size <= self.max_bytes:
            while self.cache and self.used_bytes + size > self.max_bytes:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.used_bytes -= old_size
            self.cache[index] = (copy(camera), size)
            self.used_bytes += size
        return copy(camera)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(cache=OrderedDict(), used_bytes=0, hits=0, misses=0)
        return state


class ThreadedViews:
    """Exactly one CPU lookahead future; preserves the DataLoader's order."""
    def __init__(self, loader, count, enabled=True):
        if count < 0 or len(loader) == 0:
            raise ValueError("invalid view count or empty loader")
        self.loader, self.count, self.enabled = loader, int(count), enabled
        self.iterator = None
        self.taken = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alod-view") if enabled else None
        self.future = self.executor.submit(self._read) if enabled and count else None

    def _read(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)[0]
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)[0]

    def pop(self, block=True):
        if self.taken >= self.count:
            return None
        if not self.enabled:
            if not block:
                return None
            value = self._read()
        else:
            if not block and not self.future.done():
                return None
            value = self.future.result()
        self.taken += 1
        self.future = (self.executor.submit(self._read)
                       if self.enabled and self.taken < self.count else None)
        return value

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = self.iterator = None


@dataclass
class ViewTicket:
    cpu: object
    camera: object = None
    event: object = None
    selection_camera: object = None
    ids: object = None
    epoch: int = -1
    multiplier: object = None


class CameraTransfer:
    def __init__(self, device="cuda", prefetch_bytes=512 * 2**20):
        self.device = torch.device(device)
        self.prefetch_bytes = int(prefetch_bytes)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.pending = []
        self.uploaded_bytes = 0
        self.prefetch_uploads = 0

    def _upload(self, ticket, speculative):
        size = camera_bytes(ticket.cpu)
        if speculative and size > self.prefetch_bytes:
            return
        result = copy(ticket.cpu)
        sources = []
        if self.stream is None:
            ticket.camera = result
            return
        self.pending = [(e, s) for e, s in self.pending if not e.query()]
        # The source object may come from a persistent decoded cache. Never
        # replace its CPU tensor fields in-place with CUDA tensors.
        for name in TENSORS:
            tensor = getattr(ticket.cpu, name, None)
            if isinstance(tensor, torch.Tensor):
                if tensor.device.type != "cpu":
                    raise ValueError("expected a CPU camera")
                sources.append((name, tensor if tensor.is_pinned() else tensor.pin_memory()))
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for name, tensor in sources:
                setattr(result, name, tensor.to(self.device, non_blocking=True))
            ticket.event = torch.cuda.Event()
            ticket.event.record(self.stream)
        self.pending.append((ticket.event, sources))
        ticket.camera = result
        self.uploaded_bytes += size
        self.prefetch_uploads += int(speculative)

    def submit(self, camera, speculative=True):
        ticket = ViewTicket(camera)
        self._upload(ticket, speculative)
        return ticket

    def ready(self, ticket):
        if ticket.camera is None:
            self._upload(ticket, False)
        if ticket.event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(ticket.event)
            for name in TENSORS:
                tensor = getattr(ticket.camera, name, None)
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor.record_stream(current)
        return ticket.camera

    def metadata(self, ticket):
        # Select next-view Gaussians without waiting for the large image copy
        # in the transfer stream. Only four tiny camera tensors are duplicated.
        if ticket.selection_camera is None:
            camera = copy(ticket.cpu)
            for name in METADATA:
                setattr(camera, name, getattr(camera, name).to(self.device, non_blocking=True))
            ticket.selection_camera = camera
        return ticket.selection_camera

    def close(self):
        for event, _ in self.pending:
            event.synchronize()
        self.pending.clear()
        self.stream = None


class ViewSchedule(torch.utils.data.Sampler):
    """Private RNGs keep lookahead on/off from changing view/RNG order."""
    def __init__(self, cameras, count, seed, graph=None):
        self.cameras, self.count, self.seed, self.graph = int(cameras), int(count), int(seed), graph
        if cameras <= 0 or count < 0:
            raise ValueError("invalid camera schedule")

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.graph is not None:
            import random
            rng = random.Random(self.seed)
            current = list(self.graph.nodes())[0]
            for i in range(self.count):
                neighbors = list(self.graph.neighbors(current))
                if not neighbors:
                    raise ValueError("camera view graph has an isolated vertex")
                weights = [1.0 / (self.graph[current][other].get('weight', 1) + 20) for other in neighbors]
                chosen = int(rng.choices(neighbors, weights=weights, k=1)[0])
                if not 0 <= chosen < self.cameras:
                    raise ValueError("view graph index outside the camera dataset")
                yield chosen
                current = rng.randrange(self.cameras) if i % 100 == 0 else chosen
        else:
            rng = torch.Generator().manual_seed(self.seed)
            remaining = self.count
            while remaining:
                order = torch.randperm(self.cameras, generator=rng).tolist()
                take = min(remaining, len(order))
                yield from order[:take]
                remaining -= take


def make_view_loader(cameras, opt, cache_bytes, seed, view_graph=None):
    from utils.training_runtime import direct_collate
    workers = int(getattr(opt, 'data_workers', 4))
    factor = int(getattr(opt, 'data_prefetch_factor', 1))
    if workers < 0 or factor < 1:
        raise ValueError("invalid DataLoader worker/prefetch configuration")
    dataset = CachedCameras(cameras, max(0, int(cache_bytes)) // max(1, workers))
    kwargs = dict(batch_size=1, num_workers=workers, collate_fn=direct_collate,
                  sampler=ViewSchedule(len(cameras), opt.iterations+1, seed, view_graph),
                  pin_memory=bool(getattr(opt, 'pin_memory', True)),
                  generator=torch.Generator().manual_seed(seed ^ 0x5A17))
    if workers:
        kwargs.update(prefetch_factor=factor, persistent_workers=True)
    return torch.utils.data.DataLoader(dataset, **kwargs)
