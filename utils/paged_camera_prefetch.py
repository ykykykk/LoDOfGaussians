"""Bounded camera-group decoding with one private cache per worker."""
import hashlib
from concurrent.futures import ThreadPoolExecutor


def scheduled_camera(seed, group, camera_count):
    """Checkpoint-stable schedule, independent of global random state."""
    if camera_count <= 0 or group < 0:
        raise ValueError('Invalid camera schedule dimensions')
    digest = hashlib.sha256(f'paged-camera-v1:{seed}:{group}'.encode('ascii')).digest()
    return int.from_bytes(digest[:8], 'little') % camera_count


class PagedCameraPrefetch:
    """Decode at most ``workers`` future groups plus the current camera.

    ``start`` is an inclusive zero-based step and ``end`` is exclusive.
    Each single-thread executor exclusively owns its CachedCameras instance;
    the configured total cache budget is divided across workers. The optional
    factory receives (worker_index, cache_budget_bytes), primarily for testing.
    """
    def __init__(self, dataset, seed, start, end, tiles_per_camera,
                 cache_bytes=0, compact_images=True, workers=2,
                 cache_factory=None):
        if workers < 1 or tiles_per_camera < 1 or start < 0 or end <= start:
            raise ValueError('Invalid camera prefetch dimensions')
        if len(dataset) <= 0 or cache_bytes < 0:
            raise ValueError('Empty dataset or negative cache budget')
        self.seed = seed
        self.tiles_per_camera = tiles_per_camera
        self.start, self.end = start, end
        self.workers = workers
        self.last_group = (end - 1) // tiles_per_camera
        self.group = None
        self.camera = None
        self.closed = False
        self.last_step = start - 1
        self.futures = {}
        if cache_factory is None:
            from utils.view_pipeline import CachedCameras
            cache_factory = lambda worker, budget: CachedCameras(
                dataset, budget, compact_images=compact_images)
        budgets = [cache_bytes // workers + (i < cache_bytes % workers)
                   for i in range(workers)]
        self.cameras = [cache_factory(i, budget) for i, budget in enumerate(budgets)]
        self.executors = [ThreadPoolExecutor(max_workers=1,
                          thread_name_prefix=f'paged-image-{i}')
                          for i in range(workers)]
        self.next_group = start // tiles_per_camera
        self._fill(self.next_group + workers - 1)

    def _decode(self, group):
        cache = self.cameras[group % self.workers]
        return cache[scheduled_camera(self.seed, group, len(cache))]

    def _fill(self, through):
        while self.next_group <= min(through, self.last_group):
            group = self.next_group
            self.futures[group] = self.executors[group % self.workers].submit(
                self._decode, group)
            self.next_group += 1

    def get(self, zero_based_step):
        if self.closed:
            raise RuntimeError('Camera prefetch is closed')
        if zero_based_step != self.last_step + 1 or zero_based_step >= self.end:
            raise RuntimeError('Camera prefetch requires sequential steps within its range')
        group = zero_based_step // self.tiles_per_camera
        if group != self.group:
            self.camera = self.futures.pop(group).result()
            self.group = group
            self._fill(group + self.workers)
        self.last_step = zero_based_step
        return self.camera

    def close(self):
        if self.closed:
            return
        self.closed = True
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
        self.camera = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
