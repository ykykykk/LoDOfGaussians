import random
import threading

import numpy as np
import pytest
import torch

from utils.paged_camera_prefetch import PagedCameraPrefetch, scheduled_camera


class Cache:
    def __init__(self, worker, budget):
        self.worker, self.budget = worker, budget
        self.threads = set()
        self.calls = []

    def __len__(self):
        return 13

    def __getitem__(self, index):
        self.threads.add(threading.get_ident())
        self.calls.append(index)
        return index


def collect(start, end, workers=2):
    with PagedCameraPrefetch(range(13), 17, start, end, 4,
                            cache_bytes=101, workers=workers,
                            cache_factory=Cache) as prefetch:
        result = []
        for step in range(start, end):
            result.append(prefetch.get(step))
            assert len(prefetch.futures) <= workers
            assert all(group <= prefetch.group + workers
                       for group in prefetch.futures)
        caches = prefetch.cameras
    return result, caches


@pytest.mark.parametrize('workers', [1, 2, 3])
def test_sequence_and_midgroup_resume(workers):
    full, caches = collect(101, 157, workers)
    resumed, _ = collect(119, 157, workers)
    assert full == [scheduled_camera(17, step // 4, 13) for step in range(101, 157)]
    assert resumed == full[18:]
    assert sum(cache.budget for cache in caches) == 101
    assert all(len(cache.threads) == 1 for cache in caches)
    assert len(set.union(*(cache.threads for cache in caches))) == workers
    for cache in caches:
        expected = [scheduled_camera(17, group, 13)
                    for group in range(101 // 4, (157 - 1) // 4 + 1)
                    if group % workers == cache.worker]
        assert cache.calls == expected


def test_does_not_change_rng():
    py = random.getstate()
    np_state = np.random.get_state()
    tensor_state = torch.random.get_rng_state()
    collect(10, 36)
    assert random.getstate() == py
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]
    assert torch.equal(tensor_state, torch.random.get_rng_state())


def test_bounds_and_close():
    prefetch = PagedCameraPrefetch(range(13), 17, 10, 12, 4, cache_factory=Cache)
    with pytest.raises(RuntimeError, match='sequential'):
        prefetch.get(11)
    prefetch.get(10)
    prefetch.get(11)
    with pytest.raises(RuntimeError, match='sequential'):
        prefetch.get(12)
    prefetch.close()
    prefetch.close()
    assert not prefetch.futures
    with pytest.raises(RuntimeError, match='closed'):
        prefetch.get(12)


def test_actual_caches_are_private_and_bounded():
    from types import SimpleNamespace

    class Dataset:
        def __len__(self):
            return 13

        def __getitem__(self, index):
            return SimpleNamespace(original_image=torch.zeros(3, 2, 2, dtype=torch.uint8),
                                   index=index)

    with PagedCameraPrefetch(Dataset(), 17, 0, 20, 4, cache_bytes=25) as prefetch:
        for step in range(20):
            camera = prefetch.get(step)
            assert camera.index == scheduled_camera(17, step // 4, 13)
    assert prefetch.cameras[0].cache is not prefetch.cameras[1].cache
    assert sum(cache.max_bytes for cache in prefetch.cameras) == 25
    assert all(cache.used_bytes <= cache.max_bytes for cache in prefetch.cameras)
