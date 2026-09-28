"""Camera lookahead must not advance stochastic optimizer/checkpoint state."""
import threading

import pytest
import torch

from train_paged import CameraLookahead, scheduled_camera


class Cameras:
    def __init__(self, count=11):
        self.count = count
        self.calls = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.calls.append((index, threading.get_ident(), threading.current_thread().name))
        return index


def collect(start, end, group_size=4):
    cameras = Cameras()
    worker = CameraLookahead(cameras, 7, start, end, group_size)
    try:
        result = [worker.get(step) for step in range(start, end)]
    finally:
        worker.close()
    return result, cameras


def test_group_schedule_replays_after_resume_inside_group():
    full, _ = collect(47001, 47019)
    resumed, _ = collect(47009, 47019)
    assert resumed == full[8:]
    assert full == [scheduled_camera(7, step//4, 11) for step in range(47001, 47019)]


def test_cache_has_one_worker_owner_and_one_decode_per_group():
    values, cameras = collect(47001, 47019)
    threads = {thread_id for _, thread_id, _ in cameras.calls}
    assert len(threads) == 1
    assert threading.get_ident() not in threads
    assert all(name.startswith('paged-image') for _, _, name in cameras.calls)
    assert len(cameras.calls) == (47018//4) - (47001//4) + 1
    assert len(values) == 18


def test_prefetch_and_resume_do_not_consume_torch_rng():
    torch.manual_seed(921)
    before = torch.get_rng_state().clone()
    collect(47001, 47019)
    collect(47009, 47019)
    assert torch.equal(before, torch.get_rng_state())


def test_prefetch_is_bounded_to_one_future_group():
    entered = threading.Event()
    release = threading.Event()

    class BlockingCameras(Cameras):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            if len(self.calls) == 2:
                entered.set()
                assert release.wait(5), 'Test did not release lookahead worker'
            return result

    cameras = BlockingCameras()
    worker = CameraLookahead(cameras, 7, 0, 40, 4)
    try:
        first = worker.get(0)
        assert entered.wait(5)
        assert len(cameras.calls) == 2
        assert worker.get(1) == first
        assert worker.get(2) == first
        assert len(cameras.calls) == 2
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize('group,count', [(-1, 11), (0, 0)])
def test_schedule_rejects_invalid_dimensions(group, count):
    with pytest.raises(ValueError):
        scheduled_camera(7, group, count)
