"""Sampled CPU enqueue time and CUDA-event spans; no per-stage synchronizes."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch


class TrainingProfile:
    def __init__(self, path, every=100, device="cuda"):
        if every < 0:
            raise ValueError("profile interval must be nonnegative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.every = int(every)
        self.device = torch.device(device)
        self.pending = []
        self.current = None
        self.started = time.perf_counter()

    def begin(self, iteration):
        self.drain()
        enabled = self.every and iteration % self.every == 0
        self.current = dict(iteration=iteration, phases={}) if enabled else None

    @contextmanager
    def phase(self, name):
        row = self.current
        if row is None:
            yield
            return
        a = b = None
        if self.device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        try:
            with torch.profiler.record_function("alod/" + name):
                yield
        finally:
            host_ms = (time.perf_counter() - start) * 1000
            if b is not None:
                b.record(torch.cuda.current_stream(self.device))
            row["phases"][name] = (host_ms, a, b)

    def finish(self, **counters):
        if self.current is not None:
            self.current.update(counters)
            self.current["elapsed_wall_s"] = time.perf_counter() - self.started
            self.pending.append(self.current)
            self.current = None
        self.drain()

    def drain(self, wait=False):
        while self.pending:
            row = self.pending[0]
            ends = [b for _, _, b in row["phases"].values() if b is not None]
            if wait:
                for event in ends:
                    event.synchronize()
            elif any(not event.query() for event in ends):
                break
            result = dict(row)
            result["phases"] = {
                name: dict(host_ms=host_ms, cuda_span_ms=a.elapsed_time(b) if a is not None else None)
                for name, (host_ms, a, b) in row["phases"].items()
            }
            self.handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.handle.flush()
            self.pending.pop(0)

    def close(self):
        self.drain(wait=True)
        self.handle.close()
