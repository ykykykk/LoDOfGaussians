"""Summarize sampled runtime metrics, not a predicted training speedup."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(path: Path, warmup: int = 100):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["iteration"] >= warmup]
    if not rows:
        raise ValueError("no profiling samples remain after warmup")
    phases = sorted({key for r in rows for key in r["phases"]})
    print(f"Samples: {len(rows)}; iterations {rows[0]['iteration']}..{rows[-1]['iteration']}")
    print("Phase                          median host ms    median CUDA span ms")
    for name in phases:
        values = [r["phases"][name] for r in rows if name in r["phases"]]
        host = statistics.median(v["host_ms"] for v in values)
        spans = [v["cuda_span_ms"] for v in values if v["cuda_span_ms"] is not None]
        gpu = f"{statistics.median(spans):.3f}" if spans else "n/a"
        print(f"{name:30s} {host:14.3f} {gpu:22s}")
    last = rows[-1]
    requested = last["requested_rows"]
    print(f"Cumulative cache/packet hit: {last['hit_rows'] / max(requested, 1):.2%}")
    print(f"Same-packet reuses: {last['same_packet_hits']}; overflow steps: {last['overflow_steps']}")
    print(f"Uploaded rows: {last['uploaded_rows']:,}; downloaded rows: {last['downloaded_rows']:,}")
    if len(rows) > 1:
        seconds = last["elapsed_wall_s"] - rows[0]["elapsed_wall_s"]
        steps = last["iteration"] - rows[0]["iteration"]
        print(f"Observed window wall time/step: {seconds * 1000 / max(steps, 1):.3f} ms (includes rebuild/save work)")
    print("CUDA spans may contain CPU launch gaps and stream waits; do not sum them as pure kernel time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    summarize(args.input, args.warmup)
