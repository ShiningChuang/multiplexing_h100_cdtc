"""Common utilities: GPU timing, roofline analysis, CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List

import torch


def gpu_timer(fn: Callable[[], object], warmup: int = 20, repeat: int = 200) -> Dict:
    """Time a no-arg callable with CUDA events.

    Returns timing percentiles plus the raw per-iteration list for archival.
    """
    # Warmup (not measured).
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    sorted_t = sorted(times_ms)
    n = len(sorted_t)

    def pct(p: float) -> float:
        # nearest-rank percentile
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return sorted_t[idx]

    return {
        "min_ms":    sorted_t[0],
        "max_ms":    sorted_t[-1],
        "median_ms": median(sorted_t),
        "p10_ms":    pct(0.10),
        "p90_ms":    pct(0.90),
        "mean_ms":   sum(sorted_t) / n,
        "times_ms":  times_ms,
    }


def roofline(
    flops: float,
    bytes_accessed: float,
    latency_ms: float,
    peak_tflops: float,
    peak_bw_gbs: float,
) -> Dict:
    """Standard roofline classification + utilization."""
    seconds = latency_ms * 1e-3
    achieved_tflops = flops / seconds / 1e12
    achieved_bw_gbs = bytes_accessed / seconds / 1e9
    intensity = flops / bytes_accessed  # FLOP / byte

    ridge = peak_tflops * 1e3 / peak_bw_gbs   # FLOP / byte
    bound = "compute" if intensity >= ridge else "memory"

    if bound == "compute":
        util_pct = 100.0 * achieved_tflops / peak_tflops
    else:
        util_pct = 100.0 * achieved_bw_gbs / peak_bw_gbs

    return {
        "tflops":    achieved_tflops,
        "bw_gbs":    achieved_bw_gbs,
        "intensity": intensity,
        "bound":     bound,
        "util_pct":  util_pct,
    }


def flush_csv(rows: List[Dict], path: Path, columns: List[str]) -> None:
    """Write rows to a CSV with a fixed column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
