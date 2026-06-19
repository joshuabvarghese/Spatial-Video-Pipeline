"""
spatial_pipeline/utils/profiler.py
------------------------------------
Thread-safe per-stage pipeline profiler with ring-buffer sample storage.
Provides mean/P50/P95/P99/max latency and CSV export.
"""

from __future__ import annotations

import csv
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(slots=True)
class StageSample:
    stage: str
    frame_id: int
    wall_time: float
    duration_ms: float


class PipelineProfiler:
    """
    Records per-stage timing samples in per-stage ring buffers.
    Thread-safe: any thread may call .record() concurrently.
    """

    def __init__(self, max_samples: int = 500, enabled: bool = True):
        self._samples: Dict[str, deque] = {}
        self._max = max_samples
        self._enabled = enabled
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Recording                                                           #
    # ------------------------------------------------------------------ #

    def record(self, stage: str, frame_id: int, duration_ms: float) -> None:
        if not self._enabled:
            return
        sample = StageSample(
            stage=stage,
            frame_id=frame_id,
            wall_time=time.perf_counter(),
            duration_ms=duration_ms,
        )
        with self._lock:
            if stage not in self._samples:
                self._samples[stage] = deque(maxlen=self._max)
            self._samples[stage].append(sample)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        with self._lock:
            snapshot = {k: list(v) for k, v in self._samples.items()}
        for stage, samples in snapshot.items():
            if not samples:
                continue
            vals = sorted(s.duration_ms for s in samples)
            n = len(vals)
            result[stage] = {
                "count":   n,
                "mean_ms": sum(vals) / n,
                "p50_ms":  vals[n // 2],
                "p95_ms":  vals[int(n * 0.95)],
                "p99_ms":  vals[min(int(n * 0.99), n - 1)],
                "max_ms":  vals[-1],
            }
        return result

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            print("[Profiler] No samples recorded.")
            return
        _W = 10
        header = f"{'Stage':<{_W}}  {'N':>6}  {'Mean':>8}  {'P50':>8}  {'P95':>8}  {'P99':>8}  {'Max':>8}"
        sep = "─" * len(header)
        print(f"\n{sep}")
        print("  PIPELINE PROFILER SUMMARY")
        print(sep)
        print(header)
        print(sep)
        for stage, v in s.items():
            print(
                f"{stage:<{_W}}  {v['count']:>6}  "
                f"{v['mean_ms']:>7.2f}ms  {v['p50_ms']:>7.2f}ms  "
                f"{v['p95_ms']:>7.2f}ms  {v['p99_ms']:>7.2f}ms  "
                f"{v['max_ms']:>7.2f}ms"
            )
        print(f"{sep}\n")

    def export_csv(self, path: str = "pipeline_profile.csv") -> None:
        with self._lock:
            all_samples = [s for buf in self._samples.values() for s in buf]
        all_samples.sort(key=lambda s: s.wall_time)
        out = Path(path)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["stage", "frame_id", "wall_time", "duration_ms"]
            )
            writer.writeheader()
            for s in all_samples:
                writer.writerow({
                    "stage": s.stage,
                    "frame_id": s.frame_id,
                    "wall_time": f"{s.wall_time:.6f}",
                    "duration_ms": f"{s.duration_ms:.3f}",
                })
        print(f"[Profiler] {len(all_samples)} samples → {out.resolve()}")
