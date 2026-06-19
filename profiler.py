"""
profiler.py — Per-Stage Pipeline Profiler

Collects timing samples for each pipeline stage and can dump them to CSV.
Uses a ring buffer to avoid unbounded memory growth during long runs.
"""

import csv
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class StageSample:
    stage: str
    frame_id: int
    wall_time: float    # absolute timestamp
    duration_ms: float  # time spent in this stage


class PipelineProfiler:
    """
    Thread-safe profiler. Each pipeline stage calls .record() after completing.
    The main thread calls .summary() for live stats or .export_csv() to dump.

    Parameters
    ----------
    max_samples : int
        Ring buffer size per stage.
    """

    def __init__(self, max_samples: int = 500):
        self._samples: Dict[str, deque] = {}
        self._max = max_samples
        self._lock = threading.Lock()
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def record(self, stage: str, frame_id: int, duration_ms: float):
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

    def summary(self) -> Dict[str, Dict]:
        """Return mean/p50/p95/p99 latency per stage."""
        result = {}
        with self._lock:
            for stage, buf in self._samples.items():
                if not buf:
                    continue
                vals = sorted(s.duration_ms for s in buf)
                n = len(vals)
                result[stage] = {
                    'count': n,
                    'mean_ms': sum(vals) / n,
                    'p50_ms': vals[int(n * 0.50)],
                    'p95_ms': vals[int(n * 0.95)],
                    'p99_ms': vals[min(int(n * 0.99), n - 1)],
                    'max_ms': vals[-1],
                }
        return result

    def export_csv(self, path: str = "pipeline_profile.csv"):
        """Dump all samples to a CSV file."""
        with self._lock:
            all_samples = [
                s for buf in self._samples.values() for s in buf
            ]
        all_samples.sort(key=lambda s: s.wall_time)

        out = Path(path)
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['stage', 'frame_id', 'wall_time', 'duration_ms'])
            writer.writeheader()
            for s in all_samples:
                writer.writerow({
                    'stage': s.stage,
                    'frame_id': s.frame_id,
                    'wall_time': f"{s.wall_time:.6f}",
                    'duration_ms': f"{s.duration_ms:.3f}",
                })
        print(f"[Profiler] Exported {len(all_samples)} samples → {out.resolve()}")

    def print_summary(self):
        summary = self.summary()
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              PIPELINE PROFILER SUMMARY                      ║")
        print("╠══════════╦════════╦════════╦════════╦════════╦══════════════╣")
        print("║ Stage    ║  Count ║  Mean  ║   P50  ║   P95  ║    Max       ║")
        print("╠══════════╬════════╬════════╬════════╬════════╬══════════════╣")
        for stage, s in summary.items():
            print(f"║ {stage:<8} ║ {s['count']:>6} ║"
                  f" {s['mean_ms']:>5.1f}ms ║"
                  f" {s['p50_ms']:>5.1f}ms ║"
                  f" {s['p95_ms']:>5.1f}ms ║"
                  f" {s['max_ms']:>8.1f}ms ║")
        print("╚══════════╩════════╩════════╩════════╩════════╩══════════════╝\n")
