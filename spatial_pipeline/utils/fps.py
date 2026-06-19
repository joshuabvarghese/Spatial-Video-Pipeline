"""
spatial_pipeline/utils/fps.py
-------------------------------
Rolling-window FPS tracker. Thread-safe, allocation-free after warmup.
"""

from __future__ import annotations

import time
import threading
from collections import deque


class FPSCounter:
    """
    Tracks frames-per-second over a rolling window of timestamps.

    Parameters
    ----------
    window : int
        Number of frame timestamps to keep. Larger = smoother but laggier.
    """

    __slots__ = ("_window", "_ts", "_lock")

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._ts: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Call once per frame."""
        with self._lock:
            self._ts.append(time.perf_counter())

    @property
    def fps(self) -> float:
        with self._lock:
            ts = self._ts
            if len(ts) < 2:
                return 0.0
            return (len(ts) - 1) / (ts[-1] - ts[0])

    def reset(self) -> None:
        with self._lock:
            self._ts.clear()
