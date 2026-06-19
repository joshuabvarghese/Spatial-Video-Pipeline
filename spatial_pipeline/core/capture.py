"""
spatial_pipeline/core/capture.py
----------------------------------
CameraCapture thread and FrameRingBuffer.

Design rationale
----------------
The ring buffer is the heart of the zero-lag guarantee.  We use a
3-slot "latest-write-wins" buffer rather than a queue because:

  1. Queues accumulate lag: if inference runs at 15 FPS and capture at 30,
     a queue grows 2× per second, producing stale detections.
  2. We only ever care about the *most recent* frame — older frames are
     waste.  A 3-slot buffer ensures the writer always has a free slot
     (slot_being_written ≠ slot_being_read ≠ slot_just_written), so the
     writer never blocks on a reader holding the only slot.

The only lock is a single pointer swap (latest_idx), held for
microseconds — not across the memcpy of a full frame.

Thread safety model
-------------------
  - Single producer (CameraCapture.run) calls put().
  - Multiple consumers may call get_latest() concurrently; each gets
    the same latest packet (read-many).  The "consumed" flag is per-
    consumer via the returned Optional — if None, the frame was already
    seen.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from spatial_pipeline.utils.fps import FPSCounter
from spatial_pipeline.utils.config import CaptureConfig


# ------------------------------------------------------------------ #
#  Frame packet                                                        #
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class FramePacket:
    """Immutable snapshot of one captured frame plus metadata."""
    frame: np.ndarray       # BGR uint8, shape (H, W, 3)
    frame_id: int           # monotonically increasing, unique per session
    capture_ts: float       # time.perf_counter() at capture
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        self.height, self.width = self.frame.shape[:2]


# ------------------------------------------------------------------ #
#  Ring buffer                                                         #
# ------------------------------------------------------------------ #

class FrameRingBuffer:
    """
    3-slot lock-minimised ring buffer optimised for single-producer /
    multi-consumer access patterns.

    The slot pointer swap is the only critical section — it is held for
    ~100 ns on modern hardware.
    """

    SLOTS: int = 3

    def __init__(self) -> None:
        self._slots: list[Optional[FramePacket]] = [None] * self.SLOTS
        self._write_idx: int = 0
        self._latest_idx: int = -1
        self._lock = threading.Lock()
        self._dropped: int = 0

    # -- Producer -- #

    def put(self, packet: FramePacket) -> None:
        slot = self._write_idx % self.SLOTS
        self._slots[slot] = packet
        with self._lock:
            if self._latest_idx != -1:
                self._dropped += 1          # downstream too slow
            self._latest_idx = slot
        self._write_idx += 1

    # -- Consumer -- #

    def get_latest(self) -> Optional[FramePacket]:
        """Non-blocking. Returns the newest unread packet, or None."""
        with self._lock:
            idx = self._latest_idx
            self._latest_idx = -1
        if idx < 0:
            return None
        return self._slots[idx]

    def peek_latest(self) -> Optional[FramePacket]:
        """Non-blocking. Returns newest packet without consuming it."""
        with self._lock:
            idx = self._latest_idx
        if idx < 0:
            return None
        return self._slots[idx]

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    def reset_dropped(self) -> None:
        with self._lock:
            self._dropped = 0


# ------------------------------------------------------------------ #
#  Camera capture thread                                               #
# ------------------------------------------------------------------ #

class CameraCapture(threading.Thread):
    """
    Dedicated thread that reads frames from a camera or video file and
    pushes them into a FrameRingBuffer at the source's native rate.

    Isolation guarantees
    --------------------
    This thread's only job is I/O: reading from the kernel's camera
    buffer as fast as possible and placing frames into the ring buffer.
    It does zero image processing.  Any hiccup in downstream threads
    (inference, rendering) never causes frame drops here.

    Parameters
    ----------
    cfg : CaptureConfig
        Source, FPS cap, and buffer depth configuration.
    """

    def __init__(self, cfg: Optional[CaptureConfig] = None) -> None:
        super().__init__(name="CameraCapture", daemon=True)
        self._cfg = cfg or CaptureConfig()
        self.buffer = FrameRingBuffer()
        self._stop_event = threading.Event()
        self._frame_id: int = 0
        self._fps = FPSCounter(window=60)
        self._cap: Optional[cv2.VideoCapture] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def fps(self) -> float:
        return self._fps.fps

    @property
    def is_healthy(self) -> bool:
        return self._error is None and self.is_alive()

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def resolution(self) -> tuple[int, int]:
        """Returns (width, height) or (0, 0) if not open."""
        if self._cap and self._cap.isOpened():
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return 0, 0

    # ------------------------------------------------------------------ #
    #  Thread body                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            src = self._cfg.source
            try:
                src = int(src)
            except (ValueError, TypeError):
                pass

            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open source: {src!r}")

            # Minimise kernel-side buffer for live cameras to reduce
            # capture-to-display latency.
            if isinstance(src, int):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            min_interval = 1.0 / self._cfg.target_fps
            last_ts = 0.0

            while not self._stop_event.is_set():
                now = time.perf_counter()
                remaining = min_interval - (now - last_ts)
                if remaining > 0:
                    time.sleep(remaining)
                    continue

                ret, frame = self._cap.read()
                ts = time.perf_counter()

                if not ret:
                    if isinstance(src, str):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break  # camera disconnected

                self.buffer.put(FramePacket(
                    frame=frame,
                    frame_id=self._frame_id,
                    capture_ts=ts,
                ))
                self._fps.tick()
                self._frame_id += 1
                last_ts = ts

        except Exception as exc:
            self._error = exc
        finally:
            if self._cap:
                self._cap.release()
