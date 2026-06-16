"""
capture.py — CameraCapture Thread + FrameRingBuffer

Decouples camera I/O from the rest of the pipeline using a 3-slot
lock-minimised ring buffer. The capture thread runs at the camera's
native frame rate; downstream threads consume at their own pace without
ever stalling the camera.
"""

import cv2
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FramePacket:
    """Everything downstream threads need about a captured frame."""
    frame: np.ndarray          # BGR uint8
    frame_id: int              # monotonically increasing
    capture_ts: float          # time.perf_counter() at capture
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.height, self.width = self.frame.shape[:2]


class FrameRingBuffer:
    """
    3-slot ring buffer optimised for single-producer / single-consumer.

    The writer always writes to slot (write_idx % 3).
    The reader always reads from the slot that was most recently completed.
    A single atomic integer (latest_idx) acts as the handoff — no queue,
    no blocking, no frame accumulation lag.
    """

    SLOTS = 3

    def __init__(self):
        self._slots: list[Optional[FramePacket]] = [None] * self.SLOTS
        self._write_idx = 0
        self._latest_idx = -1          # -1 → nothing ready yet
        self._lock = threading.Lock()  # only guards the pointer swap
        self._dropped = 0

    # ------------------------------------------------------------------ #
    #  Producer side                                                       #
    # ------------------------------------------------------------------ #

    def put(self, packet: FramePacket) -> None:
        slot = self._write_idx % self.SLOTS
        self._slots[slot] = packet
        with self._lock:
            if self._latest_idx != -1:
                self._dropped += 1          # previous frame was never read
            self._latest_idx = slot
        self._write_idx += 1

    # ------------------------------------------------------------------ #
    #  Consumer side                                                       #
    # ------------------------------------------------------------------ #

    def get_latest(self) -> Optional[FramePacket]:
        """Non-blocking. Returns the most recent frame or None."""
        with self._lock:
            idx = self._latest_idx
            self._latest_idx = -1       # mark as consumed
        if idx == -1:
            return None
        return self._slots[idx]

    @property
    def dropped_frames(self) -> int:
        return self._dropped


# ------------------------------------------------------------------ #
#  Camera Capture Thread                                               #
# ------------------------------------------------------------------ #

class CameraCapture(threading.Thread):
    """
    Dedicated thread that reads frames from a camera or video file and
    pushes them into a FrameRingBuffer at the source's native rate.

    Parameters
    ----------
    source : int | str
        Camera index (0, 1, …) or path to a video file.
    target_fps : int
        Cap the capture rate to this FPS (useful for files).
    """

    def __init__(self, source=0, target_fps: int = 60):
        super().__init__(name="CameraCapture", daemon=True)
        self.source = source
        self.target_fps = target_fps
        self.buffer = FrameRingBuffer()

        self._stop_event = threading.Event()
        self._frame_id = 0
        self._fps_tracker = _FPSTracker(window=30)
        self._cap: Optional[cv2.VideoCapture] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def stop(self):
        self._stop_event.set()

    @property
    def fps(self) -> float:
        return self._fps_tracker.fps

    @property
    def is_healthy(self) -> bool:
        return self._error is None and self.is_alive()

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    # ------------------------------------------------------------------ #
    #  Thread body                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open source: {self.source!r}")

            # Prefer lower latency buffers when reading a live camera
            if isinstance(self.source, int):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            min_interval = 1.0 / self.target_fps
            last_capture = 0.0

            while not self._stop_event.is_set():
                now = time.perf_counter()
                elapsed = now - last_capture
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                    continue

                ret, frame = self._cap.read()
                ts = time.perf_counter()

                if not ret:
                    if isinstance(self.source, str):
                        # Loop video files
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                packet = FramePacket(
                    frame=frame,
                    frame_id=self._frame_id,
                    capture_ts=ts,
                )
                self.buffer.put(packet)
                self._fps_tracker.tick()
                self._frame_id += 1
                last_capture = ts

        except Exception as exc:
            self._error = exc
        finally:
            if self._cap:
                self._cap.release()


class _FPSTracker:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._timestamps: list[float] = []
        self._window = window
        self._lock = threading.Lock()

    def tick(self):
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self._window:
                self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        with self._lock:
            ts = self._timestamps
            if len(ts) < 2:
                return 0.0
            return (len(ts) - 1) / (ts[-1] - ts[0])
