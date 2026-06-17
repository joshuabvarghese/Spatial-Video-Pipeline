"""
tracker.py — Pixel Velocity Calculator + Trailing Path

Computes:
  - Per-object pixel velocity (px/s) using exponential moving average
  - Velocity direction vector (unit vector for arrow rendering)
  - Trail: a deque of past centroid positions, fading in opacity
  - Optional simple Kalman filter for centroid prediction under occlusion

All operations are pure Python / NumPy — no additional dependencies.
"""

import time
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from inference import Detection


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass
class TrailPoint:
    cx: int
    cy: int
    ts: float        # time.perf_counter()
    alpha: float     # opacity 0-1 (decays with age)


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    last_detection: Detection
    last_ts: float

    # Velocity in pixels/second (EMA smoothed)
    vx: float = 0.0
    vy: float = 0.0

    # Trail of past positions
    trail: deque = field(default_factory=lambda: deque(maxlen=60))

    # Kalman state: [cx, cy, vx, vy]
    kalman_state: Optional[np.ndarray] = None
    kalman_P: Optional[np.ndarray] = None

    @property
    def speed_px_per_s(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def velocity_direction(self) -> Tuple[float, float]:
        """Unit vector in velocity direction."""
        s = self.speed_px_per_s
        if s < 1e-3:
            return 0.0, 0.0
        return self.vx / s, self.vy / s


# ------------------------------------------------------------------ #
#  Simple IoU-based multi-object tracker                               #
# ------------------------------------------------------------------ #

class VelocityTracker:
    """
    Lightweight tracker that:
      1. Matches new detections to existing tracks via IoU.
      2. Updates pixel velocity with EMA smoothing.
      3. Maintains a per-track position trail.
      4. Optionally predicts position with a simple Kalman filter.

    Parameters
    ----------
    ema_alpha : float
        EMA weight for velocity update. Higher = faster response, noisier.
    trail_fade_s : float
        Trail points older than this (seconds) are removed.
    iou_threshold : float
        Minimum IoU to associate a detection with an existing track.
    max_lost_frames : int
        How many frames a track survives without a matching detection.
    use_kalman : bool
        Enable Kalman filter for centroid smoothing / prediction.
    """

    def __init__(self,
                 ema_alpha: float = 0.4,
                 trail_fade_s: float = 2.0,
                 iou_threshold: float = 0.3,
                 max_lost_frames: int = 10,
                 use_kalman: bool = True):
        self.ema_alpha = ema_alpha
        self.trail_fade_s = trail_fade_s
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.use_kalman = use_kalman

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 0
        self._lost_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Update                                                              #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[Detection], ts: float) -> List[TrackedObject]:
        """
        Match detections to tracks, update velocities and trails.
        Returns the current list of active tracked objects.
        """
        now = ts

        # --- Match detections → tracks via greedy IoU ---
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        if self._tracks and detections:
            track_ids = list(self._tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for ti, tid in enumerate(track_ids):
                t = self._tracks[tid]
                td = t.last_detection
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _box_iou(
                        td.x1, td.y1, td.x2, td.y2,
                        det.x1, det.y1, det.x2, det.y2
                    )

            # Greedy match: pick highest IoU pairs first
            while True:
                if iou_matrix.size == 0:
                    break
                best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                ti, di = best
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                tid = track_ids[ti]
                self._update_track(self._tracks[tid], detections[di], now)
                self._lost_counts[tid] = 0
                matched_track_ids.add(tid)
                if di in unmatched_dets:
                    unmatched_dets.remove(di)
                # Zero out row and column so they can't be reused
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # --- Age out unmatched tracks ---
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_ids:
                self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
                if self._lost_counts[tid] > self.max_lost_frames:
                    del self._tracks[tid]
                    del self._lost_counts[tid]
                else:
                    # Predict forward with Kalman if enabled
                    if self.use_kalman:
                        self._kalman_predict(self._tracks[tid])

        # --- Create new tracks for unmatched detections ---
        for di in unmatched_dets:
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            track = TrackedObject(
                track_id=tid,
                class_name=det.class_name,
                last_detection=det,
                last_ts=now,
            )
            if self.use_kalman:
                self._kalman_init(track)
            self._tracks[tid] = track
            self._lost_counts[tid] = 0

        # --- Prune old trail points ---
        for track in self._tracks.values():
            cutoff = now - self.trail_fade_s
            while track.trail and track.trail[0].ts < cutoff:
                track.trail.popleft()
            # Update alpha values based on age
            for pt in track.trail:
                age = now - pt.ts
                pt.alpha = max(0.0, 1.0 - age / self.trail_fade_s)

        return list(self._tracks.values())

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_track(self, track: TrackedObject, det: Detection, now: float):
        dt = now - track.last_ts
        if dt > 0.001:
            # Raw velocity in px/s
            raw_vx = (det.cx - track.last_detection.cx) / dt
            raw_vy = (det.cy - track.last_detection.cy) / dt

            # EMA smoothing
            a = self.ema_alpha
            track.vx = a * raw_vx + (1 - a) * track.vx
            track.vy = a * raw_vy + (1 - a) * track.vy

        # Append trail point
        track.trail.append(TrailPoint(
            cx=det.cx, cy=det.cy, ts=now, alpha=1.0
        ))

        if self.use_kalman:
            self._kalman_update(track, det.cx, det.cy)

        track.last_detection = det
        track.last_ts = now

    # ------------------------------------------------------------------ #
    #  Kalman filter (constant-velocity model)                            #
    # ------------------------------------------------------------------ #

    def _kalman_init(self, track: TrackedObject):
        track.kalman_state = np.array([
            track.last_detection.cx,
            track.last_detection.cy,
            0.0, 0.0
        ], dtype=np.float64)
        track.kalman_P = np.eye(4, dtype=np.float64) * 500.0

    def _kalman_predict(self, track: TrackedObject):
        if track.kalman_state is None:
            return
        # State transition: assume constant velocity
        F = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        Q = np.eye(4) * 10.0   # process noise
        track.kalman_state = F @ track.kalman_state
        track.kalman_P = F @ track.kalman_P @ F.T + Q

    def _kalman_update(self, track: TrackedObject, cx: int, cy: int):
        if track.kalman_state is None:
            return
        # Measurement matrix: we observe cx, cy only
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * 25.0   # measurement noise (px²)

        self._kalman_predict(track)
        z = np.array([cx, cy], dtype=np.float64)
        y = z - H @ track.kalman_state
        S = H @ track.kalman_P @ H.T + R
        K = track.kalman_P @ H.T @ np.linalg.inv(S)
        track.kalman_state += K @ y
        track.kalman_P = (np.eye(4) - K @ H) @ track.kalman_P


def _box_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0
