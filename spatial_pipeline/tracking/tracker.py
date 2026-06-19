"""
spatial_pipeline/tracking/tracker.py
--------------------------------------
Multi-object tracker with pixel-velocity estimation.

Design
------
Matching:   Greedy IoU — O(N·M) but N,M ≤ ~20 in practice; fast enough.
            A Hungarian solver would be strictly correct but adds a scipy
            dependency and ~0.3 ms for negligible real-world gain.

Velocity:   Exponential Moving Average on raw pixel displacement / Δt.
            EMA is causally stable and requires no history buffer.

Smoothing:  Optional constant-velocity Kalman filter (4-state: cx, cy,
            vx, vy).  Improves centroid estimates under partial occlusion
            and jitter from detection noise.

Trail:      Deque of TrailPoint — each point carries an alpha that decays
            linearly with age so the renderer can draw a fading polyline
            without iterating the whole history.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spatial_pipeline.inference.engine import Detection
from spatial_pipeline.utils.config import TrackerConfig


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class TrailPoint:
    cx: int
    cy: int
    ts: float      # perf_counter timestamp
    alpha: float   # 0.0 (invisible) – 1.0 (fully opaque)


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    last_detection: Detection
    last_ts: float

    vx: float = 0.0   # pixels / second  (EMA smoothed)
    vy: float = 0.0

    trail: deque = field(default_factory=lambda: deque(maxlen=60))

    # Kalman: state [cx, cy, vx, vy], covariance P (4×4)
    _kstate: Optional[np.ndarray] = field(default=None, repr=False)
    _kP:     Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def speed_px_s(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def velocity_dir(self) -> Tuple[float, float]:
        """Unit vector in velocity direction, or (0, 0) if nearly static."""
        s = self.speed_px_s
        return (self.vx / s, self.vy / s) if s > 1.0 else (0.0, 0.0)


# ------------------------------------------------------------------ #
#  Tracker                                                             #
# ------------------------------------------------------------------ #

class VelocityTracker:
    """
    IoU-based greedy multi-object tracker with EMA velocity and
    optional Kalman centroid smoothing.

    Thread-safety: not internally locked.  Call update() from a single
    thread (the render/orchestration thread).
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None) -> None:
        self._cfg = cfg or TrackerConfig()
        self._tracks:      Dict[int, TrackedObject] = {}
        self._lost_counts: Dict[int, int]           = {}
        self._next_id: int = 0

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def update(
        self,
        detections: List[Detection],
        ts: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Match *detections* to existing tracks, update velocities/trails,
        create new tracks for unmatched detections, prune lost tracks.

        Returns the list of currently active TrackedObjects.

        Parameters
        ----------
        detections : list of Detection
        ts : float  time.perf_counter() timestamp; uses now() if None.
        """
        now = ts if ts is not None else time.perf_counter()

        unmatched_det_indices = list(range(len(detections)))
        matched_track_ids:     set[int] = set()

        # ---- Match detections → tracks --------------------------------
        if self._tracks and detections:
            tids = list(self._tracks.keys())
            iou_mat = np.zeros((len(tids), len(detections)), dtype=np.float32)
            for ti, tid in enumerate(tids):
                prev = self._tracks[tid].last_detection
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = _box_iou(
                        prev.x1, prev.y1, prev.x2, prev.y2,
                        det.x1,  det.y1,  det.x2,  det.y2,
                    )
            # Greedy: take highest IoU pairs until threshold is unmet
            while iou_mat.size:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[ti, di] < self._cfg.iou_threshold:
                    break
                tid = tids[ti]
                self._update_track(self._tracks[tid], detections[di], now)
                matched_track_ids.add(tid)
                self._lost_counts[tid] = 0
                if di in unmatched_det_indices:
                    unmatched_det_indices.remove(di)
                iou_mat[ti, :] = 0.0
                iou_mat[:, di] = 0.0

        # ---- Age out unmatched tracks ---------------------------------
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_ids:
                self._lost_counts[tid] = self._lost_counts.get(tid, 0) + 1
                if self._lost_counts[tid] > self._cfg.max_lost_frames:
                    del self._tracks[tid]
                    del self._lost_counts[tid]
                elif self._cfg.use_kalman:
                    self._kalman_predict(self._tracks[tid])

        # ---- Create new tracks ----------------------------------------
        for di in unmatched_det_indices:
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            track = TrackedObject(
                track_id=tid,
                class_name=det.class_name,
                last_detection=det,
                last_ts=now,
                trail=deque(maxlen=self._cfg.trail_max_len),
            )
            if self._cfg.use_kalman:
                self._kalman_init(track)
            self._tracks[tid] = track
            self._lost_counts[tid] = 0

        # ---- Prune stale trail points ---------------------------------
        cutoff = now - self._cfg.trail_fade_s
        for track in self._tracks.values():
            while track.trail and track.trail[0].ts < cutoff:
                track.trail.popleft()
            for pt in track.trail:
                age = now - pt.ts
                pt.alpha = max(0.0, 1.0 - age / self._cfg.trail_fade_s)

        return list(self._tracks.values())

    def reset(self) -> None:
        """Clear all tracks (e.g. on scene cut)."""
        self._tracks.clear()
        self._lost_counts.clear()
        self._next_id = 0

    # ---------------------------------------------------------------- #
    #  Track update helpers                                             #
    # ---------------------------------------------------------------- #

    def _update_track(
        self,
        track: TrackedObject,
        det: Detection,
        now: float,
    ) -> None:
        dt = now - track.last_ts
        if dt > 1e-3:
            raw_vx = (det.cx - track.last_detection.cx) / dt
            raw_vy = (det.cy - track.last_detection.cy) / dt
            a = self._cfg.ema_alpha
            track.vx = a * raw_vx + (1.0 - a) * track.vx
            track.vy = a * raw_vy + (1.0 - a) * track.vy

        track.trail.append(TrailPoint(cx=det.cx, cy=det.cy, ts=now, alpha=1.0))

        if self._cfg.use_kalman:
            self._kalman_update(track, det.cx, det.cy)

        track.last_detection = det
        track.last_ts        = now

    # ---------------------------------------------------------------- #
    #  Kalman filter — constant-velocity model                         #
    # ---------------------------------------------------------------- #
    #
    #  State:  x = [cx, cy, vx, vy]ᵀ
    #  Transition (1-step, dt=1):  F = [[1 0 1 0]
    #                                   [0 1 0 1]
    #                                   [0 0 1 0]
    #                                   [0 0 0 1]]
    #  Observation:  z = [cx, cy]ᵀ,  H = [[1 0 0 0]
    #                                      [0 1 0 0]]

    _F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float64)
    _H = np.array([[1,0,0,0],[0,1,0,0]],                     np.float64)
    _Q = np.eye(4, dtype=np.float64) *  8.0   # process noise
    _R = np.eye(2, dtype=np.float64) * 20.0   # measurement noise

    def _kalman_init(self, track: TrackedObject) -> None:
        track._kstate = np.array(
            [track.last_detection.cx, track.last_detection.cy, 0.0, 0.0],
            dtype=np.float64,
        )
        track._kP = np.eye(4, dtype=np.float64) * 400.0

    def _kalman_predict(self, track: TrackedObject) -> None:
        if track._kstate is None:
            return
        track._kstate = self._F @ track._kstate
        track._kP     = self._F @ track._kP @ self._F.T + self._Q

    def _kalman_update(self, track: TrackedObject, cx: int, cy: int) -> None:
        if track._kstate is None:
            return
        self._kalman_predict(track)
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ track._kstate
        S = self._H @ track._kP @ self._H.T + self._R
        K = track._kP @ self._H.T @ np.linalg.inv(S)
        track._kstate = track._kstate + K @ y
        track._kP     = (np.eye(4) - K @ self._H) @ track._kP


# ------------------------------------------------------------------ #
#  Geometry helpers                                                    #
# ------------------------------------------------------------------ #

def _box_iou(ax1,ay1,ax2,ay2, bx1,by1,bx2,by2) -> float:
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0
