"""
spatial_pipeline/viz/renderer.py
----------------------------------
Frame compositor.  Draws all visual layers onto the latest raw frame
and returns an annotated copy.

Layer order (painter's algorithm, back to front):
  1. Sobel edge overlay      (optional, semi-transparent green)
  2. HSV color-mask overlay  (optional, semi-transparent cyan)
  3. Per-track trail         (fading polyline)
  4. Per-track velocity arrow
  5. Per-track bounding box + corner accents
  6. Per-track label pill    (class · confidence · speed)
  7. HUD panel               (thread FPS, latency, health dot)

All drawing uses OpenCV primitives (cv2.*).  This is correct here —
OpenCV's drawing functions are pure UI layer, not image-processing math.
The from-scratch constraint applies only to the processing stage.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np

from spatial_pipeline.tracking.tracker import TrackedObject
from spatial_pipeline.utils.config import RenderConfig


# ------------------------------------------------------------------ #
#  Colour palette                                                      #
# ------------------------------------------------------------------ #

_PALETTE: list[tuple[int,int,int]] = [   # BGR
    ( 50, 220, 100),  # spring green
    (235, 110,  20),  # vivid blue
    ( 20, 140, 235),  # vivid orange
    (170,  30, 220),  # purple
    (220, 220,  30),  # cyan
    ( 30, 200, 220),  # gold
    (140,  60, 240),  # hot pink
    ( 60, 240, 220),  # lime
]

def _track_color(track_id: int) -> tuple[int,int,int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ------------------------------------------------------------------ #
#  Renderer                                                            #
# ------------------------------------------------------------------ #

class FrameRenderer:
    """
    Stateless compositor.  Create once; call render() every frame.

    Parameters
    ----------
    cfg : RenderConfig
        All visual toggles and parameters.
    """

    def __init__(self, cfg: Optional[RenderConfig] = None) -> None:
        self.cfg = cfg or RenderConfig()

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def render(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        *,
        edges:      Optional[np.ndarray] = None,
        color_mask: Optional[np.ndarray] = None,
        hud_data:   Optional[Dict]       = None,
    ) -> np.ndarray:
        """
        Return a new annotated BGR frame.  Never mutates *frame* in-place.
        """
        out = frame.copy()

        if self.cfg.show_edges and edges is not None:
            out = self._blend_edges(out, edges)

        if color_mask is not None:
            out = self._blend_mask(out, color_mask)

        for track in tracks:
            color = _track_color(track.track_id)
            if self.cfg.show_trail:
                self._draw_trail(out, track, color)
            if self.cfg.show_velocity_arrow:
                self._draw_velocity_arrow(out, track, color)
            self._draw_box(out, track, color)
            self._draw_label(out, track, color)

        if self.cfg.show_hud and hud_data:
            self._draw_hud(out, hud_data)

        return out

    # ---------------------------------------------------------------- #
    #  Layer helpers                                                    #
    # ---------------------------------------------------------------- #

    def _blend_edges(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        overlay = np.zeros_like(frame)
        overlay[..., 1] = edges          # green channel
        return cv2.addWeighted(frame, 1 - self.cfg.edge_alpha,
                               overlay,  self.cfg.edge_alpha, 0)

    def _blend_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        overlay[mask > 0] = (200, 255, 200)
        return cv2.addWeighted(frame, 0.80, overlay, 0.20, 0)

    def _draw_trail(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        color: tuple,
    ) -> None:
        pts = list(track.trail)
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            fade = max(0.0, b.alpha)
            c    = tuple(int(ch * fade) for ch in color)
            t    = max(1, int(self.cfg.trail_thickness * fade))
            cv2.line(frame, (a.cx, a.cy), (b.cx, b.cy), c, t,
                     lineType=cv2.LINE_AA)

    def _draw_velocity_arrow(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        color: tuple,
    ) -> None:
        speed = track.speed_px_s
        if speed < 3.0:
            return
        dx, dy = track.velocity_dir
        cx, cy  = track.last_detection.cx, track.last_detection.cy
        length  = min(speed * self.cfg.arrow_scale * 30, 90)
        ex = int(cx + dx * length)
        ey = int(cy + dy * length)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color,
                        thickness=2, line_type=cv2.LINE_AA, tipLength=0.32)
        cv2.putText(frame, f"{speed:.0f}px/s", (ex + 5, ey - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _draw_box(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        color: tuple,
    ) -> None:
        det = track.last_detection
        t   = self.cfg.box_thickness
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, t,
                      lineType=cv2.LINE_AA)
        # Corner accents — reinforces the box without filling it
        ac = min(18, max(4, (det.x2 - det.x1) // 5))
        at = t + 1
        corners = [
            ((det.x1, det.y1), ( 1,  1)),
            ((det.x2, det.y1), (-1,  1)),
            ((det.x1, det.y2), ( 1, -1)),
            ((det.x2, det.y2), (-1, -1)),
        ]
        for (px, py), (sx, sy) in corners:
            cv2.line(frame, (px, py), (px + sx * ac, py), color, at)
            cv2.line(frame, (px, py), (px, py + sy * ac), color, at)

    def _draw_label(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        color: tuple,
    ) -> None:
        det  = track.last_detection
        text = (
            f"#{track.track_id} {det.class_name} "
            f"{det.confidence:.0%}  {track.speed_px_s:.0f}px/s"
        )
        fs, ft = self.cfg.font_scale, self.cfg.font_thickness
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        x, y = det.x1, det.y1 - 6
        # Pill background
        cv2.rectangle(frame,
                      (x - 2,      y - th - bl - 2),
                      (x + tw + 4, y + 2),
                      color, cv2.FILLED)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), ft, cv2.LINE_AA)

    # ---------------------------------------------------------------- #
    #  HUD                                                              #
    # ---------------------------------------------------------------- #

    def _draw_hud(self, frame: np.ndarray, data: Dict) -> None:
        h, w = frame.shape[:2]

        rows: list[tuple[str, str]] = []
        if "capture_fps"  in data: rows.append(("Capture",   f"{data['capture_fps']:5.1f} FPS"))
        if "process_fps"  in data: rows.append(("Process",   f"{data['process_fps']:5.1f} FPS"))
        if "infer_fps"    in data: rows.append(("Inference", f"{data['infer_fps']:5.1f} FPS"))
        if "render_fps"   in data: rows.append(("Display",   f"{data['render_fps']:5.1f} FPS"))
        if "infer_ms"     in data: rows.append(("Inf. lat",  f"{data['infer_ms']:5.1f} ms"))
        if "pipeline_ms"  in data: rows.append(("Pipeline",  f"{data['pipeline_ms']:5.1f} ms"))
        if "dropped"      in data: rows.append(("Dropped",   str(data["dropped"])))
        if "tracks"       in data: rows.append(("Tracks",    str(data["tracks"])))

        fs = 0.42; lh = 17; pad = 8
        panel_w = 192; panel_h = len(rows) * lh + pad * 2
        ox, oy  = w - panel_w - pad, pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (ox - 2, oy), (ox + panel_w, oy + panel_h),
                      (18, 18, 18), cv2.FILLED)
        cv2.addWeighted(overlay, self.cfg.hud_alpha, frame,
                        1 - self.cfg.hud_alpha, 0, frame)

        for i, (label, value) in enumerate(rows):
            y = oy + pad + i * lh + lh // 2
            cv2.putText(frame, f"{label:<9} {value}",
                        (ox + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (190, 225, 190), 1, cv2.LINE_AA)

        # Health dot: green ≥28 FPS / amber ≥18 / red below
        fps = data.get("render_fps", 0)
        dot = (0,220,80) if fps >= 28 else (0,200,220) if fps >= 18 else (30,60,230)
        cv2.circle(frame, (ox - 2 + 7, oy + 7), 5, dot, cv2.FILLED)
