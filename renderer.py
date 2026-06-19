"""
renderer.py — Render Compositor

Draws on top of the raw frame:
  - Bounding boxes with class label + confidence
  - Fading trailing path (polyline, alpha-blended)
  - Velocity arrow (direction + magnitude)
  - Sobel edge overlay (optional)
  - HUD panel: per-thread FPS, pipeline latency, inference time
  - Color-coded thread health indicators

All drawing is done with OpenCV primitives (this is UI, not processing,
so using the wrappers here is correct and expected).
"""

import cv2
import math
import numpy as np
import time
from typing import List, Optional, Dict
from dataclasses import dataclass

from tracker import TrackedObject, TrailPoint


# Distinct colours per track ID (BGR)
_TRACK_COLORS = [
    (0, 255, 127),   # spring green
    (255, 128, 0),   # deep sky blue
    (0, 128, 255),   # orange
    (147, 20, 255),  # deep pink
    (255, 255, 0),   # cyan
    (0, 215, 255),   # gold
    (180, 105, 255), # hot pink
    (255, 0, 144),   # violet
]


def track_color(track_id: int):
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


@dataclass
class RenderConfig:
    show_edges: bool = False
    edge_alpha: float = 0.35
    show_trail: bool = True
    trail_thickness: int = 2
    show_velocity_arrow: bool = True
    arrow_scale: float = 0.05      # seconds of future motion to project
    show_hud: bool = True
    hud_alpha: float = 0.7
    box_thickness: int = 2
    font_scale: float = 0.55
    font_thickness: int = 1


class FrameRenderer:
    """
    Composites all visual layers onto a copy of the raw frame.
    Stateless — called once per display frame with current tracked objects.
    """

    def __init__(self, cfg: Optional[RenderConfig] = None):
        self.cfg = cfg or RenderConfig()

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #

    def render(self,
               frame: np.ndarray,
               tracks: List[TrackedObject],
               edges: Optional[np.ndarray] = None,
               color_mask: Optional[np.ndarray] = None,
               hud_data: Optional[Dict] = None) -> np.ndarray:
        """
        Returns a new BGR frame with all overlays composited.
        Never modifies the input frame in-place.
        """
        out = frame.copy()

        # ---- Sobel edge overlay ----
        if self.cfg.show_edges and edges is not None:
            out = self._blend_edges(out, edges)

        # ---- Color mask overlay ----
        if color_mask is not None:
            out = self._blend_mask(out, color_mask, color=(0, 255, 200), alpha=0.25)

        # ---- Per-track overlays ----
        for track in tracks:
            color = track_color(track.track_id)
            det = track.last_detection

            if self.cfg.show_trail:
                self._draw_trail(out, track.trail, color)

            if self.cfg.show_velocity_arrow:
                self._draw_velocity_arrow(out, track, color)

            self._draw_box(out, det, color)
            self._draw_label(out, track, color)

        # ---- HUD ----
        if self.cfg.show_hud and hud_data:
            self._draw_hud(out, hud_data)

        return out

    # ------------------------------------------------------------------ #
    #  Drawing helpers                                                     #
    # ------------------------------------------------------------------ #

    def _blend_edges(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Alpha-blend green Sobel edges onto the frame."""
        edge_bgr = np.zeros_like(frame)
        edge_bgr[..., 1] = edges   # green channel
        alpha = self.cfg.edge_alpha
        return cv2.addWeighted(frame, 1 - alpha, edge_bgr, alpha, 0)

    def _blend_mask(self, frame: np.ndarray, mask: np.ndarray,
                    color: tuple, alpha: float) -> np.ndarray:
        overlay = frame.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    def _draw_trail(self, frame: np.ndarray, trail, color: tuple):
        pts = list(trail)
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            a = pts[i - 1]
            b = pts[i]
            # Fade opacity: blend toward black
            fade = b.alpha
            c = tuple(int(ch * fade) for ch in color)
            thickness = max(1, int(self.cfg.trail_thickness * fade))
            cv2.line(frame, (a.cx, a.cy), (b.cx, b.cy), c, thickness,
                     lineType=cv2.LINE_AA)

    def _draw_velocity_arrow(self, frame: np.ndarray, track: TrackedObject, color: tuple):
        speed = track.speed_px_per_s
        if speed < 2.0:
            return
        dx, dy = track.velocity_direction
        cx, cy = track.last_detection.cx, track.last_detection.cy
        # Project arrow tip: scale * speed pixels ahead
        length = min(speed * self.cfg.arrow_scale * 30, 80)
        ex = int(cx + dx * length)
        ey = int(cy + dy * length)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color,
                        thickness=2, line_type=cv2.LINE_AA,
                        tipLength=0.3)

        # Speed label
        label = f"{speed:.0f} px/s"
        cv2.putText(frame, label, (ex + 4, ey - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                    self.cfg.font_thickness, cv2.LINE_AA)

    def _draw_box(self, frame: np.ndarray, det, color: tuple):
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2),
                      color, self.cfg.box_thickness, lineType=cv2.LINE_AA)
        # Corner accents (looks more professional)
        accent_len = min(20, (det.x2 - det.x1) // 4)
        t = self.cfg.box_thickness + 1
        for (px, py), (dx, dy) in [
            ((det.x1, det.y1), (1, 1)),
            ((det.x2, det.y1), (-1, 1)),
            ((det.x1, det.y2), (1, -1)),
            ((det.x2, det.y2), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx * accent_len, py), color, t)
            cv2.line(frame, (px, py), (px, py + dy * accent_len), color, t)

    def _draw_label(self, frame: np.ndarray, track: TrackedObject, color: tuple):
        det = track.last_detection
        text = f"#{track.track_id} {det.class_name} {det.confidence:.0%}"
        fs = self.cfg.font_scale
        ft = self.cfg.font_thickness
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)

        x, y = det.x1, det.y1 - 6
        # Background pill
        cv2.rectangle(frame,
                      (x - 2, y - th - baseline - 2),
                      (x + tw + 4, y + 2),
                      color, cv2.FILLED)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), ft, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, data: Dict):
        h, w = frame.shape[:2]
        lines = []

        if 'capture_fps' in data:
            lines.append(f"Capture  : {data['capture_fps']:5.1f} FPS")
        if 'process_fps' in data:
            lines.append(f"Process  : {data['process_fps']:5.1f} FPS")
        if 'infer_fps' in data:
            lines.append(f"Inference: {data['infer_fps']:5.1f} FPS")
        if 'render_fps' in data:
            lines.append(f"Display  : {data['render_fps']:5.1f} FPS")
        if 'infer_ms' in data:
            lines.append(f"Inf. lat : {data['infer_ms']:5.1f} ms")
        if 'pipeline_ms' in data:
            lines.append(f"Pipeline : {data['pipeline_ms']:5.1f} ms")
        if 'dropped' in data:
            lines.append(f"Dropped  : {data['dropped']}")
        if 'tracks' in data:
            lines.append(f"Tracks   : {data['tracks']}")

        fs = 0.45
        ft = 1
        line_h = 18
        pad = 8
        panel_w = 180
        panel_h = len(lines) * line_h + pad * 2

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - pad, pad),
                      (w - pad, pad + panel_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, self.cfg.hud_alpha, frame,
                        1 - self.cfg.hud_alpha, 0, frame)

        for i, line in enumerate(lines):
            y = pad * 2 + i * line_h + line_h // 2
            x = w - panel_w
            cv2.putText(frame, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (200, 230, 200), ft, cv2.LINE_AA)

        # FPS health indicator dot
        fps = data.get('render_fps', 0)
        dot_color = (0, 255, 80) if fps >= 28 else (0, 200, 255) if fps >= 18 else (0, 80, 255)
        cv2.circle(frame, (w - panel_w - pad + 8, pad + 8), 5, dot_color, cv2.FILLED)
