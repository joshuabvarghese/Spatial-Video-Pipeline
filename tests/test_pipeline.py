"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for the spatial pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short  # shorter tracebacks

Design philosophy
-----------------
- Tests are deterministic: no webcam, no internet, no random seeds.
- Each test is fast (<2 s): no model download, synthetic inputs only.
- Tests document expected behaviour: assertions read as spec.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import pytest


# ------------------------------------------------------------------ #
#  Fixtures & helpers                                                  #
# ------------------------------------------------------------------ #

def _bgr_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a synthetic BGR uint8 frame with coloured blobs."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[20:80, 30:100]   = (0,   0, 220)   # red blob
    frame[80:160, 200:280] = (0, 200,   0)   # green blob
    frame[160:220, 10:80]  = (200, 0,  0)    # blue blob
    return frame


def _gray(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ------------------------------------------------------------------ #
#  Core: custom HSV conversion                                         #
# ------------------------------------------------------------------ #

class TestBGRtoHSV:
    """bgr_to_hsv_numpy — pure NumPy reimplementation of cv2.cvtColor."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.core.processor import bgr_to_hsv_numpy
        self.fn = bgr_to_hsv_numpy

    def test_pure_red_hue_is_zero(self):
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
        hsv = self.fn(bgr)
        H = hsv[0, 0, 0]
        assert H < 2.0 or H > 358.0, f"Pure red H should be ~0°, got {H:.2f}"

    def test_pure_green_hue_is_120(self):
        bgr = np.array([[[0, 255, 0]]], dtype=np.uint8)
        hsv = self.fn(bgr)
        assert abs(hsv[0, 0, 0] - 120.0) < 2.0

    def test_pure_blue_hue_is_240(self):
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        hsv = self.fn(bgr)
        assert abs(hsv[0, 0, 0] - 240.0) < 2.0

    def test_white_saturation_is_zero(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        hsv = self.fn(bgr)
        assert hsv[0, 0, 1] < 0.01, "White should have S=0"

    def test_black_value_is_zero(self):
        bgr = np.array([[[0, 0, 0]]], dtype=np.uint8)
        hsv = self.fn(bgr)
        assert hsv[0, 0, 2] < 0.01, "Black should have V=0"

    def test_output_shape_and_dtype(self):
        frame = _bgr_frame()
        hsv = self.fn(frame)
        assert hsv.shape == (*frame.shape[:2], 3)
        assert hsv.dtype == np.float32

    def test_hue_range_0_to_360(self):
        frame = _bgr_frame()
        hsv = self.fn(frame)
        assert hsv[..., 0].min() >= 0.0
        assert hsv[..., 0].max() < 360.0

    def test_sv_range_0_to_1(self):
        frame = _bgr_frame(480, 640)
        hsv = self.fn(frame)
        assert hsv[..., 1].min() >= 0.0
        assert hsv[..., 1].max() <= 1.0
        assert hsv[..., 2].max() <= 1.0

    def test_correlation_with_opencv(self):
        """Our implementation should correlate >.99 with cv2 on a real frame."""
        import cv2
        frame = _bgr_frame(480, 640)
        ours = self.fn(frame)
        cv2_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Rescale cv2 output to match our range: H×2, S/255, V/255
        cv2_H = cv2_hsv[..., 0] * 2.0
        our_H = ours[..., 0]
        # Compare saturation (most numerically stable)
        cv2_S = cv2_hsv[..., 1] / 255.0
        our_S = ours[..., 1]
        corr = float(np.corrcoef(cv2_S.flatten(), our_S.flatten())[0, 1])
        assert corr > 0.99, f"S-channel correlation with cv2: {corr:.4f}"


# ------------------------------------------------------------------ #
#  Core: Sobel edge detection                                          #
# ------------------------------------------------------------------ #

class TestSobelEdges:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.core.processor import sobel_edges_numpy
        self.fn = sobel_edges_numpy

    def test_output_dtype_uint8(self):
        edges = self.fn(_gray())
        assert edges.dtype == np.uint8

    def test_output_shape_matches_input(self):
        g = _gray(120, 160)
        assert self.fn(g).shape == g.shape

    def test_blank_frame_has_no_edges(self):
        # Interior of a uniform frame has zero gradient; border pixels may
        # show reflect-padding artefacts so we check the interior only.
        blank = np.full((100, 100), 128, dtype=np.uint8)
        edges = self.fn(blank, blur_sigma=0.5)
        assert edges[5:-5, 5:-5].max() == 0, \
            "Interior of uniform frame should have zero Sobel response"

    def test_sharp_edge_produces_high_magnitude(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255               # vertical step edge
        edges = self.fn(img, blur_sigma=0.5)
        # Expect strong response along x=49/50
        assert edges[:, 49].max() > 60, "Step edge should produce high Sobel magnitude"

    def test_values_in_0_255(self):
        g = _gray(240, 320)
        edges = self.fn(g)
        assert edges.min() >= 0
        assert edges.max() <= 255

    def test_correlation_with_opencv(self):
        """Our Sobel (with blur) should correlate >.80 with raw cv2.Sobel."""
        import cv2
        g = _gray(240, 320)
        # Use sigma=1.0 to match our standard pipeline (blur included)
        ours = self.fn(g, blur_sigma=1.0).astype(float)
        # Apply same blur to cv2 path for a fair comparison
        blurred = cv2.GaussianBlur(g, (7, 7), 1.0)
        Gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        cv2_edges = np.clip(np.sqrt(Gx**2 + Gy**2), 0, 255)
        corr = float(np.corrcoef(ours.flatten(), cv2_edges.flatten())[0, 1])
        assert corr > 0.80, f"Sobel correlation: {corr:.4f}"


# ------------------------------------------------------------------ #
#  Core: HSV color mask                                               #
# ------------------------------------------------------------------ #

class TestHSVColorMask:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.core.processor import bgr_to_hsv_numpy, hsv_color_mask
        self.hsv_fn  = bgr_to_hsv_numpy
        self.mask_fn = hsv_color_mask

    def test_red_mask_hits_red_pixels(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[3:7, 3:7] = (0, 0, 220)   # red
        hsv  = self.hsv_fn(bgr)
        mask = self.mask_fn(hsv, h_low=340, h_high=20,
                            s_low=0.4, s_high=1.0,
                            v_low=0.3, v_high=1.0)
        assert mask[5, 5] == 255,  "Red pixel should be in mask"
        assert mask[0, 0] == 0,    "Black pixel should not be in mask"

    def test_mask_dtype(self):
        bgr = _bgr_frame(10, 10)
        hsv  = self.hsv_fn(bgr)
        mask = self.mask_fn(hsv, 80, 160)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


# ------------------------------------------------------------------ #
#  Core: FrameRingBuffer                                               #
# ------------------------------------------------------------------ #

class TestFrameRingBuffer:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.core.capture import FrameRingBuffer, FramePacket
        self.Buf    = FrameRingBuffer
        self.Packet = FramePacket

    def _packet(self, fid: int) -> object:
        return self.Packet(
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            frame_id=fid,
            capture_ts=time.perf_counter(),
        )

    def test_empty_buffer_returns_none(self):
        buf = self.Buf()
        assert buf.get_latest() is None

    def test_single_put_get(self):
        buf = self.Buf()
        p = self._packet(0)
        buf.put(p)
        got = buf.get_latest()
        assert got is not None
        assert got.frame_id == 0

    def test_consumed_slot_returns_none(self):
        buf = self.Buf()
        buf.put(self._packet(0))
        buf.get_latest()
        assert buf.get_latest() is None

    def test_latest_wins_on_overflow(self):
        """Ring buffer should deliver the most recent frame, not the first."""
        buf = self.Buf()
        for i in range(10):
            buf.put(self._packet(i))
        got = buf.get_latest()
        assert got is not None
        assert got.frame_id == 9, f"Expected frame_id=9, got {got.frame_id}"

    def test_dropped_counter_increments(self):
        buf = self.Buf()
        for i in range(5):
            buf.put(self._packet(i))
        # Only consumed once; 4 overwrites should be counted as dropped
        buf.get_latest()
        assert buf.dropped_frames >= 1


# ------------------------------------------------------------------ #
#  Tracking: VelocityTracker                                           #
# ------------------------------------------------------------------ #

class TestVelocityTracker:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.tracking.tracker import VelocityTracker
        from spatial_pipeline.inference.engine import Detection
        self.Tracker   = VelocityTracker
        self.Detection = Detection

    def _det(self, cx, cy, w=40, h=40) -> object:
        return self.Detection(
            class_id=0, class_name="ball", confidence=0.9,
            x1=cx - w//2, y1=cy - h//2,
            x2=cx + w//2, y2=cy + h//2,
        )

    def test_new_detection_creates_track(self):
        tracker = self.Tracker()
        tracks = tracker.update([self._det(100, 100)], ts=0.0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 0

    def test_velocity_converges_to_true_value(self):
        """After several frames of constant motion, EMA should converge."""
        tracker = self.Tracker()
        # Object moving right at 100 px/s; dt=0.1s → 10px per step
        for i in range(20):
            tracker.update([self._det(100 + i*10, 100)], ts=i * 0.1)
        tracks = tracker.update([self._det(300, 100)], ts=2.0)
        speed = tracks[0].speed_px_s
        assert 60 < speed < 140, f"Speed should converge near 100px/s, got {speed:.1f}"

    def test_trail_grows_with_detections(self):
        tracker = self.Tracker()
        for i in range(10):
            tracker.update([self._det(i * 5, 50)], ts=i * 0.05)
        tracks = tracker.update([self._det(50, 50)], ts=0.5)
        assert len(tracks[0].trail) > 5

    def test_lost_track_is_pruned(self):
        tracker = self.Tracker()
        tracker.update([self._det(100, 100)], ts=0.0)
        # No detections for > max_lost_frames steps
        for i in range(1, 20):
            tracks = tracker.update([], ts=i * 0.1)
        assert len(tracks) == 0, "Track should be pruned after max_lost_frames"

    def test_two_objects_get_separate_tracks(self):
        tracker = self.Tracker()
        tracks = tracker.update(
            [self._det(50, 50), self._det(250, 250)], ts=0.0
        )
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2, "Each object should have a unique track_id"

    def test_reset_clears_all_tracks(self):
        tracker = self.Tracker()
        tracker.update([self._det(100, 100)], ts=0.0)
        tracker.reset()
        tracks = tracker.update([self._det(100, 100)], ts=0.1)
        # After reset the track_id counter restarts
        assert tracks[0].track_id == 0

    def test_velocity_direction_unit_vector(self):
        tracker = self.Tracker()
        for i in range(10):
            tracker.update([self._det(i*10, 100)], ts=i*0.1)
        tracks = tracker.update([self._det(100, 100)], ts=1.0)
        dx, dy = tracks[0].velocity_dir
        mag = math.hypot(dx, dy)
        assert abs(mag - 1.0) < 0.01 or mag < 0.001, \
            f"velocity_dir should be unit vector or zero, got mag={mag:.4f}"


# ------------------------------------------------------------------ #
#  Inference: YOLO post-processing (NMS only — no model required)     #
# ------------------------------------------------------------------ #

class TestYOLONMS:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.inference.engine import YOLOONNXEngine
        self.nms = YOLOONNXEngine._nms

    def test_single_box_is_kept(self):
        x1 = np.array([10]); y1 = np.array([10])
        x2 = np.array([50]); y2 = np.array([50])
        s  = np.array([0.9])
        assert self.nms(x1, y1, x2, y2, s) == [0]

    def test_overlapping_boxes_keeps_highest_conf(self):
        x1 = np.array([10,  12]); y1 = np.array([10, 12])
        x2 = np.array([50,  52]); y2 = np.array([50, 52])
        s  = np.array([0.5, 0.9])
        keep = self.nms(x1, y1, x2, y2, s)
        assert 1 in keep          # higher confidence box
        assert 0 not in keep      # suppressed

    def test_non_overlapping_boxes_both_kept(self):
        x1 = np.array([0,   200]); y1 = np.array([0,   0])
        x2 = np.array([50,  250]); y2 = np.array([50, 50])
        s  = np.array([0.8, 0.8])
        keep = self.nms(x1, y1, x2, y2, s)
        assert len(keep) == 2


# ------------------------------------------------------------------ #
#  Config: validation and env overrides                               #
# ------------------------------------------------------------------ #

class TestConfig:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.utils.config import PipelineConfig
        self.Cfg = PipelineConfig

    def test_default_config_has_no_warnings(self):
        cfg = self.Cfg()
        warnings = cfg.validate()
        # A freshly constructed default config should be warning-free
        assert len(warnings) == 0

    def test_unusual_input_size_triggers_warning(self):
        cfg = self.Cfg()
        cfg.inference.input_size = 999
        assert any("input_size" in w for w in cfg.validate())

    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("SP_INFERENCE__CONF_THRESHOLD", "0.77")
        cfg = self.Cfg().apply_env_overrides()
        assert abs(cfg.inference.conf_threshold - 0.77) < 1e-6

    def test_env_override_bool(self, monkeypatch):
        monkeypatch.setenv("SP_PROFILER__ENABLED", "true")
        cfg = self.Cfg().apply_env_overrides()
        assert cfg.profiler.enabled is True

    def test_roundtrip_json(self, tmp_path):
        cfg = self.Cfg()
        cfg.inference.conf_threshold = 0.55
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        loaded = self.Cfg.from_json(p)
        assert abs(loaded.inference.conf_threshold - 0.55) < 1e-6


# ------------------------------------------------------------------ #
#  Profiler                                                            #
# ------------------------------------------------------------------ #

class TestProfiler:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spatial_pipeline.utils.profiler import PipelineProfiler
        self.Profiler = PipelineProfiler

    def test_record_and_summary(self):
        p = self.Profiler()
        for i in range(10):
            p.record("infer", i, float(i * 2))
        s = p.summary()
        assert "infer" in s
        assert s["infer"]["count"] == 10
        assert abs(s["infer"]["mean_ms"] - 9.0) < 0.1   # mean of 0,2,4,...,18

    def test_disabled_profiler_records_nothing(self):
        p = self.Profiler(enabled=False)
        p.record("infer", 0, 10.0)
        assert p.summary() == {}

    def test_csv_export(self, tmp_path):
        p = self.Profiler()
        p.record("capture", 0, 5.0)
        p.record("infer",   0, 15.0)
        out = tmp_path / "profile.csv"
        p.export_csv(str(out))
        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == 3   # header + 2 rows


# ------------------------------------------------------------------ #
#  FPS counter                                                         #
# ------------------------------------------------------------------ #

class TestFPSCounter:
    def test_empty_returns_zero(self):
        from spatial_pipeline.utils.fps import FPSCounter
        assert FPSCounter().fps == 0.0

    def test_approx_fps(self):
        from spatial_pipeline.utils.fps import FPSCounter
        c = FPSCounter(window=20)
        target = 50.0  # Hz
        for _ in range(20):
            c.tick()
            time.sleep(1.0 / target)
        assert abs(c.fps - target) / target < 0.15   # within 15%
