"""
demo_synthetic.py — Headless Demo (No Webcam Required)
=======================================================

Validates the entire pipeline using synthetic frames:
  - Moving colored blobs (simulates tracked objects)
  - Custom HSV + Sobel tested on those frames
  - YOLO inference tested on a real downloaded test image
  - Tracker velocity computed over multiple frames
  - Renderer output saved to demo_output.jpg

Run with:
    python demo_synthetic.py
"""

import time
import math
import numpy as np
import cv2
import sys
from pathlib import Path


# ------------------------------------------------------------------ #
#  Synthetic frame generator                                           #
# ------------------------------------------------------------------ #

def make_synthetic_frame(t: float, w: int = 640, h: int = 480) -> np.ndarray:
    """
    Generates a frame with:
      - Dark gradient background
      - Two moving colored circles (simulate tracked objects)
      - Some static edges (simulate real-world clutter)
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Subtle gradient background
    for y in range(h):
        frame[y, :] = [int(20 + y * 0.04), int(15 + y * 0.03), int(25 + y * 0.05)]

    # Static rectangle (creates edges for Sobel)
    cv2.rectangle(frame, (50, 50), (200, 180), (60, 60, 80), -1)
    cv2.rectangle(frame, (400, 250), (590, 420), (40, 70, 60), -1)

    # Moving ball 1: red, circular motion
    cx1 = int(w / 2 + 150 * math.cos(t * 1.5))
    cy1 = int(h / 2 + 100 * math.sin(t * 1.5))
    cv2.circle(frame, (cx1, cy1), 30, (30, 30, 220), -1)   # BGR red
    cv2.circle(frame, (cx1, cy1), 30, (255, 255, 255), 2)

    # Moving ball 2: green, diagonal bounce
    cx2 = int(100 + (w - 200) * (0.5 + 0.5 * math.sin(t * 0.8)))
    cy2 = int(100 + (h - 200) * (0.5 + 0.5 * math.cos(t * 1.1)))
    cv2.circle(frame, (cx2, cy2), 22, (30, 200, 30), -1)   # BGR green
    cv2.circle(frame, (cx2, cy2), 22, (200, 255, 200), 2)

    return frame


# ------------------------------------------------------------------ #
#  Unit tests for each module                                          #
# ------------------------------------------------------------------ #

def test_custom_hsv():
    print("\n[TEST] Custom BGR→HSV conversion")
    from processor import bgr_to_hsv_numpy

    # Pure red pixel in BGR
    red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
    hsv = bgr_to_hsv_numpy(red_bgr)
    H, S, V = hsv[0, 0]
    print(f"  Red BGR → HSV: H={H:.1f}° S={S:.3f} V={V:.3f}")
    assert abs(H - 0.0) < 2 or abs(H - 360.0) < 2, f"Hue should be ~0°, got {H}"
    assert abs(S - 1.0) < 0.01, f"Saturation should be 1.0, got {S}"
    assert abs(V - 1.0) < 0.01, f"Value should be 1.0, got {V}"

    # Pure green pixel
    green_bgr = np.array([[[0, 255, 0]]], dtype=np.uint8)
    hsv_g = bgr_to_hsv_numpy(green_bgr)
    H_g = hsv_g[0, 0, 0]
    print(f"  Green BGR → HSV: H={H_g:.1f}° (expected ~120°)")
    assert abs(H_g - 120.0) < 2, f"Hue should be ~120°, got {H_g}"

    # Test on a full frame
    frame = make_synthetic_frame(0.0)
    t0 = time.perf_counter()
    hsv_full = bgr_to_hsv_numpy(frame)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Full 640×480 frame: {ms:.1f}ms | shape={hsv_full.shape}")
    print("  ✓ Custom HSV conversion PASSED")


def test_sobel():
    print("\n[TEST] Custom Sobel edge detection")
    from processor import sobel_edges_numpy, gaussian_blur_numpy

    frame = make_synthetic_frame(0.0)
    gray = (0.114 * frame[..., 0].astype(np.float32) +
            0.587 * frame[..., 1].astype(np.float32) +
            0.299 * frame[..., 2].astype(np.float32)).astype(np.uint8)

    t0 = time.perf_counter()
    edges = sobel_edges_numpy(gray, blur_sigma=1.0)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  Sobel on 640×480: {ms:.1f}ms | dtype={edges.dtype} | max={edges.max()}")
    assert edges.dtype == np.uint8, "Expected uint8 output"
    assert edges.max() > 50, "Expected non-trivial edges in synthetic frame"

    # Compare with OpenCV's cv2.Sobel for sanity
    gray_f = gray.astype(np.float32)
    Gx_cv = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    Gy_cv = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    edges_cv = np.clip(np.sqrt(Gx_cv**2 + Gy_cv**2), 0, 255).astype(np.uint8)

    # Should be similar (not identical due to blur pre-step)
    corr = np.corrcoef(edges.flatten(), edges_cv.flatten())[0, 1]
    print(f"  Correlation with cv2.Sobel: {corr:.3f} (>0.8 expected)")
    assert corr > 0.80, f"Sobel correlation too low: {corr:.3f}"
    print("  ✓ Custom Sobel PASSED")


def test_capture_buffer():
    print("\n[TEST] FrameRingBuffer (no camera)")
    from capture import FrameRingBuffer, FramePacket

    buf = FrameRingBuffer()
    frame = make_synthetic_frame(0.0)

    # Test put/get
    p1 = FramePacket(frame=frame, frame_id=0, capture_ts=time.perf_counter())
    buf.put(p1)
    got = buf.get_latest()
    assert got is not None, "Should get packet"
    assert got.frame_id == 0

    # Test overwrite behaviour (ring buffer drops intermediate)
    buf.put(FramePacket(frame=frame, frame_id=1, capture_ts=time.perf_counter()))
    buf.put(FramePacket(frame=frame, frame_id=2, capture_ts=time.perf_counter()))
    buf.put(FramePacket(frame=frame, frame_id=3, capture_ts=time.perf_counter()))
    got2 = buf.get_latest()
    assert got2 is not None
    # Should have the most recent
    assert got2.frame_id == 3, f"Expected frame_id=3, got {got2.frame_id}"
    print(f"  Ring buffer skipped to frame_id={got2.frame_id} as expected")
    print("  ✓ FrameRingBuffer PASSED")


def test_tracker():
    print("\n[TEST] VelocityTracker")
    from tracker import VelocityTracker
    from inference import Detection

    tracker = VelocityTracker(ema_alpha=0.6)

    # Simulate object moving right at 100 px/s
    detections = [
        Detection(0, "ball", 0.9, 100, 200, 160, 260),
    ]
    t0 = time.perf_counter()
    tracks = tracker.update(detections, t0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 0

    # Move right by 10px after 0.1s → 100 px/s
    t1 = t0 + 0.1
    detections2 = [
        Detection(0, "ball", 0.9, 110, 200, 170, 260),
    ]
    tracks = tracker.update(detections2, t1)
    speed = tracks[0].speed_px_per_s
    print(f"  Speed after 1 update: {speed:.1f} px/s (expected ~60 with EMA)")
    assert speed > 10, "Speed should be positive"

    # Several updates to let EMA converge
    for i in range(10):
        t_i = t1 + (i + 1) * 0.1
        x = 110 + (i + 1) * 10
        dets = [Detection(0, "ball", 0.9, x, 200, x + 60, 260)]
        tracks = tracker.update(dets, t_i)

    speed_conv = tracks[0].speed_px_per_s
    trail_len = len(tracks[0].trail)
    print(f"  Speed after 10 updates: {speed_conv:.1f} px/s (converging to 100)")
    print(f"  Trail length: {trail_len}")
    assert 50 < speed_conv < 150, f"Converged speed out of range: {speed_conv}"
    print("  ✓ VelocityTracker PASSED")


def test_renderer():
    print("\n[TEST] FrameRenderer")
    from renderer import FrameRenderer, RenderConfig
    from tracker import TrackedObject, TrailPoint
    from inference import Detection
    from collections import deque

    frame = make_synthetic_frame(1.0)

    # Build a fake track
    det = Detection(0, "ball", 0.88, 200, 150, 280, 230)
    trail = deque(maxlen=60)
    now = time.perf_counter()
    for i in range(20):
        trail.append(TrailPoint(cx=200 - i * 3, cy=180 - i * 2,
                                ts=now - i * 0.05, alpha=1.0 - i * 0.05))
    track = TrackedObject(
        track_id=0, class_name="ball", last_detection=det,
        last_ts=now, vx=120.0, vy=-40.0, trail=trail
    )

    edges = np.random.randint(0, 80, (frame.shape[0], frame.shape[1]), dtype=np.uint8)
    hud = {'capture_fps': 59.2, 'process_fps': 55.1, 'infer_fps': 28.4,
           'render_fps': 31.5, 'infer_ms': 34.2, 'pipeline_ms': 67.8,
           'dropped': 0, 'tracks': 1}

    cfg = RenderConfig(show_edges=True, show_trail=True, show_velocity_arrow=True)
    renderer = FrameRenderer(cfg)

    t0 = time.perf_counter()
    annotated = renderer.render(frame, [track], edges=edges, hud_data=hud)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  Render time: {ms:.2f}ms | output shape: {annotated.shape}")
    assert annotated.shape == frame.shape
    print("  ✓ FrameRenderer PASSED")

    return annotated


def test_yolo_on_test_image():
    print("\n[TEST] YOLOv8-nano inference on a real test image")
    from inference import YOLOONNXEngine
    import urllib.request

    # Download a small test image
    test_img_path = "/tmp/test_yolo.jpg"
    if not Path(test_img_path).exists():
        url = "https://raw.githubusercontent.com/ultralytics/assets/main/zidane.jpg"
        try:
            print(f"  Downloading test image…")
            urllib.request.urlretrieve(url, test_img_path)
        except Exception as e:
            print(f"  Could not download test image ({e}). Skipping live YOLO test.")
            return None

    img = cv2.imread(test_img_path)
    if img is None:
        print("  Could not load test image. Skipping.")
        return None

    engine = YOLOONNXEngine(conf_threshold=0.35)

    # Warmup
    engine.infer(img)

    t0 = time.perf_counter()
    detections = engine.infer(img)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  Inference time: {ms:.1f}ms")
    print(f"  Detections: {len(detections)}")
    for d in detections:
        print(f"    {d.class_name} {d.confidence:.0%} [{d.x1},{d.y1}→{d.x2},{d.y2}]")

    assert len(detections) > 0, "Expected at least 1 detection on test image"
    print("  ✓ YOLO inference PASSED")
    return detections, img


def save_demo_output(annotated: np.ndarray, path: str = "demo_output.jpg"):
    cv2.imwrite(path, annotated)
    print(f"\n[DEMO] Annotated output saved → {path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("  SPATIAL PIPELINE — SYNTHETIC DEMO & UNIT TESTS")
    print("=" * 60)

    results = {}

    test_capture_buffer()
    test_custom_hsv()
    test_sobel()
    test_tracker()
    annotated = test_renderer()

    yolo_result = test_yolo_on_test_image()

    # Save renderer output
    if annotated is not None:
        save_demo_output(annotated, "demo_output.jpg")

    # If YOLO ran, save a labelled test image too
    if yolo_result is not None:
        from renderer import FrameRenderer, RenderConfig
        from tracker import TrackedObject, TrailPoint
        from collections import deque
        import time as tm

        detections, img = yolo_result
        now = tm.perf_counter()

        # Wrap detections as minimal TrackedObject list for renderer
        tracks = []
        for i, d in enumerate(detections):
            t = TrackedObject(track_id=i, class_name=d.class_name,
                              last_detection=d, last_ts=now,
                              vx=0, vy=0)
            tracks.append(t)

        renderer = FrameRenderer(RenderConfig(show_hud=True))
        hud = {'render_fps': 0, 'capture_fps': 0, 'infer_ms': 0,
               'pipeline_ms': 0, 'dropped': 0, 'tracks': len(tracks)}
        out = renderer.render(img, tracks, hud_data=hud)
        cv2.imwrite("yolo_demo_output.jpg", out)
        print(f"[DEMO] YOLO annotated image saved → yolo_demo_output.jpg")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nTo run the live pipeline:")
    print("  python pipeline.py --source 0")
    print("  python pipeline.py --source 0 --show-edges --track-color red --profile")


if __name__ == "__main__":
    main()
