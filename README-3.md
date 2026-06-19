# Edge-Deployed Real-Time Spatial Video Pipeline

A production-grade, multithreaded computer vision pipeline for real-time object detection,
pixel-velocity tracking, and spatial analytics — built without CPU/GPU bottlenecks.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    THREAD TOPOLOGY                       │
│                                                          │
│  Thread 1: CameraCapture                                 │
│    └─► FrameRingBuffer (lock-free, 3-slot)               │
│              │                                           │
│  Thread 2: ImageProcessor                                │
│    ├─► Custom HSV pipeline (no OpenCV wrapper)           │
│    ├─► Custom Sobel edge detection (raw convolution)     │
│    └─► ColorBlobTracker                                  │
│              │                                           │
│  Thread 3: YOLOInferenceEngine                           │
│    ├─► YOLOv8-nano via ONNX Runtime                      │
│    ├─► BoundingBox + class label                         │
│    └─► Pixel velocity calculator                         │
│              │                                           │
│  Thread 4: RenderCompositor (main thread)                │
│    ├─► Overlay: trailing path, velocity vector           │
│    ├─► HUD: FPS, latency, thread health                  │
│    └─► Display / optional MJPEG stream                   │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Zero-drop capture**: Ring buffer decouples camera I/O from inference
- **Custom image processing**: HSV conversion and Sobel edge detection written from scratch using NumPy matrix ops — no `cv2.Sobel`, no `cv2.cvtColor`
- **YOLOv8-nano ONNX**: Lightweight inference, ~5ms per frame on CPU
- **Pixel velocity tracking**: Smooth exponential moving-average velocity with direction vector
- **Trailing path**: Deque of past centroid positions drawn as a fading polyline
- **Live HUD**: Per-thread FPS, pipeline latency, inference time, drop rate
- **Profiler**: `--profile` flag dumps per-stage timing to CSV

## Installation

```bash
pip install opencv-python numpy onnxruntime ultralytics
python pipeline.py --source 0          # webcam index 0
python pipeline.py --source video.mp4  # file
python pipeline.py --source 0 --profile
python pipeline.py --source 0 --show-edges   # overlay Sobel edges
python pipeline.py --source 0 --track-color  # HSV color blob mode
```

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Entry point, CLI args, thread orchestration |
| `capture.py` | CameraCapture thread + FrameRingBuffer |
| `processor.py` | Custom HSV + Sobel from scratch |
| `inference.py` | YOLOv8-nano ONNX inference thread |
| `tracker.py` | Velocity calc, trailing path, Kalman smoother |
| `renderer.py` | HUD compositor, overlay drawing |
| `profiler.py` | Per-stage timing, CSV export |
| `demo_synthetic.py` | Headless demo using synthetic frames (no webcam needed) |
