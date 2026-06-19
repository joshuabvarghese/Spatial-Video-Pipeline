# Spatial Pipeline

**Edge-deployed real-time spatial video pipeline** — multithreaded, zero-lag, sub-5 ms AI inference.

```
Capture (60 FPS) ──► Process ──► YOLO Inference ──► Render (≥30 FPS guaranteed)
     Thread 1           Thread 2       Thread 3           Main thread
```

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17%2B-orange)](https://onnxruntime.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## The Problem This Solves

Naïve single-threaded CV pipelines serialize every step:

```
capture → process → infer → display → capture → process → ...
```

At 30ms inference latency, display FPS drops to 33 FPS maximum — and any
variance in inference (e.g. a heavy detection frame) causes visible hitches.

This pipeline isolates every stage in its own thread and connects them
with **latest-wins ring buffers** rather than queues.  The display thread
runs at the monitor's native rate regardless of inference speed.

---

## Architecture

### Thread Topology

```
┌─────────────────────────────────────────────────────────────┐
│  Thread 1: CameraCapture                                    │
│    └─► drains kernel camera buffer at native FPS            │
│    └─► pushes to FrameRingBuffer (3-slot, lock-minimised)   │
│                       │                                     │
│  Thread 2: ImageProcessor                                   │
│    ├─► Custom BGR→HSV  (vectorised NumPy, no cv2.cvtColor)  │
│    ├─► Custom Sobel    (manual 3×3 unroll, no cv2.Sobel)    │
│    └─► HSV color mask  (for blob tracking mode)             │
│                       │                                     │
│  Thread 3: YOLOInferenceThread                              │
│    ├─► YOLOv8-nano via ONNX Runtime                         │
│    ├─► Automatic EP: CUDA → CoreML → CPU                    │
│    └─► Pure-NumPy NMS  (no torchvision, no cv2.dnn)         │
│                       │                                     │
│  Main Thread: RenderCompositor                              │
│    ├─► VelocityTracker  (IoU match + EMA + Kalman)          │
│    ├─► FrameRenderer    (trail, arrow, bbox, HUD)           │
│    └─► cv2.imshow / VideoWriter                             │
└─────────────────────────────────────────────────────────────┘
```

### Ring Buffer Design

Each inter-thread handoff uses a 3-slot "latest-wins" ring buffer:

```
Writer:  always writes to slot (write_idx % 3)
Reader:  atomically swaps the "latest" slot pointer

Critical section: one integer swap (~100 ns)
No blocking, no queue accumulation, no lag growth.
```

If inference runs at 15 FPS while capture runs at 60 FPS, the display
thread shows 60 FPS of real camera frames with 15 detection updates per
second — never buffered stale frames from 250 ms ago.

---

## Features

| Feature | Detail |
|---------|--------|
| **From-scratch HSV** | Full ITU-R BT.601 vectorised formula, 6 broadcast ops |
| **From-scratch Sobel** | Manual 3×3 kernel unroll as overlapping array slices |
| **YOLOv8-nano ONNX** | Auto-downloads + exports once, cached at `~/.cache/spatial_pipeline/` |
| **Pixel velocity** | EMA-smoothed, direction arrow projects future position |
| **Kalman filter** | 4-state constant-velocity model, handles occlusion |
| **Fading trail** | 60-point deque, alpha decays linearly with age |
| **HUD** | Per-thread FPS, inference latency, pipeline latency, health dot |
| **Profiler** | Per-stage mean/P50/P95/P99/max, CSV export |
| **Config system** | Typed dataclasses, JSON file, env-var overrides (12-factor) |
| **Headless mode** | `--no-display` for server/embedded benchmarking |
| **Video output** | `--output out.mp4` writes annotated stream |

### Live Key Bindings

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `E` | Toggle Sobel edge overlay |
| `T` | Toggle trailing path |
| `V` | Toggle velocity arrows |
| `H` | Toggle HUD |

---

## Installation

```bash
# Clone
git clone https://github.com/your-org/spatial-pipeline
cd spatial-pipeline

# Install (CPU)
pip install -e ".[dev]"

# Install (GPU — CUDA 12)
pip install -e ".[gpu,dev]"
```

**Requirements:** Python ≥ 3.10, OpenCV, NumPy, ONNX Runtime

---

## Quick Start

```bash
# Webcam (camera index 0)
python scripts/run_pipeline.py --source 0

# Video file with Sobel overlay and profiler
python scripts/run_pipeline.py --source demo.mp4 --show-edges --profile

# Track only people, high confidence, save to file
python scripts/run_pipeline.py --source 0 --classes person --conf 0.55 --output out.mp4

# Track a red ball (HSV color mode)
python scripts/run_pipeline.py --source 0 --track-color red

# Headless 30-second benchmark
python scripts/run_pipeline.py --source 0 --no-display --duration 30 --profile

# Load from config file (override with env vars)
SP_INFERENCE__CONF_THRESHOLD=0.6 \
python scripts/run_pipeline.py --config configs/default.json
```

---

## Programmatic API

```python
from spatial_pipeline import Pipeline
from spatial_pipeline.utils.config import (
    PipelineConfig, CaptureConfig, InferenceConfig, RenderConfig
)

cfg = PipelineConfig()
cfg.capture.source            = 0
cfg.inference.conf_threshold  = 0.45
cfg.inference.target_classes  = ["person", "cell phone"]
cfg.render.show_edges         = True
cfg.profiler.enabled          = True

pipeline = Pipeline(cfg)
pipeline.run()   # blocks; Ctrl-C to exit
```

---

## Project Structure

```
spatial_pipeline/          # installable package
├── __init__.py            # public API surface
├── pipeline.py            # top-level orchestrator
├── core/
│   ├── capture.py         # CameraCapture thread, FrameRingBuffer
│   └── processor.py       # custom HSV + Sobel + ImageProcessor thread
├── inference/
│   └── engine.py          # YOLOONNXEngine + InferenceThread
├── tracking/
│   └── tracker.py         # VelocityTracker, Kalman filter, trail
├── viz/
│   └── renderer.py        # FrameRenderer, HUD compositor
└── utils/
    ├── config.py           # typed PipelineConfig, JSON/env overrides
    ├── fps.py              # rolling FPS counter
    └── profiler.py         # per-stage timing, CSV export

scripts/
└── run_pipeline.py        # CLI entry point

tests/
└── test_pipeline.py       # 25 unit tests, fully deterministic

benchmarks/
└── bench_pipeline.py      # microbenchmarks vs OpenCV baselines

configs/
└── default.json           # reference config
```

---

## Custom Image Processing (No OpenCV Wrappers)

### BGR → HSV (`core/processor.py :: bgr_to_hsv_numpy`)

The standard OpenCV `cv2.cvtColor` call is a black box.  This reimplements
the full ITU-R formula using six NumPy broadcast operations over the entire
image — no Python loops over pixels:

```
Cmax = max(R, G, B)     Cmin = min(R, G, B)     Δ = Cmax − Cmin
V = Cmax
S = Δ / Cmax   (guarded)
H = 60° × { (G−B)/Δ mod 6  if R=Cmax
           { (B−R)/Δ + 2    if G=Cmax
           { (R−G)/Δ + 4    if B=Cmax
```

### Sobel Edges (`core/processor.py :: sobel_edges_numpy`)

Rather than calling `cv2.Sobel`, the 3×3 convolution is expressed as
nine overlapping array slices — each a vectorised multiply over the full
image:

```python
p = np.pad(img, 1, mode="reflect")
Gx = (p[0:h, 0:w]*-1 + p[0:h, 2:w+2]*1 +
      p[1:h+1, 0:w]*-2 + p[1:h+1, 2:w+2]*2 +
      p[2:h+2, 0:w]*-1 + p[2:h+2, 2:w+2]*1)
# ... Gy similarly
magnitude = np.sqrt(Gx**2 + Gy**2).clip(0, 255).astype(np.uint8)
```

Correlation with `cv2.Sobel` on natural images: **r > 0.87** (difference
is the Gaussian pre-blur fused into our implementation).

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=spatial_pipeline --cov-report=term-missing

# Specific class
pytest tests/ -v -k "TestVelocityTracker"
```

All 25 tests are deterministic (no webcam, no network, synthetic inputs only).

---

## Running Benchmarks

```bash
python benchmarks/bench_pipeline.py
python benchmarks/bench_pipeline.py --resolution 1280x720 --iters 200
```

Sample output (Apple M2, 640×480):

```
════════════════════════════════════════════════════════════════════
  SPATIAL PIPELINE BENCHMARKS  —  480×640  iters=100

── BGR → HSV ─────────────────────────────────────────────────────
  bgr_to_hsv_numpy  (custom NumPy)        mean=  6.21ms  p50=  6.10ms
  cv2.cvtColor      (OpenCV C++)          mean=  0.31ms  p50=  0.30ms
  Ratio: 20.0×  (slower — Cython/Numba JIT planned for v2)

── Sobel Edge Detection ──────────────────────────────────────────
  sobel_edges_numpy  (custom NumPy)       mean=  9.45ms  p50=  9.20ms
  cv2.Sobel          (OpenCV C++)         mean=  0.48ms  p50=  0.47ms
  Ratio: 19.7×  (custom includes Gaussian pre-blur)

── FrameRingBuffer  put/get throughput ──────────────────────────
  FrameRingBuffer.put()                   mean=  0.003ms  → 333,000 fps
  FrameRingBuffer.get_latest()            mean=  0.001ms

── ONNX Inference ────────────────────────────────────────────────
  YOLOONNXEngine.infer() 320px CPU        mean=  4.8ms  p95= 6.2ms
  Theoretical inference throughput: 208 fps
════════════════════════════════════════════════════════════════════
```

> **Why is custom NumPy slower than cv2?**  OpenCV's C++ backend runs
> SIMD-vectorised loops with zero GIL overhead.  The NumPy ops still beat
> pure Python by ~100×, and on platforms without a full OpenCV build (e.g.
> minimal Docker images, some embedded targets) they're the only option.
> A Cython or Numba JIT path is tracked as a future improvement.

---

## Configuration Reference

All settings live in `spatial_pipeline/utils/config.py` as typed dataclasses.
Override via JSON config file, CLI flags, or environment variables:

```bash
# Environment variable convention:  SP_<SECTION>__<FIELD>=value
SP_INFERENCE__CONF_THRESHOLD=0.55
SP_RENDER__SHOW_EDGES=true
SP_PROFILER__ENABLED=true
SP_CAPTURE__TARGET_FPS=30
```

---

## Roadmap

- [ ] **Numba JIT** for HSV / Sobel — close the gap with OpenCV C++
- [ ] **TensorRT backend** — FP16 quantised inference, <2 ms on Jetson
- [ ] **Re-ID embeddings** — persist track IDs across occlusion using cosine similarity
- [ ] **RTSP source** — replace cv2.VideoCapture with a custom RTSP demuxer
- [ ] **gRPC metrics sink** — stream HUD data to Prometheus / Grafana
- [ ] **Multi-camera** — fan-out ring buffers to N parallel inference threads

---

## License

MIT — see [LICENSE](LICENSE).
