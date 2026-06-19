"""
benchmarks/bench_pipeline.py
-----------------------------
Microbenchmarks for the compute-heavy pipeline stages.

Measures throughput and latency for:
  - bgr_to_hsv_numpy (custom) vs cv2.cvtColor
  - sobel_edges_numpy (custom) vs cv2.Sobel
  - ONNX inference at 320 / 640 input sizes
  - End-to-end ImageProcessor.run() time
  - Ring buffer put/get throughput

Run with:
    python benchmarks/bench_pipeline.py
    python benchmarks/bench_pipeline.py --resolution 1280x720
    python benchmarks/bench_pipeline.py --iters 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _timeit(fn, iters: int = 100) -> dict:
    """Return timing stats (ms) for *fn()* called *iters* times."""
    # Warmup
    for _ in range(max(5, iters // 10)):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    n = len(samples)
    return {
        "mean":  statistics.mean(samples),
        "p50":   samples[n // 2],
        "p95":   samples[int(n * 0.95)],
        "p99":   samples[min(int(n * 0.99), n - 1)],
        "min":   samples[0],
        "max":   samples[-1],
    }


def _row(label: str, stats: dict) -> str:
    return (
        f"  {label:<38}  "
        f"mean={stats['mean']:>6.2f}ms  "
        f"p50={stats['p50']:>6.2f}ms  "
        f"p95={stats['p95']:>6.2f}ms  "
        f"min={stats['min']:>6.2f}ms"
    )


def _hr(title: str = "") -> None:
    w = 88
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * w)


# ------------------------------------------------------------------ #
#  Benchmark groups                                                    #
# ------------------------------------------------------------------ #

def bench_hsv(frame: np.ndarray, iters: int) -> None:
    _hr("BGR → HSV")
    from spatial_pipeline.core.processor import bgr_to_hsv_numpy

    s_ours = _timeit(lambda: bgr_to_hsv_numpy(frame), iters)
    s_cv2  = _timeit(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), iters)

    print(_row("bgr_to_hsv_numpy  (custom NumPy)", s_ours))
    print(_row("cv2.cvtColor      (OpenCV C++)",   s_cv2))
    ratio = s_ours["mean"] / s_cv2["mean"]
    print(f"  Ratio (custom / cv2): {ratio:.1f}×  "
          f"({'faster' if ratio < 1 else 'slower'})\n")


def bench_sobel(frame: np.ndarray, iters: int) -> None:
    _hr("Sobel Edge Detection")
    from spatial_pipeline.core.processor import sobel_edges_numpy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    s_ours = _timeit(lambda: sobel_edges_numpy(gray, blur_sigma=1.0), iters)
    s_cv2  = _timeit(lambda: _cv2_sobel(gray), iters)

    print(_row("sobel_edges_numpy  (custom NumPy)", s_ours))
    print(_row("cv2.Sobel          (OpenCV C++)",   s_cv2))
    ratio = s_ours["mean"] / s_cv2["mean"]
    print(f"  Ratio: {ratio:.1f}×  "
          f"(custom includes Gaussian pre-blur; cv2 does not)\n")


def _cv2_sobel(gray: np.ndarray) -> np.ndarray:
    gf = gray.astype(np.float32)
    Gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(Gx**2 + Gy**2), 0, 255).astype(np.uint8)


def bench_ring_buffer(iters: int) -> None:
    _hr("FrameRingBuffer  put/get throughput")
    from spatial_pipeline.core.capture import FrameRingBuffer, FramePacket

    buf = FrameRingBuffer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _put():
        buf.put(FramePacket(frame=frame, frame_id=0, capture_ts=time.perf_counter()))

    def _get():
        buf.get_latest()

    s_put = _timeit(_put, iters)
    s_get = _timeit(_get, iters)
    print(_row("FrameRingBuffer.put()", s_put))
    print(_row("FrameRingBuffer.get_latest()", s_get))
    # Throughput
    fps_put = 1000.0 / s_put["mean"]
    print(f"  Theoretical put() throughput: {fps_put:,.0f} fps\n")


def bench_inference(frame: np.ndarray, iters: int) -> None:
    _hr("ONNX Inference")
    from spatial_pipeline.inference.engine import YOLOONNXEngine, InferenceConfig

    engine = YOLOONNXEngine(InferenceConfig(input_size=320))
    s = _timeit(lambda: engine.infer(frame), iters)
    print(_row(f"YOLOONNXEngine.infer()  320px  {engine.active_provider}", s))
    print(f"  Theoretical inference throughput: {1000/s['mean']:.1f} fps\n")


def bench_processor_thread(frame: np.ndarray, iters: int) -> None:
    _hr("ImageProcessor  (full stage, threaded)")
    from spatial_pipeline.core.capture import FramePacket
    from spatial_pipeline.core.processor import ImageProcessor, ProcessorConfig

    proc = ImageProcessor(ProcessorConfig(sobel_sigma=1.0))
    proc.start()

    completed = [0]

    def _round_trip():
        pkt = FramePacket(frame=frame, frame_id=0, capture_ts=time.perf_counter())
        proc.submit(pkt)
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            r = proc.get_latest()
            if r is not None:
                completed[0] += 1
                return r
            time.sleep(0.001)

    s = _timeit(_round_trip, min(iters, 50))
    proc.stop()
    proc.join(timeout=2)
    print(_row("ImageProcessor  submit→get round-trip", s))
    print(f"  Stage FPS ceiling: {1000/s['mean']:.1f} fps\n")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    p = argparse.ArgumentParser(description="Spatial Pipeline Benchmarks")
    p.add_argument("--resolution", default="640x480",
                   help="Frame resolution WxH (default: 640x480)")
    p.add_argument("--iters", type=int, default=100,
                   help="Iterations per benchmark (default: 100)")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip ONNX inference benchmark")
    args = p.parse_args()

    w, h = (int(x) for x in args.resolution.lower().split("x"))
    rng   = np.random.default_rng(42)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    print(f"\n{'═'*88}")
    print(f"  SPATIAL PIPELINE BENCHMARKS  —  {h}×{w}  iters={args.iters}")
    print(f"{'═'*88}\n")

    bench_hsv(frame, args.iters)
    bench_sobel(frame, args.iters)
    bench_ring_buffer(args.iters)

    if not args.skip_inference:
        bench_inference(frame, min(args.iters, 50))

    bench_processor_thread(frame, min(args.iters, 50))

    print(f"{'═'*88}\n")


if __name__ == "__main__":
    main()
