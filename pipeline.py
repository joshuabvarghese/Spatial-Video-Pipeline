"""
pipeline.py — Edge-Deployed Real-Time Spatial Video Pipeline
============================================================

Entry point.  Spins up all threads, orchestrates the data flow,
and runs the display loop at ≥30 FPS.

Usage
-----
    python pipeline.py --source 0
    python pipeline.py --source video.mp4 --show-edges
    python pipeline.py --source 0 --track-color red --profile
    python pipeline.py --source 0 --classes person --conf 0.45
    python pipeline.py --source 0 --no-display --duration 30  # headless benchmark
"""

import argparse
import time
import sys
import signal
import threading

import cv2
import numpy as np

from capture import CameraCapture
from processor import ImageProcessor
from inference import YOLOONNXEngine, InferenceThread
from tracker import VelocityTracker
from renderer import FrameRenderer, RenderConfig
from profiler import PipelineProfiler


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Edge-Deployed Real-Time Spatial Video Pipeline"
    )
    p.add_argument("--source", default=0,
                   help="Camera index (int) or video file path")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Filter YOLO to specific class names, e.g. 'person cat'")
    p.add_argument("--conf", type=float, default=0.40,
                   help="YOLO confidence threshold (default 0.40)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold (default 0.45)")
    p.add_argument("--input-size", type=int, default=320,
                   help="YOLO model input resolution (default 320)")
    p.add_argument("--track-color", default=None,
                   choices=["red", "orange", "yellow", "green", "blue"],
                   help="Enable HSV color-blob tracker for this color")
    p.add_argument("--show-edges", action="store_true",
                   help="Overlay Sobel edges on display frame")
    p.add_argument("--sobel-sigma", type=float, default=1.0,
                   help="Gaussian pre-blur sigma for Sobel (default 1.0)")
    p.add_argument("--no-trail", action="store_true",
                   help="Disable trailing path")
    p.add_argument("--no-arrows", action="store_true",
                   help="Disable velocity arrows")
    p.add_argument("--no-hud", action="store_true",
                   help="Disable HUD overlay")
    p.add_argument("--profile", action="store_true",
                   help="Enable profiler; dumps CSV on exit")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode (no window); useful for benchmarking")
    p.add_argument("--duration", type=int, default=0,
                   help="Auto-exit after N seconds (0 = run forever)")
    p.add_argument("--output", default=None,
                   help="Save annotated output to this .mp4 path")
    p.add_argument("--capture-fps", type=int, default=60,
                   help="Camera capture target FPS (default 60)")
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #

class SpatialPipeline:
    def __init__(self, args):
        self.args = args
        self._running = False

        # ---- Source ----
        src = args.source
        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        # ---- Threads ----
        self.capture = CameraCapture(source=src, target_fps=args.capture_fps)

        self.processor = ImageProcessor(
            sobel_sigma=args.sobel_sigma,
            track_color=args.track_color,
        )

        print("[Pipeline] Initialising YOLO engine…")
        engine = YOLOONNXEngine(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            input_size=args.input_size,
        )
        self.inference = InferenceThread(engine, target_classes=args.classes)

        # ---- Tracker & Renderer ----
        self.tracker = VelocityTracker()
        self.renderer = FrameRenderer(RenderConfig(
            show_edges=args.show_edges,
            show_trail=not args.no_trail,
            show_velocity_arrow=not args.no_arrows,
            show_hud=not args.no_hud,
        ))

        # ---- Profiler ----
        self.profiler = PipelineProfiler()
        if not args.profile:
            self.profiler.disable()

        # ---- Video writer ----
        self._writer = None
        self._writer_lock = threading.Lock()

        # ---- Stats ----
        self._render_fps_ts: list = []
        self._last_result = None   # most recent InferenceResult

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self):
        self._running = True

        # Start threads
        self.capture.start()
        self.processor.start()
        self.inference.start()

        print("[Pipeline] All threads started. Press Q to quit.")

        start_time = time.perf_counter()
        frame_count = 0
        last_proc_submit = -1
        last_infer_submit = -1
        tracks = []

        while self._running:
            loop_start = time.perf_counter()

            # ---- Duration limit ----
            if self.args.duration > 0:
                if time.perf_counter() - start_time >= self.args.duration:
                    print(f"[Pipeline] Duration {self.args.duration}s reached. Exiting.")
                    break

            # ---- 1. Grab latest captured frame ----
            packet = self.capture.buffer.get_latest()
            if packet is not None:
                # Feed to processor (non-blocking)
                self.processor.submit(packet)
                last_proc_submit = packet.frame_id

            # ---- 2. Grab latest processed frame ----
            processed = self.processor.get_latest()
            if processed is not None:
                # Feed to inference thread (non-blocking)
                self.inference.submit(processed)
                last_infer_submit = processed.source.frame_id
                self.profiler.record(
                    "process", processed.source.frame_id, processed.process_ms
                )

            # ---- 3. Grab latest inference result ----
            result = self.inference.get_latest()
            if result is not None:
                self._last_result = result
                self.profiler.record(
                    "infer", result.source.source.frame_id, result.infer_ms
                )
                # Update tracker
                now = time.perf_counter()
                tracks = self.tracker.update(result.detections, now)

            # ---- 4. Render ----
            if self._last_result is not None:
                frame = self._last_result.source.source.frame
                edges = self._last_result.source.edges
                color_mask = self._last_result.source.color_mask

                # Build HUD data
                render_fps = self._compute_render_fps()
                pipeline_ms = 0.0
                if self._last_result:
                    capture_ts = self._last_result.source.source.capture_ts
                    pipeline_ms = (time.perf_counter() - capture_ts) * 1000

                hud = {
                    'capture_fps': self.capture.fps,
                    'process_fps': self.processor.fps,
                    'infer_fps': self.inference.fps,
                    'render_fps': render_fps,
                    'infer_ms': self._last_result.infer_ms,
                    'pipeline_ms': pipeline_ms,
                    'dropped': self.capture.buffer.dropped_frames,
                    'tracks': len(tracks),
                }

                annotated = self.renderer.render(
                    frame, tracks,
                    edges=edges,
                    color_mask=color_mask,
                    hud_data=hud,
                )

                # ---- Init writer on first frame ----
                if self.args.output and self._writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(
                        self.args.output, fourcc, 30, (w, h)
                    )

                if self._writer:
                    self._writer.write(annotated)

                if not self.args.no_display:
                    cv2.imshow("Spatial Pipeline", annotated)

                frame_count += 1

            # ---- Key handling ----
            if not self.args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('e'):
                    self.renderer.cfg.show_edges = not self.renderer.cfg.show_edges
                elif key == ord('t'):
                    self.renderer.cfg.show_trail = not self.renderer.cfg.show_trail
                elif key == ord('v'):
                    self.renderer.cfg.show_velocity_arrow = not self.renderer.cfg.show_velocity_arrow
                elif key == ord('h'):
                    self.renderer.cfg.show_hud = not self.renderer.cfg.show_hud
            else:
                # Headless: small sleep to avoid spinning CPU
                time.sleep(0.001)

        self._shutdown(frame_count, time.perf_counter() - start_time)

    def _compute_render_fps(self) -> float:
        now = time.perf_counter()
        self._render_fps_ts.append(now)
        if len(self._render_fps_ts) > 60:
            self._render_fps_ts.pop(0)
        if len(self._render_fps_ts) < 2:
            return 0.0
        return (len(self._render_fps_ts) - 1) / (self._render_fps_ts[-1] - self._render_fps_ts[0])

    def _shutdown(self, frame_count: int, elapsed: float):
        print("\n[Pipeline] Shutting down…")
        self.capture.stop()
        self.processor.stop()
        self.inference.stop()
        self.capture.join(timeout=2)
        self.processor.join(timeout=2)
        self.inference.join(timeout=2)

        if self._writer:
            self._writer.release()
        if not self.args.no_display:
            cv2.destroyAllWindows()

        print(f"[Pipeline] Processed {frame_count} display frames in {elapsed:.1f}s "
              f"({frame_count/elapsed:.1f} FPS avg)")

        if self.args.profile:
            self.profiler.print_summary()
            self.profiler.export_csv("pipeline_profile.csv")

        if self.capture.error:
            print(f"[Pipeline] Capture error: {self.capture.error}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    pipeline = SpatialPipeline(args)

    # Handle Ctrl-C gracefully
    def _sigint(sig, frame):
        print("\n[Pipeline] Interrupted.")
        pipeline._running = False
    signal.signal(signal.SIGINT, _sigint)

    pipeline.run()


if __name__ == "__main__":
    main()
