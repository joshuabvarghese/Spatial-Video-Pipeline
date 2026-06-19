"""
spatial_pipeline/pipeline.py
------------------------------
Pipeline orchestrator.  Owns all threads and the display loop.

Thread topology (data flow left → right):

    CameraCapture ──ring_buf──► ImageProcessor ──slot──► InferenceThread
                                                               │
                  ◄────────────────────────────────────────────┘
                  (render thread pulls latest result each frame)

The render loop (main thread) targets 30 FPS independently of inference.
If inference is slower than 30 FPS, the render loop reuses the last
InferenceResult — objects are still displayed and tracked, just not
updated.  This is the "zero hitch" guarantee.

Public API
----------
    p = Pipeline(cfg)
    p.run()             # blocks until exit
    # or
    p.start_async()     # non-blocking; call p.stop() later
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import List, Optional

import cv2

from spatial_pipeline.core.capture import CameraCapture
from spatial_pipeline.core.processor import ImageProcessor
from spatial_pipeline.inference.engine import YOLOONNXEngine, InferenceThread, InferenceResult
from spatial_pipeline.tracking.tracker import VelocityTracker, TrackedObject
from spatial_pipeline.viz.renderer import FrameRenderer
from spatial_pipeline.utils.config import PipelineConfig
from spatial_pipeline.utils.fps import FPSCounter
from spatial_pipeline.utils.profiler import PipelineProfiler


class Pipeline:
    """
    Top-level orchestrator.  Owns all resources; safe to re-enter.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully-typed configuration object.  See utils/config.py.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self._cfg = cfg or PipelineConfig()

        warnings = self._cfg.validate()
        for w in warnings:
            print(f"[Pipeline] WARNING: {w}")

        # ---- Threads ----
        self.capture   = CameraCapture(self._cfg.capture)
        self.processor = ImageProcessor(self._cfg.processor)

        print("[Pipeline] Initialising YOLO inference engine…")
        engine = YOLOONNXEngine(self._cfg.inference)
        self.inference = InferenceThread(
            engine,
            target_classes=self._cfg.inference.target_classes,
        )

        # ---- Tracker & Renderer ----
        self.tracker  = VelocityTracker(self._cfg.tracker)
        self.renderer = FrameRenderer(self._cfg.render)

        # ---- Profiler ----
        self.profiler = PipelineProfiler(
            max_samples=self._cfg.profiler.max_samples,
            enabled=self._cfg.profiler.enabled,
        )

        # ---- Internal state ----
        self._running        = False
        self._stop_event     = threading.Event()
        self._render_fps     = FPSCounter(window=60)
        self._last_result:   Optional[InferenceResult] = None
        self._active_tracks: List[TrackedObject] = []
        self._writer         = None

    # ---------------------------------------------------------------- #
    #  Entry points                                                     #
    # ---------------------------------------------------------------- #

    def run(self) -> None:
        """Block until exit (Q key, duration reached, or stop() called)."""
        self._start_threads()
        try:
            self._main_loop()
        finally:
            self._shutdown()

    def start_async(self) -> None:
        """Start pipeline in background threads; returns immediately."""
        self._start_threads()
        t = threading.Thread(target=self._main_loop, name="PipelineLoop", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _start_threads(self) -> None:
        self._running = True
        self.capture.start()
        self.processor.start()
        self.inference.start()
        print("[Pipeline] All threads started.  Press Q or Esc to quit.")

    def _main_loop(self) -> None:
        cfg = self._cfg
        start_wall = time.perf_counter()
        frame_count = 0

        while not self._stop_event.is_set():

            # ---- Duration guard ----
            if cfg.output.duration_s > 0:
                if time.perf_counter() - start_wall >= cfg.output.duration_s:
                    print(f"[Pipeline] Duration {cfg.output.duration_s}s reached.")
                    break

            # ---- Stage 1: capture → processor ----
            packet = self.capture.buffer.get_latest()
            if packet is not None:
                self.processor.submit(packet)

            # ---- Stage 2: processor → inference ----
            processed = self.processor.get_latest()
            if processed is not None:
                self.inference.submit(processed)
                self.profiler.record(
                    "process",
                    processed.source.frame_id,
                    processed.process_ms,
                )

            # ---- Stage 3: consume latest inference result ----
            result = self.inference.get_latest()
            if result is not None:
                self._last_result = result
                self.profiler.record(
                    "infer",
                    result.source.source.frame_id,
                    result.infer_ms,
                )
                self._active_tracks = self.tracker.update(
                    result.detections, time.perf_counter()
                )

            # ---- Stage 4: render ----
            if self._last_result is not None:
                t_render = time.perf_counter()
                annotated = self.renderer.render(
                    self._last_result.source.source.frame,
                    self._active_tracks,
                    edges      = self._last_result.source.edges,
                    color_mask = self._last_result.source.color_mask,
                    hud_data   = self._build_hud(),
                )
                render_ms = (time.perf_counter() - t_render) * 1000
                self.profiler.record(
                    "render",
                    self._last_result.source.source.frame_id,
                    render_ms,
                )
                self._render_fps.tick()
                frame_count += 1

                # ---- Init video writer lazily ----
                if cfg.output.output_path and self._writer is None:
                    h, w = annotated.shape[:2]
                    self._writer = cv2.VideoWriter(
                        cfg.output.output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cfg.output.output_fps,
                        (w, h),
                    )

                if self._writer:
                    self._writer.write(annotated)

                if cfg.output.display:
                    cv2.imshow("Spatial Pipeline", annotated)
            else:
                # No result yet — small sleep to avoid spinning
                time.sleep(0.002)

            # ---- Key handling ----
            if cfg.output.display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                self._handle_key(key)

    def _build_hud(self) -> dict:
        pipeline_ms = 0.0
        if self._last_result:
            pipeline_ms = (
                time.perf_counter()
                - self._last_result.source.source.capture_ts
            ) * 1000
        return {
            "capture_fps": self.capture.fps,
            "process_fps": self.processor.fps,
            "infer_fps":   self.inference.fps,
            "render_fps":  self._render_fps.fps,
            "infer_ms":    self.inference.last_infer_ms,
            "pipeline_ms": pipeline_ms,
            "dropped":     self.capture.buffer.dropped_frames,
            "tracks":      len(self._active_tracks),
        }

    def _handle_key(self, key: int) -> None:
        """Live key-bindings for toggling render layers."""
        if key == ord("e"):
            self.renderer.cfg.show_edges = not self.renderer.cfg.show_edges
        elif key == ord("t"):
            self.renderer.cfg.show_trail = not self.renderer.cfg.show_trail
        elif key == ord("v"):
            self.renderer.cfg.show_velocity_arrow = not self.renderer.cfg.show_velocity_arrow
        elif key == ord("h"):
            self.renderer.cfg.show_hud = not self.renderer.cfg.show_hud

    def _shutdown(self) -> None:
        print("\n[Pipeline] Shutting down…")
        self.capture.stop()
        self.processor.stop()
        self.inference.stop()
        self.capture.join(timeout=2.0)
        self.processor.join(timeout=2.0)
        self.inference.join(timeout=2.0)

        if self._writer:
            self._writer.release()
        if self._cfg.output.display:
            cv2.destroyAllWindows()

        if self._render_fps.fps > 0:
            print(f"[Pipeline] Display held {self._render_fps.fps:.1f} FPS (rolling)")

        if self.capture.error:
            print(f"[Pipeline] Capture thread error: {self.capture.error}")

        if self._cfg.profiler.enabled:
            self.profiler.print_summary()
            self.profiler.export_csv(self._cfg.profiler.csv_path)
