"""
scripts/run_pipeline.py
------------------------
CLI entry point for the spatial pipeline.

Examples
--------
  # Webcam, default settings
  python scripts/run_pipeline.py --source 0

  # Video file + Sobel edges + profiler
  python scripts/run_pipeline.py --source video.mp4 --show-edges --profile

  # Track only people at high confidence, save output
  python scripts/run_pipeline.py --source 0 --classes person --conf 0.55 --output out.mp4

  # Headless 30-second benchmark
  python scripts/run_pipeline.py --source 0 --no-display --duration 30 --profile

  # Track red blobs (color mode)
  python scripts/run_pipeline.py --source 0 --track-color red

  # Load config from JSON
  python scripts/run_pipeline.py --config configs/default.json
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_pipeline.pipeline import Pipeline
from spatial_pipeline.utils.config import (
    PipelineConfig,
    CaptureConfig, ProcessorConfig, InferenceConfig,
    TrackerConfig, RenderConfig, OutputConfig, ProfilerConfig,
)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spatial_pipeline",
        description="Edge-Deployed Real-Time Spatial Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file (overrides defaults before CLI args are applied)
    p.add_argument("--config", default=None,
                   help="Path to JSON config file")

    # Source
    p.add_argument("--source", default="0",
                   help="Camera index (int) or video file path (default: 0)")
    p.add_argument("--capture-fps", type=int, default=60,
                   help="Camera capture target FPS (default: 60)")

    # Inference
    p.add_argument("--classes", nargs="+", default=None,
                   help="Filter to specific YOLO class names, e.g. person cat")
    p.add_argument("--conf", type=float, default=0.40,
                   help="Detection confidence threshold (default: 0.40)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--input-size", type=int, default=640, choices=[320, 416, 640],
                   help="YOLO model input resolution (default: 320)")

    # Processing
    p.add_argument("--track-color", default=None,
                   choices=["red","orange","yellow","green","blue"],
                   help="Enable HSV color-blob tracking for this color")
    p.add_argument("--sobel-sigma", type=float, default=1.0,
                   help="Gaussian pre-blur sigma for Sobel (default: 1.0)")

    # Render toggles
    p.add_argument("--show-edges", action="store_true",
                   help="Overlay Sobel edges on frame")
    p.add_argument("--no-trail", action="store_true",
                   help="Disable trailing path")
    p.add_argument("--no-arrows", action="store_true",
                   help="Disable velocity arrows")
    p.add_argument("--no-hud", action="store_true",
                   help="Disable HUD overlay")

    # Output
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode (no OpenCV window)")
    p.add_argument("--output", default=None,
                   help="Save annotated video to this .mp4 path")
    p.add_argument("--duration", type=int, default=0,
                   help="Auto-exit after N seconds (0 = run forever)")

    # Profiler
    p.add_argument("--profile", action="store_true",
                   help="Enable profiler; prints summary and exports CSV on exit")

    return p


def cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build PipelineConfig from parsed CLI args."""
    # Start from JSON/YAML file if provided
    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig()

    # Apply CLI overrides (CLI wins over config file)
    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass
    cfg.capture.source     = src
    cfg.capture.target_fps = args.capture_fps

    cfg.processor.sobel_sigma  = args.sobel_sigma
    cfg.processor.track_color  = args.track_color

    cfg.inference.conf_threshold  = args.conf
    cfg.inference.iou_threshold   = args.iou
    cfg.inference.input_size      = args.input_size
    cfg.inference.target_classes  = args.classes

    cfg.render.show_edges          = args.show_edges
    cfg.render.show_trail          = not args.no_trail
    cfg.render.show_velocity_arrow = not args.no_arrows
    cfg.render.show_hud            = not args.no_hud

    cfg.output.display     = not args.no_display
    cfg.output.output_path = args.output
    cfg.output.duration_s  = args.duration

    cfg.profiler.enabled = args.profile

    # Environment variable overrides (12-factor)
    cfg.apply_env_overrides()

    return cfg


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = cfg_from_args(args)

    pipeline = Pipeline(cfg)

    def _sigint(_sig, _frame):
        print("\n[CLI] Interrupted — shutting down…")
        pipeline.stop()

    signal.signal(signal.SIGINT, _sigint)
    pipeline.run()


if __name__ == "__main__":
    main()
