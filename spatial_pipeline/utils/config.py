"""
spatial_pipeline/utils/config.py
---------------------------------
Typed, validated configuration with YAML/env-var override support.

Design choices (staff-level notes):
- Dataclasses over dicts: IDE completion, type errors at config-parse time not runtime.
- frozen=False on leaf configs so runtime tuning (e.g. conf threshold slider) works.
- No Pydantic dependency — keeps the package lean for edge deployment.
- Environment variable overrides follow 12-factor app conventions.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List


# ------------------------------------------------------------------ #
#  Sub-configs                                                         #
# ------------------------------------------------------------------ #

@dataclass
class CaptureConfig:
    source: int | str = 0            # camera index or video file path
    target_fps: int = 60             # cap capture rate (useful for files)
    buffer_slots: int = 3            # ring buffer depth


@dataclass
class ProcessorConfig:
    sobel_sigma: float = 1.0         # Gaussian pre-blur for edge detection
    track_color: Optional[str] = None  # 'red'|'green'|'blue'|'yellow'|'orange'|None


@dataclass
class InferenceConfig:
    model_path: Optional[str] = None  # auto-downloads yolov8n if None
    input_size: int = 640             # model resolution (320=fast, 640=accurate)
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    target_classes: Optional[List[str]] = None  # None = all COCO classes


@dataclass
class TrackerConfig:
    ema_alpha: float = 0.4           # velocity smoothing factor
    trail_fade_s: float = 2.0        # trail point lifetime (seconds)
    iou_threshold: float = 0.30      # detection→track matching threshold
    max_lost_frames: int = 10        # frames before track is pruned
    use_kalman: bool = True          # Kalman filter for occlusion handling
    trail_max_len: int = 60          # maximum trail points stored


@dataclass
class RenderConfig:
    show_edges: bool = False         # Sobel edge overlay
    edge_alpha: float = 0.35
    show_trail: bool = True
    trail_thickness: int = 2
    show_velocity_arrow: bool = True
    arrow_scale: float = 0.05
    show_hud: bool = True
    hud_alpha: float = 0.70
    box_thickness: int = 2
    font_scale: float = 0.55
    font_thickness: int = 1


@dataclass
class OutputConfig:
    display: bool = True             # show OpenCV window
    output_path: Optional[str] = None  # save annotated video here
    output_fps: int = 30
    duration_s: int = 0              # 0 = run forever


@dataclass
class ProfilerConfig:
    enabled: bool = False
    max_samples: int = 500
    csv_path: str = "pipeline_profile.csv"


# ------------------------------------------------------------------ #
#  Root config                                                         #
# ------------------------------------------------------------------ #

@dataclass
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    # ---------------------------------------------------------------- #
    #  Loaders                                                          #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a YAML file (requires pyyaml)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML configs")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineConfig":
        cfg = cls()
        for section_name, section_cls in [
            ("capture", CaptureConfig),
            ("processor", ProcessorConfig),
            ("inference", InferenceConfig),
            ("tracker", TrackerConfig),
            ("render", RenderConfig),
            ("output", OutputConfig),
            ("profiler", ProfilerConfig),
        ]:
            if section_name in data:
                current = getattr(cfg, section_name)
                for k, v in data[section_name].items():
                    if hasattr(current, k):
                        setattr(current, k, v)
        return cfg

    def apply_env_overrides(self) -> "PipelineConfig":
        """
        Override config fields from environment variables.
        Convention: SP_SECTION__FIELD=value
        e.g. SP_INFERENCE__CONF_THRESHOLD=0.55
        """
        prefix = "SP_"
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if "__" not in rest:
                continue
            section, field_name = rest.lower().split("__", 1)
            section_obj = getattr(self, section, None)
            if section_obj is None:
                continue
            if not hasattr(section_obj, field_name):
                continue
            current = getattr(section_obj, field_name)
            try:
                if isinstance(current, bool):
                    setattr(section_obj, field_name, val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(section_obj, field_name, int(val))
                elif isinstance(current, float):
                    setattr(section_obj, field_name, float(val))
                else:
                    setattr(section_obj, field_name, val)
            except (ValueError, TypeError):
                pass  # malformed env var — silently skip
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """Return a list of validation warnings (non-fatal)."""
        warnings: list[str] = []
        if self.inference.input_size not in (320, 416, 640):
            warnings.append(
                f"Unusual input_size={self.inference.input_size}; "
                "standard values are 320, 416, 640"
            )
        if self.tracker.ema_alpha <= 0 or self.tracker.ema_alpha > 1:
            warnings.append("tracker.ema_alpha should be in (0, 1]")
        if self.inference.conf_threshold > 0.9:
            warnings.append("conf_threshold >0.9 may miss most detections")
        if self.capture.target_fps > 120:
            warnings.append("target_fps >120 unlikely to be achieved on most hardware")
        return warnings
