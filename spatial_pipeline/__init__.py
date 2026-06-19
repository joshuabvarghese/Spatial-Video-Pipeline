"""
spatial_pipeline
================
Edge-deployed real-time spatial video pipeline.

Multithreaded architecture:
  CameraCapture -> ImageProcessor -> YOLOInferenceEngine -> RenderCompositor

Designed for zero-lag playback at >=30 FPS under concurrent AI inference load.
"""

__version__ = "1.0.0"
__author__ = "Spatial Pipeline"
