"""
inference.py — YOLOv8-nano Inference Thread (ONNX Runtime)

Runs object detection on processed frames in a dedicated thread so that
the camera capture and display threads are never blocked by AI inference.

Uses ONNX Runtime with automatic EP selection:
  - CUDA (if available) → GPU inference
  - CoreML (macOS)      → Apple Neural Engine
  - CPU (fallback)      → optimised with OpenMP
"""

import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from processor import ProcessedFrame


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int  # pixel coords in original frame

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class InferenceResult:
    source: ProcessedFrame
    detections: List[Detection] = field(default_factory=list)
    infer_ms: float = 0.0
    infer_ts: float = 0.0
    model_input_shape: tuple = ()


# ------------------------------------------------------------------ #
#  YOLO ONNX wrapper                                                   #
# ------------------------------------------------------------------ #

class YOLOONNXEngine:
    """
    Thin wrapper around onnxruntime.InferenceSession for YOLOv8.

    Handles:
      - Model download / caching via ultralytics export
      - Input pre-processing (letterbox, normalise, NCHW)
      - Output post-processing (xywh → xyxy, NMS)
    """

    COCO_NAMES = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
        "toothbrush"
    ]

    def __init__(self, model_path: Optional[str] = None,
                 conf_threshold: float = 0.40,
                 iou_threshold: float = 0.45,
                 input_size: int = 320):
        """
        Parameters
        ----------
        model_path : str | None
            Path to a YOLOv8 ONNX file. If None, downloads yolov8n and exports.
        conf_threshold : float
            Minimum detection confidence.
        iou_threshold : float
            NMS IoU threshold.
        input_size : int
            Model input resolution (square). 320 for speed, 640 for accuracy.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        import onnxruntime as ort

        # Resolve model path
        if model_path is None:
            model_path = self._ensure_model()

        # Provider priority: CUDA → CoreML → CPU
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append(('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}))
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._active_provider = self._session.get_providers()[0]

        print(f"[YOLOEngine] Loaded {model_path} | Provider: {self._active_provider}")

    def _ensure_model(self) -> str:
        """Download YOLOv8n and export to ONNX if not already cached."""
        cache = Path.home() / ".cache" / "spatial_pipeline" / "yolov8n_320.onnx"
        if cache.exists():
            return str(cache)

        print("[YOLOEngine] Exporting YOLOv8n → ONNX (one-time, ~30s)…")
        cache.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        m = YOLO("yolov8n.pt")
        m.export(format="onnx", imgsz=self.input_size, simplify=True,
                 half=False, dynamic=False)
        import shutil
        exported = Path("yolov8n.onnx")
        if exported.exists():
            shutil.move(str(exported), str(cache))
        return str(cache)

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        """Run inference on a BGR uint8 frame, return detections."""
        orig_h, orig_w = bgr.shape[:2]

        # --- Pre-process ---
        blob, scale, pad_x, pad_y = self._letterbox(bgr, self.input_size)
        blob = blob.astype(np.float32) / 255.0          # [0,1]
        blob = np.transpose(blob, (2, 0, 1))             # HWC → CHW
        blob = np.expand_dims(blob, 0)                   # → NCHW

        # --- ONNX forward pass ---
        outputs = self._session.run(None, {self._input_name: blob})

        # --- Post-process ---
        return self._postprocess(outputs[0], orig_w, orig_h,
                                 scale, pad_x, pad_y)

    def _letterbox(self, img: np.ndarray, target: int):
        """Resize + pad to square, preserving aspect ratio."""
        import cv2
        h, w = img.shape[:2]
        scale = target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def _postprocess(self, raw: np.ndarray, orig_w, orig_h,
                     scale, pad_x, pad_y) -> List[Detection]:
        """
        YOLOv8 ONNX output shape: [1, 84, N] (for COCO 80-class model)
        Each column: [cx, cy, w, h, cls0_conf, …, cls79_conf]
        """
        # Squeeze batch dim → [84, N], transpose → [N, 84]
        if raw.ndim == 3:
            raw = raw[0]
        raw = raw.T  # [N, 84]

        boxes = raw[:, :4]          # cx, cy, w, h (in model input space)
        class_scores = raw[:, 4:]   # [N, 80]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # xywh → xyxy in model space
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Map back to original image coords
        x1 = ((x1 - pad_x) / scale).astype(int)
        y1 = ((y1 - pad_y) / scale).astype(int)
        x2 = ((x2 - pad_x) / scale).astype(int)
        y2 = ((y2 - pad_y) / scale).astype(int)

        # Clip to frame
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        # NMS (vectorised)
        indices = self._nms(x1, y1, x2, y2, confidences, self.iou_threshold)

        results = []
        for i in indices:
            cid = int(class_ids[i])
            name = self.COCO_NAMES[cid] if cid < len(self.COCO_NAMES) else f"cls{cid}"
            results.append(Detection(
                class_id=cid,
                class_name=name,
                confidence=float(confidences[i]),
                x1=int(x1[i]), y1=int(y1[i]),
                x2=int(x2[i]), y2=int(y2[i]),
            ))
        return results

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_thresh) -> List[int]:
        """Greedy NMS."""
        order = scores.argsort()[::-1]
        areas = (x2 - x1) * (y2 - y1)
        keep = []
        while len(order):
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter_w = np.maximum(0, ix2 - ix1)
            inter_h = np.maximum(0, iy2 - iy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou < iou_thresh]
        return keep


# ------------------------------------------------------------------ #
#  Inference Thread                                                     #
# ------------------------------------------------------------------ #

class InferenceThread(threading.Thread):
    """
    Wraps YOLOONNXEngine in a dedicated thread.
    Accepts ProcessedFrame objects, emits InferenceResult objects.
    """

    def __init__(self, engine: YOLOONNXEngine, target_classes: Optional[List[str]] = None):
        super().__init__(name="YOLOInference", daemon=True)
        self._engine = engine
        self._target_classes = set(target_classes) if target_classes else None

        self._input: Optional[ProcessedFrame] = None
        self._output: Optional[InferenceResult] = None
        self._lock_in = threading.Lock()
        self._lock_out = threading.Lock()
        self._new_work = threading.Event()
        self._stop_event = threading.Event()

        self._fps_ts: list[float] = []
        self.fps: float = 0.0
        self.last_infer_ms: float = 0.0

    def submit(self, processed: ProcessedFrame) -> None:
        with self._lock_in:
            self._input = processed
        self._new_work.set()

    def get_latest(self) -> Optional[InferenceResult]:
        with self._lock_out:
            r = self._output
            self._output = None
        return r

    def stop(self):
        self._stop_event.set()
        self._new_work.set()

    def run(self):
        while not self._stop_event.is_set():
            self._new_work.wait(timeout=0.1)
            self._new_work.clear()

            with self._lock_in:
                processed = self._input
                self._input = None

            if processed is None:
                continue

            t0 = time.perf_counter()
            try:
                detections = self._engine.infer(processed.source.frame)
            except Exception as e:
                print(f"[InferenceThread] Error: {e}")
                detections = []

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.last_infer_ms = elapsed_ms

            # Filter to target classes if specified
            if self._target_classes:
                detections = [d for d in detections if d.class_name in self._target_classes]

            result = InferenceResult(
                source=processed,
                detections=detections,
                infer_ms=elapsed_ms,
                infer_ts=time.perf_counter(),
            )

            with self._lock_out:
                self._output = result

            # Track FPS
            now = time.perf_counter()
            self._fps_ts.append(now)
            if len(self._fps_ts) > 30:
                self._fps_ts.pop(0)
            if len(self._fps_ts) >= 2:
                self.fps = (len(self._fps_ts) - 1) / (self._fps_ts[-1] - self._fps_ts[0])
