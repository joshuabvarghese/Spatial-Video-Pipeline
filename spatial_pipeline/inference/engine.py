"""
spatial_pipeline/inference/engine.py
--------------------------------------
YOLOv8-nano ONNX Runtime inference engine + dedicated thread.

Execution Provider selection
-----------------------------
ONNX Runtime supports pluggable EPs.  We probe in priority order:
    CUDAExecutionProvider   — discrete GPU (fastest, ~2ms/frame)
    CoreMLExecutionProvider — Apple Neural Engine (M-series: ~3ms)
    CPUExecutionProvider    — OpenMP-parallelised fallback (~5-15ms)

The session is created once and reused for the lifetime of the thread.
Inference itself is GIL-free (ONNX Runtime releases the GIL during
OrtSession::Run), so the inference thread does not block Python threads.

Post-processing
---------------
YOLOv8 ONNX exports in [1, 84, 8400] format for 640-input models
(or [1, 84, 2100] for 320-input):
    Rows 0-3:   cx, cy, w, h  (model-input pixel space)
    Rows 4-83:  per-class confidence scores

We run greedy NMS in NumPy (no torchvision, no cv2.dnn.NMSBoxes) to
keep the inference stage self-contained.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

from spatial_pipeline.core.processor import ProcessedFrame
from spatial_pipeline.utils.config import InferenceConfig
from spatial_pipeline.utils.fps import FPSCounter


# ------------------------------------------------------------------ #
#  COCO class names                                                    #
# ------------------------------------------------------------------ #

COCO_NAMES: list[str] = [
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
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name} {self.confidence:.0%} "
            f"[{self.x1},{self.y1}→{self.x2},{self.y2}])"
        )


@dataclass
class InferenceResult:
    source: ProcessedFrame
    detections: List[Detection] = field(default_factory=list)
    infer_ms: float = 0.0
    infer_ts: float = 0.0


# ------------------------------------------------------------------ #
#  Engine                                                              #
# ------------------------------------------------------------------ #

class YOLOONNXEngine:
    """
    Stateless inference engine — thread-safe after __init__.
    Call infer() from any thread.
    """

    def __init__(self, cfg: Optional[InferenceConfig] = None) -> None:
        import onnxruntime as ort

        self._cfg = cfg or InferenceConfig()
        model_path = self._cfg.model_path or self._ensure_model()

        # Provider selection
        available = ort.get_available_providers()
        providers: list = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider",
                              {"cudnn_conv_algo_search": "DEFAULT"}))
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2

        self._sess = ort.InferenceSession(model_path, opts, providers=providers)
        self._input_name: str = self._sess.get_inputs()[0].name
        self.active_provider: str = self._sess.get_providers()[0]
        print(f"[YOLOEngine] model={Path(model_path).name}  "
              f"EP={self.active_provider}  "
              f"input={self._cfg.input_size}px")

    # ---------------------------------------------------------------- #
    #  Inference                                                        #
    # ---------------------------------------------------------------- #

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        """Thread-safe.  Returns detections in original image coordinates."""
        orig_h, orig_w = bgr.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(bgr, self._cfg.input_size)
        blob = (blob.astype(np.float32) / 255.0)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])   # NCHW

        raw = self._sess.run(None, {self._input_name: blob})[0]
        return self._postprocess(raw, orig_w, orig_h, scale, pad_x, pad_y)

    # ---------------------------------------------------------------- #
    #  Pre / post-processing                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _letterbox(img: np.ndarray, target: int
                   ) -> tuple[np.ndarray, float, int, int]:
        """Resize + centre-pad to (target × target), preserve aspect ratio."""
        import cv2
        h, w = img.shape[:2]
        scale = target / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        pad_x, pad_y = (target - nw) // 2, (target - nh) // 2
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        return canvas, scale, pad_x, pad_y

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int, orig_h: int,
        scale: float, pad_x: int, pad_y: int,
    ) -> List[Detection]:
        # raw: [1, 84, N] → squeeze → [84, N] → transpose → [N, 84]
        data = raw[0].T if raw.ndim == 3 else raw.T
        boxes  = data[:, :4]
        scores = data[:, 4:]
        class_ids  = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self._cfg.conf_threshold
        if not mask.any():
            return []

        boxes, confidences, class_ids = (
            boxes[mask], confidences[mask], class_ids[mask]
        )

        # cx,cy,w,h (model space) → x1,y1,x2,y2 (original image space)
        x1 = ((boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale).astype(int)
        y1 = ((boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale).astype(int)
        x2 = ((boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale).astype(int)
        y2 = ((boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale).astype(int)

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        keep = self._nms(x1, y1, x2, y2, confidences)
        results: List[Detection] = []
        for i in keep:
            cid = int(class_ids[i])
            name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else f"cls{cid}"
            results.append(Detection(
                class_id=cid, class_name=name,
                confidence=float(confidences[i]),
                x1=int(x1[i]), y1=int(y1[i]),
                x2=int(x2[i]), y2=int(y2[i]),
            ))
        return results

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_thresh: float = 0.45) -> list[int]:
        """Greedy Non-Maximum Suppression in pure NumPy."""
        order = scores.argsort()[::-1]
        areas = (x2 - x1) * (y2 - y1)
        keep: list[int] = []
        while len(order):
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou < iou_thresh]
        return keep

    def _ensure_model(self) -> str:
        """
        Download YOLOv8n ONNX directly — no torch, no ultralytics required at runtime.
        Cached at ~/.cache/spatial_pipeline/yolov8n_320.onnx after first run.
        """
        cache = Path.home() / ".cache" / "spatial_pipeline" / "yolov8n_320.onnx"
        if cache.exists():
            print(f"[YOLOEngine] Using cached model: {cache}")
            return str(cache)

        cache.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
        print(f"[YOLOEngine] Downloading YOLOv8n ONNX (~12 MB, one-time)…")

        import urllib.request, shutil
        tmp = cache.parent / "yolov8n_download.tmp"
        try:
            def _hook(count, block, total):
                if total > 0:
                    print(f"  {min(100, count*block*100//total)}%", end="", flush=True)
            urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
            print()
            shutil.move(str(tmp), str(cache))
            print(f"[YOLOEngine] Saved → {cache}")
        except Exception as e:
            if tmp.exists(): tmp.unlink()
            raise RuntimeError(f"Download failed: {e}. Get it manually from {url} and save to {cache}") from e
        return str(cache)


# ------------------------------------------------------------------ #
#  Inference Thread                                                    #
# ------------------------------------------------------------------ #

class InferenceThread(threading.Thread):
    """
    Wraps YOLOONNXEngine in a dedicated thread with single-slot I/O.

    Same latest-wins design as the ring buffer: if the render thread
    hasn't consumed the previous result, we overwrite it with a fresher
    detection rather than building a lag queue.
    """

    def __init__(
        self,
        engine: YOLOONNXEngine,
        target_classes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="YOLOInference", daemon=True)
        self._engine = engine
        self._filter: Optional[set[str]] = (
            set(target_classes) if target_classes else None
        )
        self._input:  Optional[ProcessedFrame] = None
        self._output: Optional[InferenceResult] = None
        self._lock_in  = threading.Lock()
        self._lock_out = threading.Lock()
        self._new_work = threading.Event()
        self._stop     = threading.Event()
        self._fps      = FPSCounter(window=30)
        self.last_infer_ms: float = 0.0

    @property
    def fps(self) -> float:
        return self._fps.fps

    def submit(self, processed: ProcessedFrame) -> None:
        with self._lock_in:
            self._input = processed
        self._new_work.set()

    def get_latest(self) -> Optional[InferenceResult]:
        with self._lock_out:
            out = self._output
            self._output = None
        return out

    def stop(self) -> None:
        self._stop.set()
        self._new_work.set()

    def run(self) -> None:
        while not self._stop.is_set():
            self._new_work.wait(timeout=0.05)
            self._new_work.clear()
            with self._lock_in:
                processed = self._input
                self._input = None
            if processed is None:
                continue

            t0 = time.perf_counter()
            try:
                detections = self._engine.infer(processed.source.frame)
            except Exception as exc:
                print(f"[InferenceThread] {exc}")
                detections = []
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.last_infer_ms = elapsed_ms

            if self._filter:
                detections = [d for d in detections
                              if d.class_name in self._filter]

            result = InferenceResult(
                source=processed,
                detections=detections,
                infer_ms=elapsed_ms,
                infer_ts=time.perf_counter(),
            )
            with self._lock_out:
                self._output = result
            self._fps.tick()
