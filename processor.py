"""
processor.py — Custom Image Processing (No OpenCV Wrappers)

Implements from scratch using only NumPy array operations:

  1. BGR → HSV conversion  (ITU-R formula, vectorised)
  2. Sobel edge detection  (manual 3×3 kernel convolution)
  3. Gaussian pre-blur     (separable 1-D convolution)
  4. HSV color-range mask  (for blob tracking mode)

Why bother?  In an edge-deployed scenario the OpenCV wrappers may not
be available or you need explicit control over precision, SIMD layout,
or you want to fuse operations to avoid intermediate allocations.
"""

import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional
from capture import FramePacket


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #

@dataclass
class ProcessedFrame:
    source: FramePacket
    gray: np.ndarray           # uint8 luminance
    hsv: np.ndarray            # float32 H∈[0,360) S∈[0,1] V∈[0,1]
    edges: np.ndarray          # uint8 Sobel magnitude
    color_mask: Optional[np.ndarray] = None   # binary mask for blob mode
    process_ts: float = 0.0
    process_ms: float = 0.0    # wall time for this stage


# ------------------------------------------------------------------ #
#  From-scratch implementations                                        #
# ------------------------------------------------------------------ #

def bgr_to_hsv_numpy(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to HSV float32 WITHOUT cv2.cvtColor.

    Algorithm: standard min/max decomposition per ITU-R.
    Returns H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1].

    Vectorised over all pixels simultaneously — no Python loops.
    """
    # Normalise to [0, 1]
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0   # flip BGR→RGB
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # ------ Value ------
    V = Cmax

    # ------ Saturation ------
    S = np.where(Cmax > 1e-7, delta / Cmax, 0.0)

    # ------ Hue ------
    eps = 1e-7
    H = np.zeros_like(R)

    mask_r = (Cmax == R) & (delta > eps)
    mask_g = (Cmax == G) & (delta > eps)
    mask_b = (Cmax == B) & (delta > eps)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

    return np.stack([H, S, V], axis=-1)


def gaussian_blur_numpy(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Separable 1-D Gaussian convolution applied twice (H then V).
    No scipy, no cv2.GaussianBlur — raw NumPy stride tricks.
    """
    # Build 1-D kernel (radius = 3*sigma, always odd length)
    radius = max(1, int(3 * sigma))
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    img = gray.astype(np.float32)

    # Horizontal pass
    img = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=img
    )
    # Vertical pass
    img = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode='same'), axis=0, arr=img
    )
    return img


def _convolve2d_3x3(img: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 kernel to a 2-D float32 image using explicit stride tricks.
    Much faster than scipy.signal.convolve2d for fixed 3×3 kernels because
    we unroll the 9 multiplies manually and let NumPy broadcast them.
    """
    h, w = img.shape
    # Pad with reflect to avoid border artefacts
    p = np.pad(img, 1, mode='reflect')

    # Build 9 shifted views — no Python loop over pixels
    out = (
        p[0:h,   0:w]   * kx[0, 0] +
        p[0:h,   1:w+1] * kx[0, 1] +
        p[0:h,   2:w+2] * kx[0, 2] +
        p[1:h+1, 0:w]   * kx[1, 0] +
        p[1:h+1, 1:w+1] * kx[1, 1] +
        p[1:h+1, 2:w+2] * kx[1, 2] +
        p[2:h+2, 0:w]   * kx[2, 0] +
        p[2:h+2, 1:w+1] * kx[2, 1] +
        p[2:h+2, 2:w+2] * kx[2, 2]
    )
    return out


# Sobel kernels (standard)
_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float32)


def sobel_edges_numpy(gray: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    """
    Compute Sobel edge magnitude WITHOUT cv2.Sobel.

    Steps:
      1. Optional Gaussian pre-blur (reduces noise sensitivity)
      2. Apply Gx and Gy kernels via manual 3×3 convolution
      3. Magnitude = sqrt(Gx² + Gy²), clipped to [0, 255] uint8
    """
    blurred = gaussian_blur_numpy(gray.astype(np.float32), sigma=blur_sigma)

    Gx = _convolve2d_3x3(blurred, _SOBEL_X)
    Gy = _convolve2d_3x3(blurred, _SOBEL_Y)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    # Normalise to 0-255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def hsv_color_mask(hsv: np.ndarray,
                   h_low: float, h_high: float,
                   s_low: float = 0.4, s_high: float = 1.0,
                   v_low: float = 0.3, v_high: float = 1.0) -> np.ndarray:
    """
    Return a binary mask for pixels whose HSV values fall in the given range.
    Handles hue wrap-around (e.g. red: h_low=340, h_high=20).
    """
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    s_mask = (S >= s_low) & (S <= s_high)
    v_mask = (V >= v_low) & (V <= v_high)

    if h_low <= h_high:
        h_mask = (H >= h_low) & (H <= h_high)
    else:
        # Wrap-around: e.g. red straddles 0°
        h_mask = (H >= h_low) | (H <= h_high)

    return (h_mask & s_mask & v_mask).astype(np.uint8) * 255


# ------------------------------------------------------------------ #
#  Processing Thread                                                   #
# ------------------------------------------------------------------ #

class ImageProcessor(threading.Thread):
    """
    Consumes raw FramePackets, applies custom HSV + Sobel processing,
    and places ProcessedFrames into an output slot for the inference thread.

    Parameters
    ----------
    sobel_sigma : float
        Gaussian pre-blur sigma for Sobel edge detection.
    track_color : str | None
        One of 'red', 'green', 'blue', 'yellow', 'orange', None.
        If set, also computes an HSV color mask for blob tracking.
    """

    # Predefined HSV ranges (H in degrees, S/V in 0-1)
    _COLOR_RANGES = {
        'red':    (340, 20,  0.5, 1.0, 0.3, 1.0),
        'orange': (15,  40,  0.6, 1.0, 0.4, 1.0),
        'yellow': (40,  70,  0.5, 1.0, 0.4, 1.0),
        'green':  (80,  160, 0.4, 1.0, 0.3, 1.0),
        'blue':   (190, 260, 0.4, 1.0, 0.2, 1.0),
    }

    def __init__(self, sobel_sigma: float = 1.0, track_color: Optional[str] = None):
        super().__init__(name="ImageProcessor", daemon=True)
        self.sobel_sigma = sobel_sigma
        self.track_color = track_color

        self._input_packet: Optional[FramePacket] = None
        self._output: Optional[ProcessedFrame] = None
        self._lock_in = threading.Lock()
        self._lock_out = threading.Lock()
        self._new_frame = threading.Event()
        self._stop_event = threading.Event()

        self._fps_tracker = _LocalFPS(window=30)
        self.fps: float = 0.0

    # ------------------------------------------------------------------ #
    #  Feed / consume                                                      #
    # ------------------------------------------------------------------ #

    def submit(self, packet: FramePacket) -> None:
        with self._lock_in:
            self._input_packet = packet
        self._new_frame.set()

    def get_latest(self) -> Optional[ProcessedFrame]:
        with self._lock_out:
            result = self._output
            self._output = None
        return result

    def stop(self):
        self._stop_event.set()
        self._new_frame.set()

    # ------------------------------------------------------------------ #
    #  Thread body                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        while not self._stop_event.is_set():
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()

            with self._lock_in:
                packet = self._input_packet
                self._input_packet = None

            if packet is None:
                continue

            t0 = time.perf_counter()
            result = self._process(packet)
            result.process_ms = (time.perf_counter() - t0) * 1000
            result.process_ts = time.perf_counter()

            with self._lock_out:
                self._output = result

            self._fps_tracker.tick()
            self.fps = self._fps_tracker.fps

    def _process(self, packet: FramePacket) -> ProcessedFrame:
        frame = packet.frame

        # Luminance (fast: weighted sum without cv2.cvtColor)
        # BT.601 coefficients: Y = 0.114B + 0.587G + 0.299R
        B = frame[..., 0].astype(np.float32)
        G = frame[..., 1].astype(np.float32)
        R = frame[..., 2].astype(np.float32)
        gray = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)

        # Custom HSV
        hsv = bgr_to_hsv_numpy(frame)

        # Custom Sobel edges
        edges = sobel_edges_numpy(gray, blur_sigma=self.sobel_sigma)

        # Optional color mask
        color_mask = None
        if self.track_color and self.track_color in self._COLOR_RANGES:
            params = self._COLOR_RANGES[self.track_color]
            color_mask = hsv_color_mask(hsv, *params)

        return ProcessedFrame(
            source=packet,
            gray=gray,
            hsv=hsv,
            edges=edges,
            color_mask=color_mask,
        )


class _LocalFPS:
    def __init__(self, window=30):
        self._ts = []
        self._window = window

    def tick(self):
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) > self._window:
            self._ts.pop(0)

    @property
    def fps(self):
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])
