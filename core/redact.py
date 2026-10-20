"""Face redaction: Gaussian blur, pixelation or solid fill under a soft elliptical mask.

Pure functions over arrays. `redact` never modifies its input frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.types import Box, Config


def odd(n: float) -> int:
    """Round up to the nearest odd integer >= 1 (Gaussian kernels must be odd)."""
    k = max(1, math.ceil(n))
    return k if k % 2 == 1 else k + 1


def feather_px(face: Box, cfg: Config) -> int:
    """Width of the soft mask edge in pixels, scaled with face size."""
    return max(1, round(face.w * cfg.feather_frac))


def build_mask(face: Box, frame_w: int, frame_h: int, cfg: Config) -> tuple[Box, np.ndarray]:
    """Soft elliptical mask covering a face.

    Returns the region of interest (clipped to the frame) and a float32 alpha
    map of the same size. The ellipse is inscribed in the face box padded by
    `cfg.padding` per side; with padding >= (sqrt(2) - 1) / 2 it contains the
    face box's corners.

    The ellipse is drawn from the *unclipped* padded geometry and only the ROI
    is clipped, so a face at the frame border stays fully covered.

    Feathering goes outward: blurring a hard edge gives alpha 0.5 exactly on
    it, so `clip(2 * blurred, 0, 1)` is fully opaque everywhere inside the
    ellipse and the soft ramp lies entirely outside.
    """
    f = feather_px(face, cfg)
    cx, cy = face.center
    ax = face.w * (0.5 + cfg.padding)
    ay = face.h * (0.5 + cfg.padding)

    x0 = math.floor(cx - ax - f)
    y0 = math.floor(cy - ay - f)
    x1 = math.ceil(cx + ax + f)
    y1 = math.ceil(cy + ay + f)
    roi = Box(x0, y0, x1 - x0, y1 - y0).clip(frame_w, frame_h)
    if roi.area == 0:
        return roi, np.zeros((0, 0), np.float32)

    hard = np.zeros((roi.h, roi.w), np.float32)
    cv2.ellipse(
        hard,
        center=(round(cx - roi.x), round(cy - roi.y)),
        axes=(math.ceil(ax), math.ceil(ay)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )
    # sigma = f / 3 puts the ramp's tail (3 sigma) at the edge of the ROI.
    soft = cv2.GaussianBlur(hard, (2 * f + 1, 2 * f + 1), sigmaX=f / 3.0)
    alpha = np.clip(2.0 * soft, 0.0, 1.0)
    return roi, alpha


def blur_kernel(face: Box, cfg: Config) -> int:
    """Gaussian kernel size scaled with face width.

    A fixed kernel under-blurs large faces and obliterates small ones.
    """
    return odd(max(cfg.blur_min_kernel, face.w * cfg.blur_strength))


def _blur(roi: np.ndarray, face: Box, cfg: Config) -> np.ndarray:
    k = blur_kernel(face, cfg)
    return cv2.GaussianBlur(roi, (k, k), 0)


def _pixelate(roi: np.ndarray, face: Box, cfg: Config) -> np.ndarray:
    """Average into blocks, then upscale with nearest-neighbour.

    Block size is chosen so the padded face spans `pixel_blocks` blocks across,
    which scales the effect with face size. INTER_AREA gives a true block
    average; INTER_LINEAR would sample only a few pixels per block.
    """
    block = max(1.0, face.w * (1 + 2 * cfg.padding) / cfg.pixel_blocks)
    h, w = roi.shape[:2]
    small_w = max(1, round(w / block))
    small_h = max(1, round(h / block))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _fill(roi: np.ndarray, face: Box, cfg: Config) -> np.ndarray:
    return np.full_like(roi, cfg.fill_color)


_MODES = {"blur": _blur, "pixelate": _pixelate, "fill": _fill}


def redact(frame: np.ndarray, faces: list[Box], cfg: Config) -> np.ndarray:
    """Return a copy of `frame` with every face in `faces` redacted.

    Each face is processed independently: build its soft mask, apply the
    redaction mode to the ROI, alpha-composite over the frame.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    apply_mode = _MODES[cfg.mode]
    for face in faces:
        if face.area == 0:
            continue
        roi_box, alpha = build_mask(face, w, h, cfg)
        if roi_box.area == 0:
            continue
        ys = slice(roi_box.y, roi_box.y + roi_box.h)
        xs = slice(roi_box.x, roi_box.x + roi_box.w)
        roi = out[ys, xs]
        redacted = apply_mode(roi, face, cfg)
        a = alpha[..., None]
        out[ys, xs] = (a * redacted + (1.0 - a) * roi).round().astype(np.uint8)
    return out
