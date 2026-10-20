"""Orchestration: analyse (find faces) then render (redact them).

Images go straight through detect -> redact. Video support is added on top of
the same two steps.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.detectors import Detector, make_detector
from core.errors import FaceGuardError, media_kind
from core.redact import redact
from core.types import Box, Config, Detection


def read_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR, raising FaceGuardError instead of returning None."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceGuardError(f"Could not read image {path}. Is the file corrupted?")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img):
        raise FaceGuardError(f"Could not write {path}. Check the folder exists and is writable.")


def analyse_image(img: np.ndarray, cfg: Config, detector: Detector | None = None) -> list[Detection]:
    """Detect faces in a single image."""
    detector = detector or make_detector(cfg)
    return detector.detect(img)


def render_image(img: np.ndarray, boxes: list[Box], cfg: Config) -> np.ndarray:
    """Redact the given boxes. Output keeps the original resolution."""
    return redact(img, boxes, cfg)


def check_paths(src: str | Path, dst: str | Path) -> str:
    """Validate an input/output pair and return the media kind.

    Never let the output overwrite the source (FR-21), and keep images as
    images and videos as videos.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FaceGuardError(f"Input file not found: {src}")
    kind = media_kind(src)
    if media_kind(dst) != kind:
        raise FaceGuardError(f"Output must also be a {kind} file, got {dst.suffix!r}")
    if dst.resolve() == src.resolve():
        raise FaceGuardError("Output path is the same as the input. FaceGuard never modifies the source file.")
    return kind
