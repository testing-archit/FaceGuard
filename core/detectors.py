"""Face detectors behind one interface: detect(frame, gray=None) -> list[Detection].

YuNetDetector is the primary detector, a small anchor-free CNN run through
OpenCV's DNN module. HaarDetector is the classical Viola-Jones baseline kept
for the comparison experiment (E1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from core.types import Box, Config, Detection

# OpenCV 5's new DNN engine warns that it ignores preferable targets on every
# model load. We never set a target (CPU only), so the warning is noise.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> list[Detection]: ...


class YuNetDetector:
    """YuNet CNN face detector via `cv2.FaceDetectorYN`.

    YuNet is a lightweight anchor-free detector with a multi-scale feature
    pyramid. It regresses a box, five landmarks and a confidence score per
    face, and runs its own preprocessing and non-max suppression internally.
    Learned features make it far more robust to pose and lighting than
    hand-crafted Haar features.
    """

    def __init__(self, cfg: Config) -> None:
        path = Path(cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found at {path}. "
                "Run `python scripts/download_model.py` to restore it."
            )
        self._model = cv2.FaceDetectorYN.create(
            model=str(path),
            config="",  # empty for ONNX models
            input_size=(320, 320),  # overwritten per frame in detect()
            score_threshold=cfg.conf_threshold,
            nms_threshold=cfg.nms_threshold,
            top_k=cfg.top_k,
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> list[Detection]:
        h, w = frame.shape[:2]
        # Mandatory: YuNet decodes boxes relative to the input size, so a stale
        # size silently returns coordinates for the wrong resolution.
        if self._input_size != (w, h):
            self._model.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._model.detect(frame)
        if faces is None:
            return []
        return [_row_to_detection(row) for row in faces]


def _row_to_detection(row: np.ndarray) -> Detection:
    """Unpack one YuNet row: x, y, w, h, 5 landmark (x, y) pairs, score."""
    return Detection(
        box=Box.from_xywh(*row[:4]),
        score=float(row[14]),
        landmarks=row[4:14].reshape(5, 2).copy(),
    )


def make_detector(cfg: Config) -> Detector:
    """Build the detector named by `cfg.detector`."""
    if cfg.detector == "yunet":
        return YuNetDetector(cfg)
    raise ValueError(f"Unknown detector {cfg.detector!r}")
