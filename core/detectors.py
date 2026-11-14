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


class HaarDetector:
    """Viola-Jones Haar cascade, the classical baseline.

    Haar-like rectangle features are evaluated quickly on an integral image
    and combined by a boosted cascade that rejects most windows in the first
    stages, scanned over an image pyramid. The features were learned on
    frontal faces, so pose and lighting changes break it. Histogram
    equalisation first, as in OpenCV's reference usage, to give it a fair
    chance under uneven lighting.

    The cascade gives no confidence score and no landmarks; score is
    synthesised as 1.0 to satisfy the Detection type.
    """

    def __init__(self, cfg: Config) -> None:
        path = Path(cfg.haar_path)
        self._cascade = cv2.CascadeClassifier(str(path)) if path.is_file() else None
        if self._cascade is None or self._cascade.empty():
            raise FileNotFoundError(
                f"Haar cascade not found at {path}. "
                "Run `python scripts/download_model.py` to restore it."
            )
        self._scale_factor = cfg.haar_scale_factor
        self._min_neighbors = cfg.haar_min_neighbors
        self._min_size = (cfg.min_face_px, cfg.min_face_px)

    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> list[Detection]:
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        return [Detection(box=Box(*map(int, f)), score=1.0) for f in faces]


def tile_grid(w: int, h: int, n: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """n x n tiles covering a w x h frame, each grown by `overlap` of its size.

    Neighbouring tiles overlap, so a face cut in half by one tile boundary is
    whole in the next tile. Returns (x, y, w, h) per tile, clipped to the frame.
    """
    tw, th = w / n, h / n
    ox, oy = tw * overlap / 2, th * overlap / 2
    tiles = []
    for r in range(n):
        for c in range(n):
            x0, y0 = max(0, int(c * tw - ox)), max(0, int(r * th - oy))
            x1, y1 = min(w, int((c + 1) * tw + ox)), min(h, int((r + 1) * th + oy))
            tiles.append((x0, y0, x1 - x0, y1 - y0))
    return tiles


def nms(dets: list[Detection], iou: float) -> list[Detection]:
    """Greedy non-maximum suppression: keep the highest score, drop its overlaps."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(d.box.iou(k.box) <= iou for k in keep):
            keep.append(d)
    return keep


class TiledDetector:
    """Runs a detector on the full frame and on upscaled overlapping tiles.

    Detectors have a smallest face they can see: YuNet's finest feature map
    has a stride of 8 px, so faces of ~10-15 px are below its reliable range.
    Upscaling a tile by `tile_scale` makes those faces proportionally larger.
    Detections from all passes are mapped back to frame coordinates and
    merged by non-maximum suppression. Cost grows roughly with
    1 + tile_scale^2 * (1 + overlap)^2, so this is meant for photos and for
    crowds, not as a video default.
    """

    def __init__(self, base: Detector, cfg: Config) -> None:
        self._base = base
        self._cfg = cfg

    def detect(self, frame: np.ndarray, gray: np.ndarray | None = None) -> list[Detection]:
        cfg = self._cfg
        dets = list(self._base.detect(frame, gray))
        h, w = frame.shape[:2]
        s = cfg.tile_scale
        for x, y, tw, th in tile_grid(w, h, cfg.tile_grid, cfg.tile_overlap):
            tile = cv2.resize(frame[y : y + th, x : x + tw], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for d in self._base.detect(tile):
                b = d.box
                lm = None if d.landmarks is None else d.landmarks / s + np.array([x, y])
                dets.append(Detection(Box.from_xywh(x + b.x / s, y + b.y / s, b.w / s, b.h / s), d.score, lm))
        return nms(dets, cfg.tile_nms_iou)


def make_detector(cfg: Config) -> Detector:
    """Build the detector named by `cfg.detector`, tiled if `cfg.tile_grid` > 0."""
    if cfg.detector == "yunet":
        base: Detector = YuNetDetector(cfg)
    elif cfg.detector == "haar":
        base = HaarDetector(cfg)
    else:
        raise ValueError(f"Unknown detector {cfg.detector!r}")
    return TiledDetector(base, cfg) if cfg.tile_grid > 0 else base
