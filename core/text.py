"""Text redaction: number plates, name badges, street signs, screens.

Faces are not the only thing that identifies people. FaceGuard can also
hide visible text with OpenCV Zoo's PP-OCRv3 text *detector* (Apache 2.0,
2.4 MB, committed in `models/`). It finds where text is, it never reads it:
there is no recognition step, so no text content is extracted or stored.

Concept: DB (Differentiable Binarization) text detection. A small CNN
predicts, per pixel, the probability of being inside a text region and an
adaptive threshold map; binarising one with the other separates touching
words. Each connected region becomes a polygon, grown ("unclipped") by
`text_unclip_ratio` because the network is trained on shrunk text kernels.
FaceGuard takes each polygon's bounding rectangle.

Text regions are redacted with the configured effect but always with a
rectangular mask (an ellipse would leave the corners of a plate readable)
and their own padding. In video, text is detected on the scheduled
detection frames only, matched to the previous frame's text boxes by IoU,
and each box is held until the next detection (D-034).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from core.association import greedy_match
from core.redact import redact
from core.types import Box, Config, Track

TEXT_SOURCE = "text"
DB_MEAN = (123.675, 116.28, 103.53)  # ImageNet mean/std the model was trained with (RGB order in
DB_STD = (0.229, 0.224, 0.225)  # PaddleOCR; OpenCV Zoo's demo feeds BGR with these values as-is)
STRIDE = 32  # the network downsamples 32x: input sides must be multiples of it


class TextDetector:
    """PP-OCRv3 DB text detector through `cv2.dnn_TextDetectionModel_DB`."""

    def __init__(self, cfg: Config) -> None:
        path = Path(cfg.text_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Text detection model not found at {path}. "
                                    "Run `python scripts/download_model.py` to restore it.")
        self._m = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(path)))
        self._m.setBinaryThreshold(cfg.text_binary_threshold)
        self._m.setPolygonThreshold(cfg.text_polygon_threshold)
        self._m.setUnclipRatio(cfg.text_unclip_ratio)
        self._m.setMaxCandidates(cfg.text_max_candidates)
        self._m.setInputMean(DB_MEAN)
        self._m.setInputScale(1.0 / 255.0 / np.array(DB_STD))
        self._cfg = cfg
        self._size: tuple[int, int] | None = None

    def input_size(self, w: int, h: int) -> tuple[int, int]:
        """Frame size scaled so the long side is at most `text_max_side`, rounded to the stride."""
        s = min(1.0, self._cfg.text_max_side / max(w, h))
        return (max(STRIDE, round(w * s / STRIDE) * STRIDE), max(STRIDE, round(h * s / STRIDE) * STRIDE))

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        size = self.input_size(w, h)
        if size != self._size:
            self._m.setInputSize(size)
            self._size = size
        polys, _ = self._m.detect(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        sx, sy = w / size[0], h / size[1]
        boxes = []
        for p in polys:
            x, y, bw, bh = cv2.boundingRect(np.asarray(p, np.int32))
            b = Box.from_xywh(x * sx, y * sy, bw * sx, bh * sy).clip(w, h)
            if b.h >= self._cfg.text_min_px and b.area > 0:
                boxes.append(b)
        return boxes


def text_cfg(cfg: Config) -> Config:
    """Rendering config for text: rectangular mask, text padding, same effect."""
    return replace(cfg, mask_shape="rect", padding=cfg.text_padding)


def redact_text(frame: np.ndarray, boxes: list[Box], cfg: Config) -> np.ndarray:
    return redact(frame, boxes, text_cfg(cfg)) if boxes else frame


class TextTracks:
    """Text boxes in a video: detected on detection frames, held in between.

    A text box is matched to the previous detection's boxes by IoU (greedy),
    so a plate that stays in view keeps one ID the user can toggle. A box
    that is new on this detection is also stamped back over the buffered
    frames since the previous detection (the same lookbehind as face
    backfill, D-026), so text entering between detections is not exposed.
    """

    def __init__(self, cfg: Config, allocate_id: Callable[[], int]) -> None:
        self._cfg, self._alloc = cfg, allocate_id
        self._det = TextDetector(cfg)
        self.live: list[Track] = []
        self.finished: list[Track] = []
        self._last_detect: int | None = None

    def step(self, i: int, frame: np.ndarray, detect: bool, past_indices: list[int]) -> None:
        if detect:
            boxes = self._det.detect(frame)
            pairs, unmatched_t, unmatched_d = greedy_match([t.box for t in self.live], boxes, self._cfg.text_iou)
            keep = []
            for ti, di in pairs:
                self.live[ti].box = boxes[di]
                keep.append(self.live[ti])
            self.finished += [self.live[ti] for ti in unmatched_t]
            for di in unmatched_d:
                t = Track(id=self._alloc(), box=boxes[di], source=TEXT_SOURCE, last_seen_frame=i)
                floor = self._last_detect if self._last_detect is not None else -1
                for j in past_indices:
                    if j > floor:
                        t.history[j] = boxes[di]
                keep.append(t)
            self.live = keep
            self._last_detect = i
        for t in self.live:
            t.history[i] = t.box

    def end_all(self) -> None:
        self.finished += self.live
        self.live = []

    def all_tracks(self) -> list[Track]:
        return self.live + self.finished
