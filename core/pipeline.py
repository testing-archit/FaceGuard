"""Orchestration: analyse (find faces) then render (redact them).

Images go straight through detect -> redact. Video support is added on top of
the same two steps.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.detectors import Detector, make_detector
from core.errors import FaceGuardError, media_kind
from core.redact import redact
from core.types import Box, Config, Detection
from core.video_io import VideoReader, VideoWriter, finalise, limit_warnings

Progress = Callable[[int, int], None]  # (frames done, total frames)


@dataclass
class RunStats:
    """What happened during one video run; feeds the evaluation tables."""

    frames: int = 0
    loop_seconds: float = 0.0  # decode + detect/track + redact + write
    total_seconds: float = 0.0  # including ffmpeg finalisation
    faces_redacted: int = 0  # sum over frames
    warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.loop_seconds if self.loop_seconds > 0 else 0.0


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


def process_video_per_frame(
    src: str | Path, dst: str | Path, cfg: Config, progress: Progress | None = None
) -> RunStats:
    """Baseline: run the detector on every frame, redact, write.

    No tracking, so any frame where the detector misses a face leaks it.
    This is the reference point the tracking pipeline is measured against.
    """
    stats = RunStats()
    detector = make_detector(cfg)
    start = time.perf_counter()
    with VideoReader(src) as reader:
        stats.warnings += limit_warnings(reader.info)
        total = reader.info.frame_count
        with VideoWriter(dst, reader.info) as writer:
            for i, frame in enumerate(reader):
                boxes = [d.box for d in detector.detect(frame)]
                writer.write(redact(frame, boxes, cfg))
                stats.frames += 1
                stats.faces_redacted += len(boxes)
                if progress:
                    progress(i + 1, total)
    stats.loop_seconds = time.perf_counter() - start
    if stats.frames == 0:
        raise FaceGuardError(f"No frames could be decoded from {src}.")
    stats.warnings += finalise(writer.path, dst, src if cfg.keep_audio else None)
    stats.total_seconds = time.perf_counter() - start
    return stats
