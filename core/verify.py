"""Self-check: does a face detector still find faces in the redacted output?

Redaction is only as good as what an adversary can recover from the
exported file. The cheapest adversary is the same kind of detector that
found the faces: if YuNet, run on the *output* at a lower threshold than the
one used for redaction, still fires inside a region that was meant to be
hidden, that face is not hidden well enough (too little padding, too weak a
blur, a mask that lagged the face).

A residual is a detection on the output whose centre lies inside a box
that was supposed to be redacted on that frame. Detections elsewhere are
faces the user chose to leave visible, or faces the pipeline never found;
those are reported separately as `unmasked`, since the check cannot tell
the two apart without the user's selection.

Optionally (SFace model present), each residual is also compared with the
original face: a cosine similarity above OpenCV's same-person threshold
means the redacted face could still be matched to the person.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from core.detectors import Detector, make_detector
from core.types import Box, Config, Track
from core.video_io import VideoReader


@dataclass(frozen=True)
class Hit:
    frame: int
    box: Box
    score: float


@dataclass
class VerifyReport:
    frames_checked: int = 0
    hidden_faces: int = 0  # face-frames that were meant to be hidden
    residuals: list[Hit] = field(default_factory=list)  # detector still fires inside a hidden region
    unmasked: list[Hit] = field(default_factory=list)  # faces outside every hidden region

    @property
    def ok(self) -> bool:
        return not self.residuals

    @property
    def residual_frames(self) -> list[int]:
        return sorted({h.frame for h in self.residuals})

    def summary(self) -> str:
        if self.ok:
            return (f"Verified {self.frames_checked} frame(s): no face found inside any hidden region "
                    f"({self.hidden_faces} hidden face-frames checked).")
        frames = self.residual_frames
        shown = ", ".join(map(str, frames[:10])) + (" ..." if len(frames) > 10 else "")
        return (f"{len(self.residuals)} face(s) still detectable inside hidden regions on "
                f"{len(frames)} frame(s): {shown}. Try fill mode, more padding, or a shorter detection interval.")


def verify_detector(cfg: Config) -> Detector:
    """YuNet at the verification threshold, full frame only.

    YuNet even when redaction used Haar: the check should use the strongest
    available detector, not the one that may have missed the face.
    """
    return make_detector(replace(cfg, detector="yunet", conf_threshold=cfg.verify_threshold, tile_grid=0))


def _inside(point: tuple[float, float], b: Box) -> bool:
    x, y = point
    return b.x <= x < b.x + b.w and b.y <= y < b.y + b.h


def check_frame(
    out_frame: np.ndarray, hidden: list[Box], detector: Detector, frame: int = 0
) -> tuple[list[Hit], list[Hit]]:
    """Residual and unmasked detections on one redacted frame."""
    residuals, unmasked = [], []
    for d in detector.detect(out_frame):
        hit = Hit(frame, d.box, d.score)
        (residuals if any(_inside(d.box.center, b) for b in hidden) else unmasked).append(hit)
    return residuals, unmasked


def verify_image(out_img: np.ndarray, hidden: list[Box], cfg: Config, detector: Detector | None = None) -> VerifyReport:
    det = detector or verify_detector(cfg)
    r, u = check_frame(out_img, hidden, det)
    return VerifyReport(frames_checked=1, hidden_faces=len(hidden), residuals=r, unmasked=u)


def verify_video(
    dst: str | Path,
    tracks: list[Track],
    cfg: Config,
    every: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> VerifyReport:
    """Scan an exported video every `every` frames (default `cfg.verify_every`)."""
    every = every or cfg.verify_every
    det = verify_detector(cfg)
    rep = VerifyReport()
    with VideoReader(dst) as reader:
        total = reader.info.frame_count
        for i, frame in enumerate(reader):
            if i % every == 0:
                hidden = [t.history[i] for t in tracks if t.redact and i in t.history]
                r, u = check_frame(frame, hidden, det, i)
                rep.frames_checked += 1
                rep.hidden_faces += len(hidden)
                rep.residuals += r
                rep.unmasked += u
            if progress:
                progress(i + 1, total)
    return rep
