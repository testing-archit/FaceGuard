"""Shared data types: boxes, detections, tracks and the run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DETECTORS = ("yunet", "haar")
TRACKERS = ("lk", "csrt", "kcf", "none", "off")
MODES = ("blur", "pixelate", "fill")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, (x, y) is the top-left corner."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(self.w, 0) * max(self.h, 0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def intersection(self, other: Box) -> int:
        """Area of overlap with another box, 0 if they do not touch."""
        ix = min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        iy = min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        return max(ix, 0) * max(iy, 0)

    def iou(self, other: Box) -> float:
        """Intersection over union, the standard overlap score in [0, 1]."""
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, frame_w: int, frame_h: int) -> Box:
        """Clip to the frame. A box fully outside collapses to zero size."""
        x0 = min(max(self.x, 0), frame_w)
        y0 = min(max(self.y, 0), frame_h)
        x1 = min(max(self.x + self.w, 0), frame_w)
        y1 = min(max(self.y + self.h, 0), frame_h)
        return Box(x0, y0, x1 - x0, y1 - y0)

    def pad(self, frac: float, frame_w: int, frame_h: int) -> Box:
        """Grow by `frac` of the box size on EACH side, then clip to the frame.

        Detector boxes hug the face; hairline, ears and chin fall outside them.
        """
        dx = round(self.w * frac)
        dy = round(self.h * frac)
        grown = Box(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)
        return grown.clip(frame_w, frame_h)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Box:
        """Build from float coordinates (detector output), rounding to pixels."""
        return cls(round(x), round(y), round(w), round(h))

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class Detection:
    """One face found by a detector in one frame."""

    box: Box
    score: float
    landmarks: np.ndarray | None = None  # shape (5, 2): eyes, nose, mouth corners


@dataclass
class Track:
    """One person followed through a video. `history` maps frame -> box."""

    id: int
    box: Box
    redact: bool = True
    source: str = "detector"  # "detector" | "manual"
    last_seen_frame: int = 0  # last frame a detection matched this track
    lost: bool = False  # tracker returned None; box is frozen in place
    history: dict[int, Box] = field(default_factory=dict)


@dataclass
class Config:
    """Every tunable number in FaceGuard, with its default and unit."""

    model_path: str = "models/face_detection_yunet_2026may.onnx"
    haar_path: str = "models/haarcascade_frontalface_default.xml"

    # detection
    detector: str = "yunet"  # yunet | haar
    conf_threshold: float = 0.6  # YuNet score threshold
    nms_threshold: float = 0.3  # YuNet internal non-max suppression IoU
    top_k: int = 5000  # YuNet candidates kept before NMS
    haar_scale_factor: float = 1.1  # image pyramid step for the cascade
    haar_min_neighbors: int = 5  # overlapping hits needed to accept a face
    min_face_px: int = 24  # Haar minSize; also the "small face" bound

    # tracking
    # lk | csrt | kcf: propagate boxes between detections
    # none: hold each box still between detections
    # off:  pure per-frame baseline, a face is masked only on frames where it
    #       is detected (forces detect_interval = 1, no grace period)
    tracker: str = "lk"
    detect_interval: int = 10  # run the detector every N frames
    lk_max_corners: int = 40  # Shi-Tomasi corners seeded per face
    lk_quality_level: float = 0.01  # min corner response, relative to best
    lk_min_distance: int = 3  # px between seeded corners
    lk_win_size: int = 21  # px, LK integration window
    lk_pyr_levels: int = 3  # extra pyramid levels above full resolution
    lk_fb_threshold_px: float = 1.0  # forward-backward round-trip tolerance
    lk_min_points: int = 6  # fewer surviving points than this = track lost
    lk_reseed_below: int = 12  # re-seed corners in the box below this many points

    # association
    iou_threshold: float = 0.3  # min IoU to match a detection to a track
    max_misses: int = 15  # grace period in video frames before deletion
    duplicate_iou: float = 0.7  # two tracks above this are the same face

    # redaction
    mode: str = "pixelate"  # blur | pixelate | fill
    padding: float = 0.25  # per side; >= 0.207 keeps face-box corners masked
    blur_strength: float = 0.35  # Gaussian kernel as a fraction of box width
    blur_min_kernel: int = 11  # px, floor for tiny faces
    pixel_blocks: int = 8  # pixelate: downscale ROI to N x N blocks
    feather_frac: float = 0.08  # soft edge width as a fraction of box width
    fill_color: tuple[int, int, int] = (0, 0, 0)  # BGR

    # io
    keep_audio: bool = False

    def __post_init__(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {self.detector!r}")
        if self.tracker not in TRACKERS:
            raise ValueError(f"tracker must be one of {TRACKERS}, got {self.tracker!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.detect_interval < 1:
            raise ValueError("detect_interval must be >= 1")
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if self.padding < 0:
            raise ValueError("padding must be >= 0")
