"""Shared data types: boxes, detections, tracks and the run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DETECTORS = ("yunet", "haar")
TRACKERS = ("lk", "csrt", "kcf", "none", "off")
MODES = ("blur", "pixelate", "fill")
SHAPES = ("ellipse", "rect")
ASSOCIATIONS = ("greedy", "hungarian")
COAST_MODELS = ("velocity", "kalman")
VOICES = ("keep", "pitch", "mute")
FLAG_LOST = "lost"  # tracker lost the face; box frozen at its last position
FLAG_UNCONFIRMED = "unconfirmed"  # tracker follows it but the last detection missed it
FLAG_BACKFILL = "backfill"  # box tracked backwards from the first detection (D-026)


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
    lost_since: int = 0  # frame the tracker lost it (valid while lost)
    score: float | None = None  # detector confidence, images only (shown in the preview)
    landmarks: np.ndarray | None = None  # from the last matched detection; for face alignment
    history: dict[int, Box] = field(default_factory=dict)
    # frame -> FLAG_LOST | FLAG_UNCONFIRMED: frames where this box is a guess the
    # detector did not back up. Shown on the UI timeline as "check here".
    flags: dict[int, str] = field(default_factory=dict)


@dataclass
class Config:
    """Every tunable number in FaceGuard, with its default and unit."""

    model_path: str = "models/face_detection_yunet_2026may.onnx"
    haar_path: str = "models/haarcascade_frontalface_default.xml"
    profile_path: str = "models/haarcascade_profileface.xml"
    text_model_path: str = "models/text_detection_en_ppocrv3_2023may.onnx"
    recognizer_path: str = "models/face_recognition_sface_2021dec.onnx"  # optional (allow-list)

    # detection
    detector: str = "yunet"  # yunet | haar
    conf_threshold: float = 0.6  # YuNet score threshold
    nms_threshold: float = 0.3  # YuNet internal non-max suppression IoU
    top_k: int = 5000  # YuNet candidates kept before NMS
    haar_scale_factor: float = 1.1  # image pyramid step for the cascade
    haar_min_neighbors: int = 5  # overlapping hits needed to accept a face
    min_face_px: int = 24  # Haar minSize; also the "small face" bound
    # profile fallback (D-032): add Haar profile-cascade hits (both directions) to the detector's
    profile: bool = False
    profile_score: float = 0.5  # score given to profile hits; below YuNet's threshold so NMS prefers YuNet
    profile_min_neighbors: int = 6  # stricter than the frontal cascade: profile hits are noisier
    profile_nms_iou: float = 0.3  # merge a profile hit into an overlapping detection above this IoU
    tile_grid: int = 0  # 0 = off; n = also detect on n x n overlapping, upscaled tiles (D-022)
    tile_overlap: float = 0.25  # tile overlap, fraction of tile size, so no face is cut by every tile
    tile_scale: float = 2.0  # upscale factor for tiles: a 12 px face is seen at 24 px
    tile_nms_iou: float = 0.3  # merge duplicates from the full frame and the tiles above this IoU

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
    lk_pyr_levels: int = 4  # extra pyramid levels above full resolution (D-013)
    lk_fb_threshold_px: float = 1.0  # forward-backward round-trip tolerance
    lk_min_points: int = 6  # fewer surviving points than this = track lost
    lk_reseed_below: int = 12  # re-seed corners in the box below this many points
    lk_search_frac: float = 1.0  # LK search window margin, as a fraction of box size
    lk_search_min_px: int = 128  # px added to the margin; bounds trackable motion per frame

    # association
    association: str = "greedy"  # greedy | hungarian (optimal total IoU, D-029)
    iou_threshold: float = 0.3  # min IoU to match a detection to a track
    max_misses: int = 15  # frames a LOST track (frozen box) is kept before deletion
    max_unconfirmed: int = 60  # frames a still-tracked track may go without a detector match
    duplicate_iou: float = 0.7  # two tracks above this are the same face
    # coasting (D-023): a LOST track keeps moving at its last velocity, and its mask grows
    coast: bool = False  # off = the lost box stays frozen at its last position
    coast_model: str = "velocity"  # velocity: decaying EMA velocity | kalman: constant-velocity Kalman (D-030)
    kalman_process_noise: float = 1.0  # px/frame^2, std of the unmodelled acceleration
    kalman_measurement_noise: float = 2.0  # px, std of a tracker box centre
    kalman_sigmas: float = 2.0  # lost mask grows by this many position std devs on each side
    coast_decay: float = 0.9  # per-frame velocity decay while coasting (1 = constant velocity)
    coast_grow: float = 0.04  # per-frame growth of the mask, fraction of box size
    coast_max_scale: float = 1.8  # mask size cap while coasting, multiple of the box size
    velocity_smoothing: float = 0.5  # EMA weight of the newest per-frame displacement
    # early detection (D-017): run the detector on the next frame, off schedule, when
    refresh_on_lost: bool = True  # ... a tracker has lost its face
    unconfirmed_interval: int = 3  # ... every N frames while a tracked face went unmatched by the
    #                                 last detection (tracker and detector disagree); 0 disables
    # backfill (D-026): when a face is first detected, track it backwards over the
    # last N buffered frames so the frames before its first detection are masked too
    backfill_frames: int = 10  # 0 = off; frames held in memory (>= detect_interval covers entry gaps)
    backfill_min_visible: float = 0.5  # stop once less than this share of the box is inside the frame
    # scene cuts: a cut needs BOTH a different intensity histogram and a different layout
    scene_cut_corr: float = 0.5  # grey-histogram correlation below this; 0 disables cut detection
    scene_cut_ncc: float = 0.3  # thumbnail normalised cross-correlation below this
    scene_cut_bins: int = 32  # histogram bins
    scene_cut_width: int = 32  # px, thumbnail width for both tests (height keeps aspect)

    # redaction
    mode: str = "pixelate"  # blur | pixelate | fill
    padding: float = 0.25  # per side; >= 0.207 keeps face-box corners masked
    blur_strength: float = 1.0  # Gaussian kernel as a fraction of box width; 0.35 was re-identifiable (D-027)
    blur_min_kernel: int = 11  # px, floor for tiny faces
    pixel_blocks: int = 5  # pixelate: blocks across the padded face; 8 was re-identifiable (D-027)
    feather_frac: float = 0.08  # soft edge width as a fraction of box width
    fill_color: tuple[int, int, int] = (0, 0, 0)  # BGR
    mask_shape: str = "ellipse"  # ellipse (soft, natural) | rect (covers the whole padded box)
    smooth_window: int = 1  # frames; centred moving average of track boxes, 1 = off (D-019)

    # allow-list (D-025): SFace cosine similarity needed to leave a face visible.
    # Stricter than OpenCV's 0.363: a false match exposes a stranger.
    allow_threshold: float = 0.5
    # re-ID linking (D-031): merge a reappearing face's new track into its old one
    reid_link: bool = False  # opt-in; needs the SFace model
    reid_threshold: float = 0.5  # SFace cosine similarity needed to link two tracks
    reid_max_gap: int = 150  # frames; longest gap between two tracks that may be linked

    # text redaction (D-034): number plates, badges, signs; detected, never read
    text: bool = False
    text_binary_threshold: float = 0.3  # DB probability map binarisation (OpenCV Zoo default)
    text_polygon_threshold: float = 0.5  # min mean probability inside a region
    text_unclip_ratio: float = 2.0  # region growth; the network predicts shrunk text kernels
    text_max_candidates: int = 200
    text_max_side: int = 1280  # px; frames are downscaled to this long side for the detector
    text_min_px: int = 8  # ignore text regions shorter than this (unreadable anyway)
    text_padding: float = 0.15  # per side, rectangular mask
    text_iou: float = 0.3  # match a text box to the previous detection's in video

    # parallel analysis (core/parallel.py, D-036)
    parallel_warmup: int = 30  # frames each chunk starts early so tracks exist at its boundary

    # self-check (core/verify.py): re-run YuNet on the exported output
    verify_threshold: float = 0.3  # lower than conf_threshold: the check should be the stronger detector
    verify_every: int = 1  # video: check every N-th output frame

    # io
    keep_audio: bool = False
    voice: str = "keep"  # with keep_audio: keep | pitch (disguise, reversible) | mute (D-033)
    voice_pitch: float = 0.75  # pitch factor for voice = pitch; < 1 lowers the voice

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
        if self.mask_shape not in SHAPES:
            raise ValueError(f"mask_shape must be one of {SHAPES}, got {self.mask_shape!r}")
        if self.tile_grid < 0 or self.tile_scale < 1:
            raise ValueError("tile_grid must be >= 0 and tile_scale >= 1")
        if self.association not in ASSOCIATIONS:
            raise ValueError(f"association must be one of {ASSOCIATIONS}, got {self.association!r}")
        if self.coast_model not in COAST_MODELS:
            raise ValueError(f"coast_model must be one of {COAST_MODELS}, got {self.coast_model!r}")
        if self.voice not in VOICES:
            raise ValueError(f"voice must be one of {VOICES}, got {self.voice!r}")
        if not 0.5 <= self.voice_pitch <= 2.0:
            raise ValueError("voice_pitch must be in [0.5, 2] (atempo's single-stage range)")
        if self.backfill_frames < 0:
            raise ValueError("backfill_frames must be >= 0")
        if self.smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
