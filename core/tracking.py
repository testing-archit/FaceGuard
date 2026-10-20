"""Trackers that carry a face box from one frame to the next between detections.

All trackers implement:
    init(frame, gray, box) -> None
    update(frame, gray) -> Box | None      # None means the target is lost

`gray` is computed once per frame by the pipeline and shared by every track.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from core.types import Box, Config


class Tracker(Protocol):
    def init(self, frame: np.ndarray, gray: np.ndarray, box: Box) -> None: ...
    def update(self, frame: np.ndarray, gray: np.ndarray) -> Box | None: ...


TrackerFactory = Callable[[], Tracker]


class LKTracker:
    """Sparse pyramidal Lucas-Kanade optical flow on Shi-Tomasi corners.

    Concepts:
    - Shi-Tomasi corners: points whose structure tensor has two large
      eigenvalues, i.e. intensity changes in both directions, so their motion
      is fully determined (no aperture problem). Seeded inside the face box.
    - Lucas-Kanade: solves for each point's displacement assuming
      (1) brightness constancy, (2) small motion between frames and
      (3) all pixels in a small window move together (spatial coherence).
    - Image pyramid: running LK coarse-to-fine relaxes assumption (2), so
      large motions become small at the top level.
    - Forward-backward check: track points forward, then back again; points
      that do not return within `lk_fb_threshold_px` are bad correspondences
      (occlusion, illumination change) and are discarded.
    - Box motion: median displacement (translation) and median ratio of
      pairwise point distances (scale). Medians are robust to the few outliers
      that survive the check.

    Assumptions break on fast motion beyond the pyramid's reach, occlusion and
    sudden lighting change; those frames return None (lost).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._lk_params = dict(
            winSize=(cfg.lk_win_size, cfg.lk_win_size),
            maxLevel=cfg.lk_pyr_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._prev_gray: np.ndarray | None = None
        self._points: np.ndarray | None = None  # (N, 1, 2) float32
        self._box = np.zeros(4, np.float64)  # x, y, w, h as floats; no rounding drift

    def init(self, frame: np.ndarray, gray: np.ndarray, box: Box) -> None:
        self._prev_gray = gray
        self._box = np.array([box.x, box.y, box.w, box.h], np.float64)
        self._points = self._seed(gray)

    def _seed(self, gray: np.ndarray) -> np.ndarray | None:
        """Shi-Tomasi corners restricted to the current box."""
        h, w = gray.shape[:2]
        x, y, bw, bh = self._box
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros_like(gray)
        mask[y0:y1, x0:x1] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._cfg.lk_max_corners,
            qualityLevel=self._cfg.lk_quality_level,
            minDistance=self._cfg.lk_min_distance,
            mask=mask,
        )

    def update(self, frame: np.ndarray, gray: np.ndarray) -> Box | None:
        cfg = self._cfg
        if self._prev_gray is None or self._points is None or len(self._points) < cfg.lk_min_points:
            return None

        p0 = self._points
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **self._lk_params)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **self._lk_params)
        fb_error = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_error < cfg.lk_fb_threshold_px)
        if good.sum() < cfg.lk_min_points:
            self._points = None
            return None

        old = p0[good].reshape(-1, 2)
        new = p1[good].reshape(-1, 2)
        dx, dy = np.median(new - old, axis=0)
        scale = _median_scale(old, new)

        x, y, w, h = self._box
        cx, cy = x + w / 2 + dx, y + h / 2 + dy
        w, h = w * scale, h * scale
        self._box = np.array([cx - w / 2, cy - h / 2, w, h])

        self._prev_gray = gray
        self._points = new.reshape(-1, 1, 2).astype(np.float32)
        # Points are lost to the FB check over time; top up from the current box
        # so a track can survive long detection intervals.
        if len(self._points) < cfg.lk_reseed_below:
            seeded = self._seed(gray)
            if seeded is not None and len(seeded) > len(self._points):
                self._points = seeded
        return Box.from_xywh(*self._box)


def _median_scale(old: np.ndarray, new: np.ndarray, min_dist: float = 1.0) -> float:
    """Median ratio of pairwise point distances, new over old.

    `min_dist` drops near-coincident pairs whose ratio is numerically unstable.
    """
    i, j = np.triu_indices(len(old), k=1)
    d_old = np.linalg.norm(old[i] - old[j], axis=1)
    d_new = np.linalg.norm(new[i] - new[j], axis=1)
    keep = d_old > min_dist
    if not keep.any():
        return 1.0
    return float(np.median(d_new[keep] / d_old[keep]))


def _make_opencv_tracker(kind: str):
    """Construct TrackerCSRT / TrackerKCF across OpenCV 4 and 5 Python bindings."""
    for name in (f"Tracker{kind}", f"Tracker{kind}_create"):
        fn = getattr(cv2, name, None)
        if fn is not None:
            return fn.create() if hasattr(fn, "create") else fn()
    legacy = getattr(cv2, "legacy", None)
    fn = getattr(legacy, f"Tracker{kind}_create", None) if legacy else None
    if fn is not None:
        return fn()
    raise RuntimeError(
        f"{kind} tracker unavailable. Uninstall opencv-python and install "
        "opencv-contrib-python (the trackers live in the contrib build)."
    )


class OpenCVTracker:
    """Wrapper over OpenCV's Tracking API (CSRT or KCF), for comparison.

    - KCF (Kernelized Correlation Filter): learns a correlation filter on
      HOG features, using circulant structure so training is a few FFTs.
      Very fast, but fixed scale and it loses targets after big appearance
      changes.
    - CSRT (Discriminative Correlation Filter with Channel and Spatial
      Reliability): adds a spatial reliability map and per-channel weights,
      handles scale. More accurate, much slower.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._tracker = None

    def init(self, frame: np.ndarray, gray: np.ndarray, box: Box) -> None:
        self._tracker = _make_opencv_tracker(self._kind)
        self._tracker.init(frame, (box.x, box.y, max(box.w, 1), max(box.h, 1)))

    def update(self, frame: np.ndarray, gray: np.ndarray) -> Box | None:
        if self._tracker is None:
            return None
        ok, rect = self._tracker.update(frame)
        if not ok:
            return None
        return Box.from_xywh(*rect)


class NullTracker:
    """Holds the box still between detections ("hold last detection")."""

    def __init__(self) -> None:
        self._box: Box | None = None

    def init(self, frame: np.ndarray, gray: np.ndarray, box: Box) -> None:
        self._box = box

    def update(self, frame: np.ndarray, gray: np.ndarray) -> Box | None:
        return self._box


def tracker_factory(cfg: Config) -> TrackerFactory:
    """Return a zero-argument constructor for the configured tracker."""
    if cfg.tracker == "lk":
        return lambda: LKTracker(cfg)
    if cfg.tracker == "csrt":
        _make_opencv_tracker("CSRT")  # fail fast on the wrong OpenCV wheel
        return lambda: OpenCVTracker("CSRT")
    if cfg.tracker == "kcf":
        _make_opencv_tracker("KCF")
        return lambda: OpenCVTracker("KCF")
    if cfg.tracker in ("none", "off"):
        return NullTracker
    raise ValueError(f"Unknown tracker {cfg.tracker!r}")
