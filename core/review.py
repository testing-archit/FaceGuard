"""Review and correction: everything the UI needs between analyse and render.

No Streamlit imports here (CLAUDE.md); the UI calls these and only converts
BGR to RGB at its own boundary. All functions work on the `Track` list that
`analyse_video` returns, so an image is treated as a one-frame video and
both share one review path.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core.errors import FaceGuardError
from core.tracking import tracker_factory
from core.types import Box, Config, Detection, Track
from core.video_io import VideoReader

# Overlay drawing: purely cosmetic, never exported (FR-22).
OVERLAY_ON = (0, 200, 0)  # BGR: track will be redacted
OVERLAY_OFF = (0, 140, 255)  # BGR: user unticked it, left visible
OVERLAY_MANUAL = (255, 160, 0)  # BGR: user-drawn track
OVERLAY_DRAFT = (255, 0, 255)  # BGR: manual box being entered, not yet added

JOB_PREFIX = "faceguard-"
STALE_JOB_SECONDS = 6 * 3600  # job folders older than this are from dead sessions


def detections_to_tracks(detections: list[Detection]) -> list[Track]:
    """Wrap single-image detections as one-frame tracks, numbered from 1."""
    return [
        Track(id=i, box=d.box, history={0: d.box}, score=d.score)
        for i, d in enumerate(detections, start=1)
    ]


def next_track_id(tracks: list[Track]) -> int:
    return max((t.id for t in tracks), default=0) + 1


def delete_track(tracks: list[Track], track_id: int) -> list[Track]:
    """Drop a false-positive track (FR-25). Returns a new list."""
    return [t for t in tracks if t.id != track_id]


def set_redact(tracks: list[Track], track_id: int, redact: bool) -> None:
    """Selective redaction (FR-23): the flag follows the track through the clip."""
    for t in tracks:
        if t.id == track_id:
            t.redact = redact


def tracks_on_frame(tracks: list[Track], frame: int) -> list[Track]:
    return [t for t in tracks if frame in t.history]


def manual_track(track_id: int, history: dict[int, Box]) -> Track:
    """A user-drawn region, stored as an ordinary track (FR-24)."""
    if not history:
        raise FaceGuardError("A manual box needs at least one frame.")
    first = min(history)
    return Track(id=track_id, box=history[first], source="manual", last_seen_frame=first, history=history)


def propagate(
    src: str | Path,
    start: int,
    box: Box,
    cfg: Config,
    end: int | None = None,
) -> dict[int, Box]:
    """Carry a manual box forward from frame `start` with the configured tracker.

    The box is used as-is on `start` and then followed exactly like a
    detection between detector runs, until the tracker loses it, frame `end`
    (inclusive) or the end of the clip. The same tracker as the analysis run
    is used, so a manual track behaves like an automatic one. With tracker
    "none"/"off" there is no motion model, so the box is held still up to
    `end` (or for `max_misses` frames if no end is given).

    When the tracker loses the face, the frozen box is kept for `max_misses`
    frames, the same grace rule automatic tracks get (FR-15).
    """
    if box.area == 0:
        raise FaceGuardError("Manual box has zero size.")
    follows = cfg.tracker in ("lk", "csrt", "kcf")
    stop = end if end is not None else (None if follows else start + cfg.max_misses)
    tracker = tracker_factory(cfg)()
    need_gray = cfg.tracker == "lk"
    history: dict[int, Box] = {}
    lost_at: int | None = None
    with VideoReader(src) as reader:
        if not reader.seek(start):
            raise FaceGuardError(f"Frame {start} is past the end of the video.")
        current = box
        for i, frame in enumerate(reader, start=start):
            if stop is not None and i > stop:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if need_gray else None
            if i == start:
                tracker.init(frame, gray, box)
            elif lost_at is None:
                nxt = tracker.update(frame, gray)
                if nxt is None:
                    lost_at = i
                else:
                    current = nxt
            if lost_at is not None and i - lost_at > cfg.max_misses:
                break
            history[i] = current.clip(reader.info.width, reader.info.height)
    return history


def interpolate_boxes(a_frame: int, a: Box, b_frame: int, b: Box) -> dict[int, Box]:
    """Linear keyframe interpolation for a manual correction between two frames.

    For a face the tracker cannot follow: the user supplies the box at the
    start and end of an interval and the box is interpolated between. Motion
    need not be linear, so the whole interval must still be reviewed.
    """
    if b_frame < a_frame:
        a_frame, a, b_frame, b = b_frame, b, a_frame, a
    span = max(b_frame - a_frame, 1)
    out = {}
    for f in range(a_frame, b_frame + 1):
        u = (f - a_frame) / span
        out[f] = Box.from_xywh(*(p + (q - p) * u for p, q in zip(a.as_list(), b.as_list())))
    return out


def read_frame(src: str | Path, index: int) -> np.ndarray:
    """Decode a single frame for the preview scrubber."""
    with VideoReader(src) as reader:
        if not reader.seek(index):
            raise FaceGuardError(f"Frame {index} is past the end of the video.")
        for frame in reader:
            return frame
    raise FaceGuardError(f"Could not decode frame {index}.")


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    index: int,
    draft: Box | None = None,
    cfg: Config | None = None,
) -> np.ndarray:
    """Numbered boxes per track on a copy of `frame` (preview only).

    Green = will be redacted, orange = left visible, blue = manual. The
    padded region that the mask actually covers is drawn thin, so the user
    sees what padding does.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    thick = max(1, round(min(h, w) / 360))
    scale = max(0.4, min(h, w) / 900)
    for t in tracks:
        box = t.history.get(index)
        if box is None:
            continue
        color = OVERLAY_MANUAL if t.source == "manual" else OVERLAY_ON
        if not t.redact:
            color = OVERLAY_OFF
        _rect(out, box, color, thick)
        if cfg is not None:
            _rect(out, box.pad(cfg.padding, w, h), color, 1)
        label = f"{t.id}" + (f" {t.score:.2f}" if t.score is not None else "")
        _label(out, label, box, color, scale, thick)
    if draft is not None and draft.area > 0:
        _rect(out, draft, OVERLAY_DRAFT, thick)
        _label(out, "new", draft, OVERLAY_DRAFT, scale, thick)
    return out


def _rect(img: np.ndarray, b: Box, color: tuple[int, int, int], thick: int) -> None:
    cv2.rectangle(img, (b.x, b.y), (b.x + b.w - 1, b.y + b.h - 1), color, thick, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, b: Box, color: tuple[int, int, int], scale: float, thick: int) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = b.y - 4 if b.y - th - 6 >= 0 else b.y + b.h + th + 4
    cv2.rectangle(img, (b.x, y - th - 3), (b.x + tw + 4, y + base - 1), color, -1)
    cv2.putText(img, text, (b.x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


# --- session files ---------------------------------------------------------


def new_job_dir(root: str | Path | None = None) -> Path:
    """Private temporary folder for one editing session's uploads and outputs."""
    return Path(tempfile.mkdtemp(prefix=JOB_PREFIX, dir=root))


def remove_job_dir(path: str | Path | None) -> None:
    """Reset: delete uploaded bytes, previews and exports. Ordinary deletion,
    not secure erasure."""
    if path:
        shutil.rmtree(path, ignore_errors=True)


def remove_stale_jobs(root: str | Path | None = None, max_age_s: float = STALE_JOB_SECONDS) -> int:
    """Best-effort cleanup of job folders left by crashed sessions, run at launch."""
    base = Path(root) if root else Path(tempfile.gettempdir())
    removed = 0
    now = time.time()
    for p in base.glob(JOB_PREFIX + "*"):
        try:
            if p.is_dir() and now - p.stat().st_mtime > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
