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

THUMB_PX = 96  # face gallery crop size, longer side
THUMB_PADDING = 0.3  # context around the face box in gallery crops

JOB_PREFIX = "faceguard-"
STALE_JOB_SECONDS = 6 * 3600  # job folders older than this are from dead sessions


def detections_to_tracks(detections: list[Detection]) -> list[Track]:
    """Wrap single-image detections as one-frame tracks, numbered from 1."""
    return [
        Track(id=i, box=d.box, history={0: d.box}, score=d.score, landmarks=d.landmarks)
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


# --- review navigation and records ------------------------------------------


def frame_runs(frames: list[int] | set[int] | dict) -> list[tuple[int, int]]:
    """Contiguous runs of frame numbers as inclusive (start, end) pairs."""
    runs: list[tuple[int, int]] = []
    for f in sorted(frames):
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def flag_runs(track: Track) -> list[tuple[int, int, str]]:
    """Contiguous runs of flagged frames with their flag kind."""
    out: list[tuple[int, int, str]] = []
    for f in sorted(track.flags):
        kind = track.flags[f]
        if out and f == out[-1][1] + 1 and out[-1][2] == kind:
            out[-1] = (out[-1][0], f, kind)
        else:
            out.append((f, f, kind))
    return out


def flagged_frames(tracks: list[Track], redacted_only: bool = True) -> list[int]:
    """Frames where some box is a tracker guess the detector did not confirm."""
    return sorted({f for t in tracks if t.redact or not redacted_only for f in t.flags})


def review_stops(tracks: list[Track]) -> list[int]:
    """Where a reviewer should look: the first frame of every flagged run."""
    return sorted({a for t in tracks if t.redact for a, _, _ in flag_runs(t)})


def step_to(stops: list[int], frame: int, direction: int) -> int | None:
    """Next (direction=+1) or previous (-1) stop strictly after/before `frame`."""
    if direction > 0:
        return next((s for s in stops if s > frame), None)
    return next((s for s in reversed(stops) if s < frame), None)


class ThumbnailCollector:
    """Keeps one small face crop per track while analysis streams the video.

    Used as the pipeline's per-frame hook, so no frame is decoded twice and
    none is kept: only the crop of the largest detector-confirmed box seen
    for each track (confirmed boxes are the best framed; flagged ones may be
    drifting).
    """

    def __init__(self, size: int = THUMB_PX, padding: float = THUMB_PADDING) -> None:
        self.size = size
        self.padding = padding
        self.thumbs: dict[int, np.ndarray] = {}
        self._area: dict[int, int] = {}

    def __call__(self, i: int, frame: np.ndarray, present: list[Track]) -> None:
        h, w = frame.shape[:2]
        for t in present:
            box = t.history.get(i)
            if box is None or i in t.flags or box.area <= self._area.get(t.id, 0):
                continue
            self.thumbs[t.id] = crop_thumbnail(frame, box, self.size, self.padding)
            self._area[t.id] = box.area


def crop_thumbnail(frame: np.ndarray, box: Box, size: int = 0, padding: float = 0.0) -> np.ndarray:
    """Square-ish crop around a box (BGR), resized so its longer side is `size`."""
    h, w = frame.shape[:2]
    pb = box.pad(padding, w, h)
    if pb.area == 0:
        return np.zeros((max(size, 1), max(size, 1), 3), np.uint8)
    crop = frame[pb.y : pb.y + pb.h, pb.x : pb.x + pb.w]
    size = size or THUMB_PX
    s = size / max(pb.w, pb.h)
    return cv2.resize(crop, (max(1, round(pb.w * s)), max(1, round(pb.h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def redaction_log(tracks: list[Track], cfg: Config, media: dict, events: list | None = None) -> dict:
    """An audit record of what was hidden: no pixels, no file contents.

    Lists every region with its frame ranges, whether it was redacted, how
    it was made (detector or manual) and which frames were tracker guesses,
    plus the settings that decide the mask. Meant to be kept next to the
    exported file so a reviewer can see what was checked.
    """
    from dataclasses import asdict  # local: only needed here

    settings = {k: v for k, v in asdict(cfg).items() if not k.endswith("_path")}
    return {
        "tool": "FaceGuard",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "media": media,
        "settings": settings,
        "summary": {
            "regions": len(tracks),
            "redacted": sum(t.redact for t in tracks),
            "left_visible": [t.id for t in tracks if not t.redact],
            "manual": [t.id for t in tracks if t.source == "manual"],
            "frames_to_review": len(flagged_frames(tracks)),
        },
        "regions": [
            {
                "id": t.id,
                "source": t.source,
                "redact": t.redact,
                "frames": [list(r) for r in frame_runs(t.history)],
                "flagged": [list(r) for r in flag_runs(t)],
            }
            for t in tracks
        ],
        "scene_cuts": sorted({e.frame for e in (events or []) if e.kind == "scene_cut"}),
    }


SESSION_VERSION = 1


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def session_to_json(tracks: list[Track], src: str | Path, analysed_with: dict) -> dict:
    """A resumable review: every track with its boxes, flags and hide/show choice.

    Bound to the source file by SHA-256 so it cannot be applied to another
    video by mistake. Contains no pixels and no face embeddings (the
    allow-list is re-run from references after loading, D-025).
    """
    from core.pipeline import tracks_to_json

    return {"faceguard_session": SESSION_VERSION, "source_sha256": file_sha256(src),
            "analysed_with": analysed_with, **tracks_to_json(tracks)}


def session_from_json(data: dict, src: str | Path) -> tuple[list[Track], dict]:
    """Tracks and the detection settings they were made with. Raises on a mismatch."""
    from core.pipeline import tracks_from_json

    if not isinstance(data, dict) or data.get("faceguard_session") != SESSION_VERSION:
        raise FaceGuardError("Not a FaceGuard review file.")
    if data.get("source_sha256") != file_sha256(src):
        raise FaceGuardError("This review was saved for a different file.")
    return tracks_from_json(data), data.get("analysed_with", {})
