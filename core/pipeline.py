"""Orchestration: analyse (find faces) then render (redact them).

Images go straight through detect -> redact. For video:

- `analyse_video` decodes once and returns every track with its per-frame
  boxes (no pixels kept). The UI shows these for review.
- `render_video` decodes again and replays the stored boxes of the selected
  tracks. It never re-runs detection, so the IDs the user toggled are exactly
  the IDs rendered (DECISIONS D-010).
- `process_video` does both in one pass for the CLI. This is valid because
  every box at frame i is decided causally at frame i.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.association import IdEvent, TrackManager
from core.detectors import Detector, make_detector
from core.errors import FaceGuardError, media_kind
from core.redact import redact
from core.tracking import tracker_factory
from core.types import Box, Config, Detection, Track
from core.video_io import VideoInfo, VideoReader, VideoWriter, finalise, limit_warnings

Progress = Callable[[int, int], None]  # (frames done, total frames)


@dataclass
class RunStats:
    """What happened during one video run; feeds the evaluation tables."""

    frames: int = 0
    loop_seconds: float = 0.0  # decode + detect/track + redact + write
    total_seconds: float = 0.0  # including ffmpeg finalisation
    detect_seconds: float = 0.0
    track_seconds: float = 0.0  # tracker updates + association
    redact_seconds: float = 0.0
    detector_calls: int = 0
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


@dataclass
class Analysis:
    """Result of analysing a video: tracks with per-frame boxes, no pixels."""

    info: VideoInfo
    tracks: list[Track]
    events: list[IdEvent]
    stats: RunStats

    def boxes_at(self, frame: int, selected: set[int] | None = None) -> list[Box]:
        return boxes_at(self.tracks, frame, selected)


FrameHook = Callable[[int, np.ndarray, list[Track]], None]


def _needs_gray(cfg: Config) -> bool:
    return cfg.tracker == "lk" or cfg.detector == "haar"


def _run(src: str | Path, cfg: Config, progress: Progress | None, on_frame: FrameHook | None) -> Analysis:
    """Decode -> (detect every N frames) -> track -> associate, frame by frame."""
    detector = make_detector(cfg)
    manager = TrackManager(cfg, tracker_factory(cfg))
    interval = 1 if cfg.tracker == "off" else cfg.detect_interval
    need_gray = _needs_gray(cfg)
    stats = RunStats()
    start = time.perf_counter()
    with VideoReader(src) as reader:
        stats.warnings += limit_warnings(reader.info)
        total = reader.info.frame_count
        for i, frame in enumerate(reader):
            # Convert once per frame; LK and Haar share it (ARCHITECTURE section 10).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if need_gray else None
            detections = None
            if i % interval == 0:
                t0 = time.perf_counter()
                detections = detector.detect(frame, gray)
                stats.detect_seconds += time.perf_counter() - t0
                stats.detector_calls += 1
            t0 = time.perf_counter()
            present = manager.step(i, frame, gray, detections)
            stats.track_seconds += time.perf_counter() - t0
            if on_frame:
                on_frame(i, frame, present)
            stats.frames += 1
            if progress:
                progress(i + 1, total)
        info = reader.info
    stats.loop_seconds = time.perf_counter() - start
    if stats.frames == 0:
        raise FaceGuardError(f"No frames could be decoded from {src}.")
    return Analysis(info=info, tracks=manager.all_tracks(), events=manager.events, stats=stats)


def analyse_video(src: str | Path, cfg: Config, progress: Progress | None = None) -> Analysis:
    """Pass 1: find and track every face. Keeps boxes only, never frames."""
    return _run(src, cfg, progress, on_frame=None)


def boxes_at(tracks: list[Track], frame: int, selected: set[int] | None = None) -> list[Box]:
    """Boxes to redact on one frame: tracks marked redact (and in `selected`, if given)."""
    return [
        t.history[frame]
        for t in tracks
        if t.redact and (selected is None or t.id in selected) and frame in t.history
    ]


def render_video(
    src: str | Path,
    dst: str | Path,
    tracks: list[Track],
    cfg: Config,
    progress: Progress | None = None,
) -> RunStats:
    """Pass 2: replay stored boxes for tracks with redact=True and write the output."""
    stats = RunStats()
    start = time.perf_counter()
    with VideoReader(src) as reader:
        total = reader.info.frame_count
        with VideoWriter(dst, reader.info) as writer:
            for i, frame in enumerate(reader):
                boxes = boxes_at(tracks, i)
                t0 = time.perf_counter()
                writer.write(redact(frame, boxes, cfg))
                stats.redact_seconds += time.perf_counter() - t0
                stats.faces_redacted += len(boxes)
                stats.frames += 1
                if progress:
                    progress(i + 1, total)
    stats.loop_seconds = time.perf_counter() - start
    stats.warnings += finalise(writer.path, dst, src if cfg.keep_audio else None)
    stats.total_seconds = time.perf_counter() - start
    return stats


def process_video(
    src: str | Path, dst: str | Path, cfg: Config, progress: Progress | None = None
) -> Analysis:
    """Analyse and render in a single pass, redacting every track (CLI path)."""
    with VideoReader(src) as probe:
        info = probe.info
    start = time.perf_counter()
    redact_time = 0.0
    faces = 0
    with VideoWriter(dst, info) as writer:

        def write(i: int, frame: np.ndarray, present: list[Track]) -> None:
            nonlocal redact_time, faces
            boxes = [t.box for t in present if t.redact]
            t0 = time.perf_counter()
            writer.write(redact(frame, boxes, cfg))
            redact_time += time.perf_counter() - t0
            faces += len(boxes)

        analysis = _run(src, cfg, progress, on_frame=write)
    analysis.stats.redact_seconds = redact_time
    analysis.stats.faces_redacted = faces
    analysis.stats.warnings += finalise(writer.path, dst, src if cfg.keep_audio else None)
    analysis.stats.total_seconds = time.perf_counter() - start
    return analysis


def tracks_to_json(tracks: list[Track], cfg: Config | None = None) -> dict:
    """Serialise tracks for evaluation (`preds.json`) or for the UI session.

    The config is stored with them because mask geometry (padding, feather)
    decides what coverage the boxes actually achieve.
    """
    return {
        "config": asdict(cfg) if cfg else None,
        "tracks": [
            {
                "id": t.id,
                "source": t.source,
                "redact": t.redact,
                "boxes": {str(f): b.as_list() for f, b in sorted(t.history.items())},
            }
            for t in tracks
        ]
    }


def tracks_from_json(data: dict) -> list[Track]:
    tracks = []
    for d in data["tracks"]:
        history = {int(f): Box(*b) for f, b in d["boxes"].items()}
        last = max(history) if history else 0
        tracks.append(
            Track(
                id=d["id"],
                box=history[last] if history else Box(0, 0, 0, 0),
                redact=d.get("redact", True),
                source=d.get("source", "detector"),
                last_seen_frame=last,
                history=history,
            )
        )
    return tracks


def save_tracks(path: str | Path, tracks: list[Track], cfg: Config | None = None) -> None:
    Path(path).write_text(json.dumps(tracks_to_json(tracks, cfg)))
