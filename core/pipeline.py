"""Orchestration: analyse (find faces) then render (redact them).

Images go straight through detect -> redact. For video:

- `analyse_video` decodes once and returns every track with its per-frame
  boxes (no pixels kept). The UI shows these for review.
- `render_video` decodes again and replays the stored boxes of the selected
  tracks. It never re-runs detection, so the IDs the user toggled are exactly
  the IDs rendered (DECISIONS D-010).
- `process_video` does both in one pass for the CLI. Every box at frame i
  is final once frame i + `backfill_frames` has been analysed, so the writer
  simply runs that many frames behind the analysis.
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
from core.backfill import BufferedFrame, FrameBuffer, backfill_track
from core.detectors import Detector, make_detector
from core.errors import FaceGuardError, media_kind
from core.redact import redact
from core.reid import link_tracks
from core.text import TEXT_SOURCE, TextTracks, redact_text
from core.smoothing import smooth_tracks
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
    early_detections: int = 0  # unscheduled detector runs (lost track or scene cut)
    scene_cuts: int = 0
    faces_redacted: int = 0  # sum over frames
    backfilled_frames: int = 0  # face-frames covered by tracking backwards (D-026)
    reid_seconds: float = 0.0  # SFace embedding for re-ID linking (D-031)
    text_seconds: float = 0.0  # text detection (D-034)
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
    return cfg.tracker == "lk" or cfg.detector == "haar" or cfg.profile


@dataclass
class FrameSignature:
    """What scene-cut detection compares between consecutive frames."""

    hist: np.ndarray  # normalised intensity histogram
    thumb: np.ndarray  # zero-mean, unit-variance grey thumbnail


def frame_signature(frame: np.ndarray, cfg: Config, gray: np.ndarray | None = None) -> FrameSignature:
    """Histogram and normalised thumbnail of a downscaled frame.

    - The histogram ignores where things are, so motion barely changes it,
      but a global brightness change (a fade, a light switched off) does.
    - The normalised thumbnail is invariant to brightness and contrast, but
      large motion (a whip pan) changes it.
    A real cut changes both, which is why `is_scene_cut` requires both.
    """
    g = gray if gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    tw = min(cfg.scene_cut_width, w)
    small = cv2.resize(g, (tw, max(1, round(h * tw / w))), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0], None, [cfg.scene_cut_bins], [0, 256])
    thumb = small.astype(np.float32)
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return FrameSignature(cv2.normalize(hist, hist).ravel(), thumb)


def is_scene_cut(prev: FrameSignature | None, cur: FrameSignature, cfg: Config) -> bool:
    """Cut = intensity distribution AND spatial layout both changed."""
    if prev is None or cfg.scene_cut_corr <= 0:
        return False
    hist_corr = cv2.compareHist(prev.hist, cur.hist, cv2.HISTCMP_CORREL)
    ncc = float((prev.thumb * cur.thumb).mean())
    return hist_corr < cfg.scene_cut_corr and ncc < cfg.scene_cut_ncc


FinalHook = Callable[[int, np.ndarray, list[Box], list[Box]], None]  # (frame, pixels, face boxes, text boxes)


def render_frame(frame: np.ndarray, faces: list[Box], texts: list[Box], cfg: Config) -> np.ndarray:
    """Redact faces (configured mask shape) and text regions (rectangles) on one frame."""
    out = redact(frame, faces, cfg)
    return redact_text(out, texts, cfg) if texts else out


def _run(
    src: str | Path,
    cfg: Config,
    progress: Progress | None,
    on_frame: FrameHook | None,
    on_final: FinalHook | None = None,
) -> Analysis:
    """Decode -> detect when due -> track -> associate, frame by frame.

    Detection is due on every `detect_interval`-th frame, and also
    immediately (DECISIONS D-017):
    - on the frame after any tracker lost its face (`refresh_on_lost`);
    - every `unconfirmed_interval` frames while a tracked face went unmatched
      at the last detection: an occluded face's track can be dragged along
      by whatever passes in front without ever being lost, so the hidden face
      is re-acquired as soon as it re-emerges, not at the next scheduled frame;
    - on a scene cut, after ending every live track.

    With `backfill_frames > 0` the last N frames are held in memory and each
    newly detected face is tracked backwards into them (core/backfill.py,
    D-026). `on_final(j, frame, boxes)` is called once frame j can no longer
    change, i.e. N frames later; with backfill off that is immediately.
    """
    detector = make_detector(cfg)
    collector = None
    if cfg.reid_link:
        from core.recognition import EmbeddingCollector, FaceMatcher

        collector = EmbeddingCollector(FaceMatcher(cfg))
    make_tracker = tracker_factory(cfg)
    manager = TrackManager(cfg, make_tracker)
    interval = 1 if cfg.tracker == "off" else cfg.detect_interval
    need_gray = _needs_gray(cfg)
    backfill = cfg.backfill_frames if cfg.tracker != "off" else 0
    buffer = FrameBuffer(backfill)
    last_cut = 0
    stats = RunStats()
    start = time.perf_counter()
    prev_sig: FrameSignature | None = None

    text = TextTracks(cfg, manager.allocate_id) if cfg.text else None

    def finalise_frame(bf: BufferedFrame) -> None:
        if on_final:
            tracks = manager.live + manager.finished
            texts = text.all_tracks() if text else []
            on_final(bf.index, bf.frame, boxes_at(tracks, bf.index), text_boxes_at(texts, bf.index))

    with VideoReader(src) as reader:
        stats.warnings += limit_warnings(reader.info)
        total = reader.info.frame_count
        for i, frame in enumerate(reader):
            # Convert once per frame; LK and Haar share it (ARCHITECTURE section 10).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if need_gray else None
            cut = False
            if cfg.scene_cut_corr > 0:
                sig = frame_signature(frame, cfg, gray)
                cut = is_scene_cut(prev_sig, sig, cfg)
                prev_sig = sig
            if cut:
                manager.scene_cut(i)
                if text:
                    text.end_all()
                stats.scene_cuts += 1
                last_cut = i
            due = i % interval == 0
            early = not due and (
                cut
                or (cfg.refresh_on_lost and manager.any_lost)
                or (cfg.unconfirmed_interval > 0 and i % cfg.unconfirmed_interval == 0
                    and manager.any_unconfirmed)
            )
            detections = None
            if due or early:
                stats.early_detections += early
                t0 = time.perf_counter()
                detections = detector.detect(frame, gray)
                stats.detect_seconds += time.perf_counter() - t0
                stats.detector_calls += 1
            t0 = time.perf_counter()
            present = manager.step(i, frame, gray, detections)
            current = BufferedFrame(i, frame, gray)
            if text:
                t1 = time.perf_counter()
                past = [b.index for b in buffer.newest_first() if b.index >= last_cut]
                text.step(i, frame, due, past)
                stats.text_seconds += time.perf_counter() - t1
            if backfill and manager.born:
                past = buffer.newest_first()
                for t in manager.born:
                    if t in manager.live:  # not dropped as a duplicate on the same frame
                        stats.backfilled_frames += backfill_track(t, current, past, make_tracker(), cfg, last_cut)
            stats.track_seconds += time.perf_counter() - t0
            if collector:
                t0 = time.perf_counter()
                collector(i, frame, present)
                stats.reid_seconds += time.perf_counter() - t0
            if on_frame:
                on_frame(i, frame, present)
            if backfill:
                done = buffer.push(current)
                if done is not None:
                    finalise_frame(done)
            else:
                finalise_frame(current)
            stats.frames += 1
            if progress:
                progress(i + 1, total)
        for bf in buffer.drain():
            finalise_frame(bf)
        info = reader.info
    tracks, events = manager.all_tracks(), list(manager.events)
    if text:
        tracks = sorted(tracks + text.all_tracks(), key=lambda t: t.id)
    if collector:
        tracks, links = link_tracks(tracks, collector.embeddings, cfg)  # text tracks have no embedding
        events += [IdEvent(frame=ln.frame, kind="reid_link", kept=ln.kept, dropped=ln.merged) for ln in links]
    stats.loop_seconds = time.perf_counter() - start
    if stats.frames == 0:
        raise FaceGuardError(f"No frames could be decoded from {src}.")
    return Analysis(info=info, tracks=tracks, events=events, stats=stats)


def analyse_video(
    src: str | Path, cfg: Config, progress: Progress | None = None, on_frame: FrameHook | None = None
) -> Analysis:
    """Pass 1: find and track every face. Keeps boxes only, never frames.

    With `smooth_window > 1` the stored boxes are temporally smoothed
    (core/smoothing.py), so preview and render both see the smoothed masks.
    """
    analysis = _run(src, cfg, progress, on_frame=on_frame)
    analysis.tracks = smooth_tracks(analysis.tracks, cfg.smooth_window)
    return analysis


def boxes_at(tracks: list[Track], frame: int, selected: set[int] | None = None) -> list[Box]:
    """Face boxes to redact on one frame: tracks marked redact (and in `selected`, if given)."""
    return [
        t.history[frame]
        for t in tracks
        if t.redact and t.source != TEXT_SOURCE and (selected is None or t.id in selected) and frame in t.history
    ]


def text_boxes_at(tracks: list[Track], frame: int) -> list[Box]:
    """Text boxes to redact on one frame (rectangular masks, see core/text.py)."""
    return [t.history[frame] for t in tracks if t.redact and t.source == TEXT_SOURCE and frame in t.history]


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
                writer.write(render_frame(frame, boxes, text_boxes_at(tracks, i), cfg))
                stats.redact_seconds += time.perf_counter() - t0
                stats.faces_redacted += len(boxes)
                stats.frames += 1
                if progress:
                    progress(i + 1, total)
    stats.loop_seconds = time.perf_counter() - start
    stats.warnings += finalise(writer.path, dst, src if cfg.keep_audio else None, cfg.voice, cfg.voice_pitch)
    stats.total_seconds = time.perf_counter() - start
    return stats


def process_video(
    src: str | Path, dst: str | Path, cfg: Config, progress: Progress | None = None
) -> Analysis:
    """Analyse and render in a single pass, redacting every track (CLI path).

    Smoothing needs future frames, so with `smooth_window > 1` this falls back
    to the two-pass analyse + render.
    """
    if cfg.smooth_window > 1:
        start = time.perf_counter()
        analysis = analyse_video(src, cfg, progress)
        r = render_video(src, dst, analysis.tracks, cfg, progress)
        s = analysis.stats
        s.loop_seconds += r.loop_seconds
        s.redact_seconds, s.faces_redacted = r.redact_seconds, r.faces_redacted
        s.warnings += r.warnings
        s.total_seconds = time.perf_counter() - start
        return analysis
    with VideoReader(src) as probe:
        info = probe.info
    start = time.perf_counter()
    redact_time = 0.0
    faces = 0
    with VideoWriter(dst, info) as writer:

        def write(i: int, frame: np.ndarray, boxes: list[Box], texts: list[Box]) -> None:
            nonlocal redact_time, faces
            t0 = time.perf_counter()
            writer.write(render_frame(frame, boxes, texts, cfg))
            redact_time += time.perf_counter() - t0
            faces += len(boxes)

        analysis = _run(src, cfg, progress, on_frame=None, on_final=write)
    analysis.stats.redact_seconds = redact_time
    analysis.stats.faces_redacted = faces
    analysis.stats.warnings += finalise(writer.path, dst, src if cfg.keep_audio else None, cfg.voice, cfg.voice_pitch)
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
                "flags": {str(f): v for f, v in sorted(t.flags.items())},
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
                flags={int(f): v for f, v in d.get("flags", {}).items()},
            )
        )
    return tracks


def save_tracks(path: str | Path, tracks: list[Track], cfg: Config | None = None) -> None:
    Path(path).write_text(json.dumps(tracks_to_json(tracks, cfg)))
