"""Chunked parallel analysis for long videos.

Detection and tracking are sequential within a shot, but independent across
far-apart parts of a video. The frame range is split into one contiguous
chunk per worker process; each chunk is analysed exactly as a short video
would be (it starts with a detection on its first frame), and the tracks
are then stitched across each boundary:

- every chunk after the first starts `parallel_warmup` frames early, so a
  face already being tracked at the boundary is also being tracked by the
  next chunk (a cold start in a hard segment, where the detector misses,
  would otherwise expose it);
- tracks of both chunks with a box on the last frame of chunk k are the
  same face if those boxes overlap by IoU >= `iou_threshold`;
- matched pairs are merged under the earlier ID, everything else is
  renumbered so IDs stay unique.

Each process runs single-threaded OpenCV to avoid oversubscribing cores.
Results differ from a sequential run only near boundaries (the detection
schedule restarts there, D-036).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2

from core.association import IdEvent, greedy_match
from core.pipeline import Analysis, Progress, RunStats, _run
from core.types import Config, Track
from core.video_io import VideoReader

MIN_CHUNK_FRAMES = 150  # below this, process start-up and the boundary detections cost more than they save


def chunks(n_frames: int, workers: int, min_chunk: int = MIN_CHUNK_FRAMES) -> list[tuple[int, int]]:
    """Split [0, n_frames) into at most `workers` contiguous ranges of at least `min_chunk` frames."""
    k = max(1, min(workers, n_frames // max(min_chunk, 1)))
    bounds = [round(n_frames * j / k) for j in range(k + 1)]
    return [(bounds[j], bounds[j + 1]) for j in range(k)]


def _worker(args: tuple[str, Config, int, int]) -> Analysis:
    src, cfg, first, stop = args
    cv2.setNumThreads(1)
    return _run(src, cfg, None, None, first=first, stop=stop)


def _trim(t: Track, start: int) -> Track:
    """Drop a warm-up track's boxes before its chunk's own start."""
    t.history = {f: b for f, b in t.history.items() if f >= start}
    t.flags = {f: v for f, v in t.flags.items() if f >= start}
    return t


def stitch(parts: list[tuple[tuple[int, int], Analysis]], cfg: Config) -> tuple[list[Track], list[IdEvent]]:
    """Renumber tracks and merge those that continue across chunk boundaries.

    Chunk k (after the first) was analysed from `start - warmup`, so its
    tracks already exist on frame `start - 1`, the last frame of chunk k - 1.
    Tracks of both chunks with a box on that frame are matched by IoU; a
    match continues the earlier track. The warm-up boxes themselves are
    dropped: frames before `start` belong to chunk k - 1.
    """
    out: list[Track] = []
    events: list[IdEvent] = []
    next_id = 1
    for (start, stop), a in parts:
        renumber: dict[int, int] = {}
        if start:
            m = start - 1
            ending = [t for t in out if m in t.history and t.source == "detector"]
            warm = [t for t in a.tracks if m in t.history and t.source == "detector"]
            pairs, _, _ = greedy_match([t.history[m] for t in ending], [t.history[m] for t in warm],
                                       cfg.iou_threshold)
            for ei, wi in pairs:
                keep, cont = ending[ei], _trim(warm[wi], start)
                keep.history.update(cont.history)
                keep.flags.update(cont.flags)
                keep.box, keep.last_seen_frame = cont.box, cont.last_seen_frame
                renumber[cont.id] = keep.id
        for t in a.tracks:
            if t.id in renumber:
                continue
            _trim(t, start)
            if not t.history:
                continue  # existed only during the warm-up
            renumber[t.id] = t.id = next_id
            next_id += 1
            out.append(t)
        events += [replace(e, kept=renumber.get(e.kept, e.kept), dropped=renumber.get(e.dropped, e.dropped))
                   for e in a.events if e.frame >= start]
    return sorted(out, key=lambda t: t.id), events


def analyse_parallel(
    src: str | Path, cfg: Config, workers: int | None = None, progress: Progress | None = None
) -> Analysis:
    """Analyse a video in parallel chunks; falls back to one chunk for short clips.

    Smoothing and re-ID linking need the whole video, so they are applied
    after stitching (smoothing) or not available (re-ID raises).
    """
    from core.smoothing import smooth_tracks

    if cfg.reid_link:
        raise ValueError("re-ID linking needs the whole video; use analyse_video for it")
    workers = workers or os.cpu_count() or 1
    with VideoReader(src) as r:
        info = r.info
    ranges = chunks(info.frame_count, workers)
    start_t = time.perf_counter()
    jobs = [(str(src), cfg, max(0, a - cfg.parallel_warmup) if a else 0, b) for a, b in ranges]
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            results = []
            for k, res in enumerate(pool.map(_worker, jobs)):
                results.append(res)
                if progress:
                    progress(ranges[k][1], info.frame_count)
    tracks, events = stitch(list(zip(ranges, results)), cfg)
    stats = RunStats(frames=info.frame_count)
    for a in results:
        for f in ("detect_seconds", "track_seconds", "detector_calls", "early_detections",
                  "scene_cuts", "backfilled_frames", "text_seconds"):
            setattr(stats, f, getattr(stats, f) + getattr(a.stats, f))
        stats.warnings += [w for w in a.stats.warnings if w not in stats.warnings]
    stats.loop_seconds = time.perf_counter() - start_t
    return Analysis(info=info, tracks=smooth_tracks(tracks, cfg.smooth_window), events=events, stats=stats)
