"""Evaluation metrics: redaction coverage, leak frames, detection P/R/F1, ID consistency.

    python eval/metrics.py --pred out/preds.json --truth data/ground_truth.json

Coverage is computed by rasterising the same `build_mask` the renderer uses,
so the metric measures exactly what the exported video shows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import tracks_from_json  # noqa: E402
from core.redact import build_mask  # noqa: E402
from core.types import Box, Config, Track  # noqa: E402
from core.video_io import VideoReader  # noqa: E402

COVERED_ALPHA = 0.5  # a pixel counts as masked at or above this alpha
LEAK_COVERAGE = 0.90  # a face below this coverage is a leak
MATCH_IOU = 0.5  # detection / identity matching threshold

GtFrames = dict[int, list[tuple[int, Box]]]  # frame -> [(person id, box)]


def load_video_truth(entry: dict, n_frames: int | None = None) -> GtFrames:
    """Ground-truth boxes per frame, linearly interpolating between keyframes.

    Interpolation only fills gaps between two keyframes where the same person
    id appears in both; a person missing from a keyframe is treated as absent.
    """
    keys = {int(f): [(g["id"], Box(*g["box"])) for g in faces] for f, faces in entry["frames"].items()}
    if not entry.get("interpolate"):
        return keys
    out: GtFrames = {}
    frames = sorted(keys)
    for a, b in zip(frames, frames[1:]):
        out[a] = keys[a]
        b_by_id = dict(keys[b])
        for f in range(a + 1, b):
            u = (f - a) / (b - a)
            out[f] = [
                (pid, _lerp(box, b_by_id[pid], u)) for pid, box in keys[a] if pid in b_by_id
            ]
    out[frames[-1]] = keys[frames[-1]]
    return out


def _lerp(a: Box, b: Box, u: float) -> Box:
    return Box.from_xywh(*(x + (y - x) * u for x, y in zip(a.as_list(), b.as_list())))


def coverage(gt: Box, preds: list[Box], frame_w: int, frame_h: int, cfg: Config) -> float:
    """Fraction of the ground-truth face box under the union of applied masks."""
    g = gt.clip(frame_w, frame_h)
    if g.area == 0:
        return 1.0
    alpha = np.zeros((g.h, g.w), np.float32)
    for p in preds:
        if p.area == 0:
            continue
        roi, a = build_mask(p, frame_w, frame_h, cfg)
        x0, y0 = max(g.x, roi.x), max(g.y, roi.y)
        x1, y1 = min(g.x + g.w, roi.x + roi.w), min(g.y + g.h, roi.y + roi.h)
        if x1 <= x0 or y1 <= y0:
            continue
        window = alpha[y0 - g.y : y1 - g.y, x0 - g.x : x1 - g.x]
        np.maximum(window, a[y0 - roi.y : y1 - roi.y, x0 - roi.x : x1 - roi.x], out=window)
    return float((alpha >= COVERED_ALPHA).mean())


@dataclass
class VideoMetrics:
    frames_with_faces: int = 0
    faces: int = 0  # face-frames
    mean_coverage: float = 0.0
    leak_frames: int = 0
    leak_frame_ids: list[int] = field(default_factory=list)
    leak_frames_by_segment: dict[str, int] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    id_consistency: float = 0.0
    id_switches: int = 0  # times a GT person's matched track id changed
    tracks: int = 0


def evaluate_video(
    tracks: list[Track], truth: GtFrames, frame_w: int, frame_h: int, cfg: Config,
    segments: dict[str, list[int]] | None = None,
) -> VideoMetrics:
    """Per-frame coverage and leaks, per-frame box P/R/F1, and identity metrics."""
    m = VideoMetrics(tracks=len(tracks))
    pred_by_frame: dict[int, list[tuple[int, Box]]] = defaultdict(list)
    for t in tracks:
        if t.redact:
            for f, b in t.history.items():
                pred_by_frame[f].append((t.id, b))

    covs: list[float] = []
    tp = fp = fn = 0
    person_track_frames: dict[int, Counter] = defaultdict(Counter)
    track_person_frames: dict[int, Counter] = defaultdict(Counter)
    last_track_for_person: dict[int, int] = {}

    for f in sorted(truth):
        gts = truth[f]
        preds = pred_by_frame.get(f, [])
        pred_boxes = [b for _, b in preds]
        if gts:
            m.frames_with_faces += 1
        leaked = False
        for _, g in gts:
            c = coverage(g, pred_boxes, frame_w, frame_h, cfg)
            covs.append(c)
            leaked |= c < LEAK_COVERAGE
        if leaked:
            m.leak_frame_ids.append(f)

        # one-to-one IoU matching for P/R and identity
        used: set[int] = set()
        for pid, g in gts:
            best, best_iou = None, MATCH_IOU
            for k, (tid, b) in enumerate(preds):
                if k not in used and g.iou(b) >= best_iou:
                    best, best_iou = k, g.iou(b)
            if best is None:
                fn += 1
                continue
            used.add(best)
            tp += 1
            tid = preds[best][0]
            person_track_frames[pid][tid] += 1
            track_person_frames[tid][pid] += 1
            if pid in last_track_for_person and last_track_for_person[pid] != tid:
                m.id_switches += 1
            last_track_for_person[pid] = tid
        fp += len(preds) - len(used)

    m.faces = len(covs)
    m.mean_coverage = float(np.mean(covs)) if covs else 1.0
    m.leak_frames = len(m.leak_frame_ids)
    m.precision = tp / (tp + fp) if tp + fp else 0.0
    m.recall = tp / (tp + fn) if tp + fn else 0.0
    m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall) if m.precision + m.recall else 0.0
    # ID consistency (EVALUATION 3.5): each track is assigned the person it
    # overlaps most; frames where it overlaps someone else count against it.
    matched = sum(sum(c.values()) for c in track_person_frames.values())
    majority = sum(max(c.values()) for c in track_person_frames.values())
    m.id_consistency = majority / matched if matched else 1.0
    if segments:
        m.leak_frames_by_segment = {
            name: sum(1 for f in m.leak_frame_ids if lo <= f < hi) for name, (lo, hi) in segments.items()
        }
    return m


def evaluate_files(pred_path: str | Path, truth_path: str | Path, video: str | None = None) -> VideoMetrics:
    preds = json.loads(Path(pred_path).read_text())
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in (preds.get("config") or {}).items()})
    truth_doc = json.loads(Path(truth_path).read_text())
    entries = truth_doc["videos"]
    entry = next((e for e in entries if video is None or Path(e["file"]).name == Path(video).name), None)
    if entry is None:
        raise SystemExit(f"No ground truth for video {video!r} in {truth_path}")
    video_path = Path(truth_path).parent / entry["file"]
    if not video_path.exists():
        video_path = ROOT / "data" / entry["file"]
    with VideoReader(video_path) as r:
        w, h = r.info.width, r.info.height
    return evaluate_video(
        tracks_from_json(preds), load_video_truth(entry), w, h, cfg, entry.get("segments")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True, help="tracks JSON from cli.py --save-tracks")
    ap.add_argument("--truth", required=True, help="ground-truth JSON")
    ap.add_argument("--video", help="which video in the truth file (default: first)")
    args = ap.parse_args()
    m = evaluate_files(args.pred, args.truth, args.video)
    print(json.dumps({k: v for k, v in m.__dict__.items() if k != "leak_frame_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
