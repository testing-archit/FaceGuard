"""Run every experiment in docs/EVALUATION.md and regenerate the results.

    python eval/run_experiments.py --config eval/experiments.yaml

Writes eval/out/raw_results.json, the plots in eval/out/, and the generated
block of docs/RESULTS.md (between the BEGIN/END GENERATED markers; the
hand-recorded milestone notes below it are left alone).

Datasets (all optional except the ones present on disk):
- videos: any ground-truth file with a "videos" list (the synthetic clips
  carry exact per-frame boxes; hand-annotated clips are interpolated)
- images: hand-annotated images from data/ground_truth.json, plus stills
  sampled from a synthetic clip (exact boxes, condition = segment name)

Timings are the processing loop (decode + detect/track + redact + write),
mean and sample std over `runs` repeats, as in the M2/M3 notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.detectors import make_detector  # noqa: E402
from core.pipeline import process_video  # noqa: E402
from core.smoothing import jitter_px, smooth_tracks  # noqa: E402
from core.types import Box, Config, Track  # noqa: E402
from core.video_io import VideoReader  # noqa: E402
from eval.metrics import (  # noqa: E402
    DetectionCounts,
    GtFrames,
    VideoMetrics,
    count_detections,
    coverage,
    evaluate_video,
    load_video_truth,
    mask_area_ratio,
    pr_curve,
    size_bucket,
)

BEGIN = "<!-- BEGIN GENERATED: eval/run_experiments.py -->"
END = "<!-- END GENERATED -->"
DEFAULTS = {
    "runs": 3,
    "out": "eval/out",
    "results_md": "docs/RESULTS.md",
    "base_config": {},
    "experiments": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15", "e16"],
    "videos": [],
    "images": {},
    "e1_detectors": ["yunet", "haar"],
    "e2_intervals": [1, 5, 10, 15, 30],
    "e3_interval": 10,
    "e3_trackers": ["lk", "csrt", "kcf", "none", "off"],
    "e4_thresholds": [0.3, 0.5, 0.6, 0.7, 0.9],
    "e6_paddings": [0.0, 0.1, 0.2, 0.25, 0.3],
    "e7_unconfirmed_intervals": [1, 3],
    "e8_paddings": [0.0, 0.1, 0.2, 0.25],
    "e9_windows": [1, 3, 5, 9],
    "e10_trackers": ["lk", "csrt", "kcf"],
    "e11_tile_grids": [2, 3],
    "e12_backfill": [0, 5, 10, 20],
    "e14_trackers": ["lk", "kcf"],
    "e14_variants": [{"label": "baseline"}, {"label": "Hungarian association", "association": "hungarian"},
                     {"label": "re-ID linking", "reid_link": True},
                     {"label": "coast, velocity", "coast": True},
                     {"label": "coast, Kalman", "coast": True, "coast_model": "kalman"}],
    "e13_settings": [{"label": "blur (default 0.35)", "mode": "blur"},
                     {"label": "pixelate (default 8 blocks)", "mode": "pixelate"},
                     {"label": "fill", "mode": "fill"}],
}

# E7 refresh policies, cumulative: each adds one early-detection rule (D-017).
def e7_policies(unconfirmed: list[int]) -> list[tuple[str, dict]]:
    return (
        [("scheduled only", {"refresh_on_lost": False, "unconfirmed_interval": 0}),
         ("+ lost track", {"refresh_on_lost": True, "unconfirmed_interval": 0})]
        + [(f"+ unconfirmed, every {n}", {"refresh_on_lost": True, "unconfirmed_interval": n}) for n in unconfirmed]
    )
PR_FLOOR = 0.05  # YuNet threshold for the PR sweep; every operating point above it is reachable


# --- datasets --------------------------------------------------------------


@dataclass
class VideoSet:
    name: str
    path: Path
    truth: GtFrames
    width: int
    height: int
    frames: int
    segments: dict[str, list[int]]
    synthetic: bool


@dataclass
class ImageItem:
    name: str
    image: np.ndarray
    gts: list[Box]
    conditions: list[str] = field(default_factory=list)
    source: str = "annotated"
    ignore: list[Box] = field(default_factory=list)  # regions too ambiguous to label


def resolve(truth_path: Path, file: str) -> Path:
    for base in (truth_path.parent, ROOT / "data"):
        p = base / file
        if p.exists():
            return p
    return truth_path.parent / file


def load_videos(spec: list[dict]) -> list[VideoSet]:
    out = []
    for item in spec:
        tp = ROOT / item["truth"]
        if not tp.exists():
            print(f"skip videos in {item['truth']}: file not found", file=sys.stderr)
            continue
        for entry in json.loads(tp.read_text()).get("videos", []):
            path = resolve(tp, entry["file"])
            if not path.exists():
                print(f"skip {entry['file']}: video not found", file=sys.stderr)
                continue
            with VideoReader(path) as r:
                info = r.info
            out.append(VideoSet(
                name=path.stem, path=path, truth=load_video_truth(entry),
                width=info.width, height=info.height, frames=info.frame_count,
                segments=entry.get("segments", {}), synthetic=bool(entry.get("synthetic")),
            ))
    return out


def load_images(spec: dict) -> list[ImageItem]:
    items: list[ImageItem] = []
    for tfile in spec.get("annotated", []):
        tp = ROOT / tfile
        if not tp.exists():
            continue
        for entry in json.loads(tp.read_text()).get("images", []):
            img = cv2.imread(str(resolve(tp, entry["file"])), cv2.IMREAD_COLOR)
            if img is None:
                print(f"skip image {entry['file']}: not readable", file=sys.stderr)
                continue
            items.append(ImageItem(
                name=Path(entry["file"]).name, image=img,
                gts=[Box(*f["box"]) for f in entry["faces"]],
                conditions=entry.get("conditions", []),
                ignore=[Box(*b) for b in entry.get("ignore", [])],
            ))
    st = spec.get("synthetic_stills")
    if st and (ROOT / st["truth"]).exists():
        tp = ROOT / st["truth"]
        entry = json.loads(tp.read_text())["videos"][0]
        truth = load_video_truth(entry)
        segs = entry.get("segments", {})
        every = int(st.get("every", 15))
        with VideoReader(resolve(tp, entry["file"])) as r:
            for i, frame in enumerate(r):
                if i % every:
                    continue
                seg = next((n for n, (lo, hi) in segs.items() if lo <= i < hi), "other")
                items.append(ImageItem(
                    name=f"{Path(entry['file']).stem}#{i}", image=frame,
                    gts=[b for _, b in truth.get(i, [])], conditions=[seg], source="synthetic",
                ))
    return items


# --- environment -----------------------------------------------------------


def environment(cfg: Config) -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    model = ROOT / cfg.model_path
    sha = hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else "missing"
    ffmpeg = "absent"
    try:
        ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    except OSError:
        pass
    return {
        "date": date.today().isoformat(),
        "cpu": cpu,
        "cores": __import__("os").cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "model": Path(cfg.model_path).name,
        "model_sha256": sha,
        "ffmpeg": ffmpeg,
    }


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


# --- runs ------------------------------------------------------------------


@dataclass
class VideoRun:
    fps_runs: list[float]
    metrics: VideoMetrics
    tracks: list[Track]
    detector_calls: int
    early_detections: int
    events: int
    backfilled: int = 0
    reid_links: int = 0

    @property
    def fps(self) -> float:
        return statistics.mean(self.fps_runs)

    @property
    def fps_std(self) -> float:
        return statistics.stdev(self.fps_runs) if len(self.fps_runs) > 1 else 0.0


class Runner:
    """Runs (video, config) pairs once and caches them: E2 and E3 share points."""

    def __init__(self, base: Config, runs: int, tmp: Path) -> None:
        self.base, self.runs, self.tmp = base, runs, tmp
        self._cache: dict[tuple, VideoRun] = {}

    def run(self, v: VideoSet, **overrides) -> VideoRun:
        cfg = replace(self.base, **overrides)
        if cfg.tracker == "off":
            cfg = replace(cfg, detect_interval=1)
        key = (v.name, repr(cfg))
        if key in self._cache:
            return self._cache[key]
        fps, analysis = [], None
        for _ in range(self.runs):
            analysis = process_video(v.path, self.tmp / "out.mp4", cfg)
            fps.append(analysis.stats.fps)
        m = evaluate_video(analysis.tracks, v.truth, v.width, v.height, cfg, v.segments)
        res = VideoRun(fps, m, analysis.tracks, analysis.stats.detector_calls,
                       analysis.stats.early_detections, len(analysis.events), analysis.stats.backfilled_frames,
                       sum(e.kind == "reid_link" for e in analysis.events))
        self._cache[key] = res
        label = f"{cfg.detector}/{cfg.tracker}/{cfg.detect_interval}"
        if cfg.tracker != "off" and (not cfg.refresh_on_lost or cfg.unconfirmed_interval != self.base.unconfirmed_interval):
            label += f"/lost={int(cfg.refresh_on_lost)},unc={cfg.unconfirmed_interval}"
        print(f"  {v.name:<18} {label:<16} {res.fps:7.1f} FPS  leak {m.leak_frames:4d}  "
              f"cov {m.mean_coverage:.4f}  tracks {m.tracks}", flush=True)
        return res


def video_row(r: VideoRun) -> dict:
    m = r.metrics
    return {
        "fps_runs": [round(f, 2) for f in r.fps_runs], "fps": round(r.fps, 2), "fps_std": round(r.fps_std, 2),
        "detector_calls": r.detector_calls, "early_detections": r.early_detections, "leak_frames": m.leak_frames,
        "leak_frames_by_segment": m.leak_frames_by_segment, "mean_coverage": round(m.mean_coverage, 5),
        "precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4),
        "id_consistency": round(m.id_consistency, 4), "id_switches": m.id_switches,
        "tracks": m.tracks, "merge_events": r.events, "backfilled_frames": r.backfilled, "reid_links": r.reid_links,
        "mean_track_frames": round(statistics.mean([len(t.history) for t in r.tracks]), 1) if r.tracks else 0,
    }


# --- experiments -----------------------------------------------------------


def e1_images(images: list[ImageItem], base: Config, detectors: list[str]) -> dict:
    """E1: detectors on still images. P/R/F1, speed, per-condition and per-size recall."""
    out = {}
    for name in detectors:
        cfg = replace(base, detector=name)
        det = make_detector(cfg)
        total = DetectionCounts()
        by_group: dict[str, DetectionCounts] = defaultdict(DetectionCounts)
        cov_by_group: dict[str, list[float]] = defaultdict(list)
        covs: list[float] = []
        secs = 0.0
        for item in images:
            t0 = time.perf_counter()
            preds = [d.box for d in det.detect(item.image)]
            secs += time.perf_counter() - t0
            total.add(count_detections(item.gts, preds, ignore=item.ignore))
            h, w = item.image.shape[:2]
            for g in item.gts:
                c = coverage(g, preds, w, h, cfg)
                covs.append(c)
                groups = [f"cond:{c_}" for c_ in item.conditions] + [f"size:{size_bucket(g)}", f"source:{item.source}"]
                hit = count_detections([g], preds)
                for grp in groups:
                    by_group[grp].add(DetectionCounts(tp=hit.tp, fn=hit.fn))
                    cov_by_group[grp].append(c)
        out[name] = {
            "images": len(images), "faces": total.tp + total.fn,
            "tp": total.tp, "fp": total.fp, "fn": total.fn,
            "precision": round(total.precision, 4), "recall": round(total.recall, 4), "f1": round(total.f1, 4),
            "mean_coverage": round(float(np.mean(covs)), 4) if covs else None,
            "ms_per_image": round(1000 * secs / max(len(images), 1), 2),
            "groups": {
                g: {"faces": c.tp + c.fn, "recall": round(c.recall, 4),
                    "coverage": round(float(np.mean(cov_by_group[g])), 4)}
                for g, c in sorted(by_group.items())
            },
        }
        print(f"  E1 {name}: P {total.precision:.3f} R {total.recall:.3f} F1 {total.f1:.3f}", flush=True)
    return out


def e4_pr(images: list[ImageItem], base: Config, thresholds: list[float]) -> dict:
    """E4: YuNet confidence sweep. One low-threshold pass gives the whole curve."""
    det = make_detector(replace(base, detector="yunet", conf_threshold=PR_FLOOR))
    per_image = [(item.gts, [(d.box, d.score) for d in det.detect(item.image)], item.ignore) for item in images]
    scores, prec, rec, ap = pr_curve(per_image)
    table = []
    for t in thresholds:
        c = DetectionCounts()
        for g, preds, ign in per_image:
            c.add(count_detections(g, [b for b, s in preds if s >= t], ignore=ign))
        table.append({"threshold": t, "precision": round(c.precision, 4), "recall": round(c.recall, 4),
                      "f1": round(c.f1, 4), "fp": c.fp, "fn": c.fn})
    return {"ap": round(ap, 4), "table": table,
            "curve": {"score": scores.round(4).tolist(), "precision": prec.round(4).tolist(),
                      "recall": rec.round(4).tolist()}}


def e9_smoothing(v: VideoSet, run: VideoRun, base: Config, windows: list[int]) -> list[dict]:
    """E9: same tracks smoothed over different windows: steadier masks vs coverage."""
    rows = []
    for w in windows:
        tracks = smooth_tracks(run.tracks, w)
        m = evaluate_video(tracks, v.truth, v.width, v.height, base, v.segments)
        rows.append({"window": w, "jitter_px": round(jitter_px(tracks), 3), "leak_frames": m.leak_frames,
                     "mean_coverage": round(m.mean_coverage, 5), "leak_frames_by_segment": m.leak_frames_by_segment})
    return rows


def e6_padding(v: VideoSet, run: VideoRun, base: Config, paddings: list[float]) -> list[dict]:
    """E6: same tracks, different padding. Coverage vs how much extra area is hidden."""
    rows = []
    for p in paddings:
        cfg = replace(base, padding=p)
        m = evaluate_video(run.tracks, v.truth, v.width, v.height, cfg, v.segments)
        sample = [t.history[f] for t in run.tracks for f in list(t.history)[::25]]
        rows.append({"padding": p, "mean_coverage": round(m.mean_coverage, 5), "leak_frames": m.leak_frames,
                     "mask_area_ratio": round(mask_area_ratio(sample, v.width, v.height, cfg), 3)})
    return rows


# --- E16 degradations ------------------------------------------------------
# Each takes a BGR frame (and a seeded RNG) and returns the degraded frame.
DARK_GAIN = 0.3  # low light: keep 30% of the brightness ...
DARK_NOISE = 6.0  # ... plus sensor noise, std in grey levels
JPEG_QUALITY = 12  # heavy compression: blocking and ringing
MOTION_BLUR_PX = 11  # horizontal motion-blur kernel length
DOWNSCALE = 0.5  # resolution halved; faces are half as wide


def _low_light(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = f.astype(np.float32) * DARK_GAIN + rng.normal(0, DARK_NOISE, f.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _jpeg(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1], cv2.IMREAD_COLOR)


def _motion_blur(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = np.zeros((MOTION_BLUR_PX, MOTION_BLUR_PX), np.float32)
    k[MOTION_BLUR_PX // 2, :] = 1.0 / MOTION_BLUR_PX
    return cv2.filter2D(f, -1, k)


def _downscale(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return cv2.resize(f, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)


DEGRADATIONS = {"low light": _low_light, "JPEG q12": _jpeg, "motion blur": _motion_blur, "half resolution": _downscale}


def degrade_video(v: VideoSet, name: str, tmp: Path) -> VideoSet:
    """Write a degraded copy of a clip and scale its ground truth to match."""
    from core.video_io import VideoWriter, VideoInfo

    fn = DEGRADATIONS[name]
    rng = np.random.default_rng(0)
    out = tmp / f"{v.name}-{name.replace(' ', '_')}.silent.mp4"
    scale = DOWNSCALE if fn is _downscale else 1.0
    with VideoReader(v.path) as r:
        info = r.info
        w, h = round(info.width * scale), round(info.height * scale)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
        for frame in r:
            writer.write(fn(frame, rng))
        writer.release()
    truth = {f: [(pid, Box.from_xywh(*(c * scale for c in b.as_list()))) for pid, b in faces]
             for f, faces in v.truth.items()}
    return VideoSet(name=f"{v.name} / {name}", path=out, truth=truth, width=w, height=h, frames=v.frames,
                    segments=v.segments, synthetic=v.synthetic)


def e16_robustness(videos: list[VideoSet], runner: Runner, n: int) -> dict:
    """E16: default pipeline on degraded copies of each clip (low light, compression, blur, resolution)."""
    out = {}
    for v in videos:
        rows = [{"condition": "original", **video_row(runner.run(v, tracker="lk", detect_interval=n))}]
        for name in DEGRADATIONS:
            dv = degrade_video(v, name, runner.tmp)
            rows.append({"condition": name, **video_row(runner.run(dv, tracker="lk", detect_interval=n))})
            off = runner.run(dv, tracker="off")
            rows[-1]["leak_frames_per_frame_detection"] = off.metrics.leak_frames
            rows[-1]["recall_per_frame_detection"] = round(off.metrics.recall, 4)
        out[v.name] = rows
    return out


SFACE_MATCH = 0.363  # OpenCV's SFace same-person cosine threshold


def e13_output_check(images: list[ImageItem], base: Config, settings: list[dict]) -> dict:
    """E13: attack the redacted output. Can a detector still find, and a recogniser still match, the face?

    For every annotated photo: detect faces with the default pipeline, redact
    them with each setting, then on the *output*
    - detectable: YuNet at 0.3 / 0.6 still fires inside a hidden region (core/verify.py);
    - matchable: SFace cosine similarity between the original and the
      redacted face (aligned with the original landmarks, the attacker's best
      case) is above OpenCV's same-person threshold 0.363;
    - re-identified: in a closed set of every original face in the dataset,
      the redacted face's nearest neighbour is its own original (rank-1).
    """
    from core.pipeline import render_image
    from core.recognition import FaceMatcher, available
    from core.verify import check_frame

    det = make_detector(base)
    v_lo = make_detector(replace(base, detector="yunet", conf_threshold=0.3, tile_grid=0))
    v_hi = make_detector(replace(base, detector="yunet", conf_threshold=0.6, tile_grid=0))
    matcher = FaceMatcher(base) if available(base) else None
    faces = [(item.image, det.detect(item.image)) for item in images]
    gallery = None
    if matcher:
        gallery = [(k, j, matcher.embed(img, d)) for k, (img, ds) in enumerate(faces) for j, d in enumerate(ds)]
        gallery = [g for g in gallery if g[2] is not None]
    rows = []
    for st in settings:
        cfg = replace(base, **{k: v for k, v in st.items() if k != "label"})
        n = lo = hi = match = reid = 0
        sims: list[float] = []
        for k, (img, ds) in enumerate(faces):
            boxes = [d.box for d in ds]
            out = render_image(img, boxes, cfg)
            n += len(boxes)
            lo += len(check_frame(out, boxes, v_lo)[0])
            hi += len(check_frame(out, boxes, v_hi)[0])
            if matcher:
                for j, d in enumerate(ds):
                    e = matcher.embed(out, d)
                    own = next((g for g in gallery if g[0] == k and g[1] == j), None)
                    if e is None or own is None:
                        continue
                    sim = float(e @ own[2])
                    sims.append(sim)
                    match += sim > SFACE_MATCH
                    best = max(gallery, key=lambda g: float(e @ g[2]))
                    reid += best[0] == k and best[1] == j
        rows.append({"setting": st["label"], "faces": n, "detectable_0.3": lo, "detectable_0.6": hi,
                     "mean_similarity": round(float(np.mean(sims)), 3) if sims else None,
                     "matchable": match if matcher else None, "reidentified": reid if matcher else None,
                     "gallery": len(gallery) if gallery else 0})
        print(f"  E13 {st['label']:<28} detectable {lo}/{n} (0.3) {hi}/{n} (0.6)  matchable {match}  rank-1 {reid}",
              flush=True)
    return {"rows": rows}


def e13_videos(videos: list[VideoSet], runner: Runner, n: int) -> dict:
    """E13 on clips: residual face-frames in the exported default-config video."""
    from core.verify import verify_video

    out = {}
    for v in videos:
        r = runner.run(v, tracker="lk", detect_interval=n)
        process_video(v.path, runner.tmp / "verify.mp4", replace(runner.base, tracker="lk", detect_interval=n))
        rep = verify_video(runner.tmp / "verify.mp4", r.tracks, runner.base)
        out[v.name] = {"frames": rep.frames_checked, "hidden_face_frames": rep.hidden_faces,
                       "residuals": len(rep.residuals), "residual_frames": len(rep.residual_frames),
                       "unmasked": len(rep.unmasked)}
    return out


# --- plots -----------------------------------------------------------------


def make_plots(res: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    e2 = res.get("e2", {})
    if e2:
        for key, ylabel, name in (("fps", "Frames per second", "fps_vs_interval.png"),
                                  ("leak_frames", "Leak frames (count)", "leakframes_vs_interval.png")):
            fig, ax = plt.subplots(figsize=(6, 4))
            for vname, d in e2.items():
                xs = [r["interval"] for r in d["lk"]]
                ys = [r[key] for r in d["lk"]]
                err = [r["fps_std"] for r in d["lk"]] if key == "fps" else None
                ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=f"{vname} (LK)")
                ax.axhline(d["off"][key], linestyle="--", linewidth=1, color=ax.lines[-1].get_color(),
                           label=f"{vname} per-frame, no tracking")
            ax.set_xlabel("Detection interval N (frames)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            save(fig, name)

    e4 = res.get("e4")
    if e4 and e4["curve"]["recall"]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(e4["curve"]["recall"], e4["curve"]["precision"], label=f"YuNet, synthetic stills (AP {e4['ap']:.3f})")
        for row in e4["table"]:
            ax.plot(row["recall"], row["precision"], "o", color="C0")
            ax.annotate(f"{row['threshold']}", (row["recall"], row["precision"]), fontsize=7,
                        xytext=(4, -10), textcoords="offset points")
        haar = res.get("e1", {}).get("haar")
        if haar:
            ax.plot(haar["recall"], haar["precision"], "s", color="C3", label="Haar, synthetic stills")
        real = res.get("e4_real")
        if real and real["curve"]["recall"]:
            ax.plot(real["curve"]["recall"], real["curve"]["precision"], color="C2",
                    label=f"YuNet, real photos (AP {real['ap']:.3f})")
            rh = res.get("e1_real", {}).get("haar")
            if rh:
                ax.plot(rh["recall"], rh["precision"], "D", color="C1", label="Haar, real photos")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        save(fig, "pr_curve.png")

    e1 = res.get("e1", {})
    if e1:
        groups = sorted({g for d in e1.values() for g in d["groups"] if not g.startswith("source:")})
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(groups)), 4))
        width = 0.8 / len(e1)
        for k, (det, d) in enumerate(e1.items()):
            xs = np.arange(len(groups)) + k * width
            ax.bar(xs, [d["groups"].get(g, {}).get("recall", 0) for g in groups], width, label=det)
        ax.set_xticks(np.arange(len(groups)) + width * (len(e1) - 1) / 2)
        ax.set_xticklabels([g.split(":", 1)[1] for g in groups], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Recall (IoU >= 0.5)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(e1), frameon=False)
        save(fig, "condition_breakdown.png")

    e6 = res.get("e6", {})
    if e6:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, (vname, rows) in enumerate(e6.items()):
            ax.plot([r["padding"] for r in rows], [r["mean_coverage"] for r in rows], marker="o", color=f"C{k}",
                    label=f"{vname}, ellipse")
            rect = res.get("e8", {}).get(vname, {}).get("rect")
            if rect:
                ax.plot([r["padding"] for r in rect], [r["mean_coverage"] for r in rect], marker="s",
                        linestyle="--", color=f"C{k}", label=f"{vname}, rectangle")
        ax.axvline((2**0.5 - 1) / 2, linestyle=":", color="grey", label="(√2−1)/2 ≈ 0.207")
        ax.set_xlabel("Padding per side (fraction of box)")
        ax.set_ylabel("Mean face coverage")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "padding_sweep.png")
    return written


# --- report ----------------------------------------------------------------


def table(headers: list[str], rows: list[list]) -> str:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
        return str(x)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def report(res: dict, exp: dict) -> str:
    env = res["environment"]
    s = [BEGIN, "",
         f"_Generated {env['date']} by `python eval/run_experiments.py --config eval/experiments.yaml`. "
         "Do not edit by hand; edit the script or the config and re-run._", "",
         "## Environment", "",
         table(["", ""], [["CPU", f"{env['cpu']}, {env['cores']} cores"], ["OS", env["os"]],
                          ["Python", env["python"]], ["OpenCV", env["opencv"]],
                          ["Model", f"`{env['model']}` (SHA-256 `{env['model_sha256'][:8]}…{env['model_sha256'][-6:]}`)"],
                          ["ffmpeg", env["ffmpeg"]], ["Peak RSS", f"{res['peak_rss_mb']:.0f} MB"],
                          ["Timing runs", f"{exp['runs']} per configuration, mean ± sample std"]]),
         "",
         "FPS is the processing loop (decode + detect/track + redact + write), "
         "excluding the final ffmpeg re-encode.", ""]

    s += ["## Datasets", ""]
    rows = [[v["name"], "synthetic, exact boxes" if v["synthetic"] else "hand-annotated, interpolated",
             v["frames"], v["face_frames"], ", ".join(v["segments"]) or "-"] for v in res["datasets"]["videos"]]
    s += [table(["Clip", "Ground truth", "Frames", "Face-frames", "Segments"], rows), ""]
    imgs = res["datasets"]["images"]
    s += [f"Images: **{imgs['annotated']}** hand-annotated, **{imgs['synthetic']}** stills sampled from "
          f"the synthetic clip ({imgs['faces']} faces in total).", ""]
    if imgs["annotated"] == 0:
        s += ["> **No annotated real images or clips yet.** Every number below comes from "
              "synthetic data built from one public group photo. Annotate real media with "
              "`eval/annotate.py` (EVALUATION.md section 1) and re-run before quoting these "
              "numbers as the tool's accuracy.", ""]
    else:
        s += ["> **About the real photos:** seven public OpenCV / MIT-CMU test photographs, labelled by an "
              "AI assistant from detector proposals with visual verification (see the provenance note in "
              "`data/ground_truth.json`). Box geometry follows YuNet, which favours YuNet at the IoU 0.5 "
              "match, and the photos are easy. **No real video is annotated**: all video results are on "
              "synthetic clips. Treat these as indicative, not as field accuracy.", ""]

    if "headline" in res:
        s += ["## Headline", ""] + [f"- {h}" for h in res["headline"]] + [""]

    if res.get("e1") or res.get("e1_real"):
        s += ["## E1 — YuNet vs Haar on images", ""]
        rows = [[src, d, r["images"], r["faces"], r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["f1"],
                 r["mean_coverage"], r["ms_per_image"]]
                for src, key in (("real photos", "e1_real"), ("synthetic stills", "e1"))
                for d, r in res.get(key, {}).items()]
        s += [table(["Set", "Detector", "Images", "Faces", "TP", "FP", "FN", "Precision", "Recall", "F1",
                     "Mean coverage", "ms / image"], rows), ""]
    if res.get("e1_video"):
        s += ["Per-frame detection on the clips (no tracking):", ""]
        rows = [[v, d, r["fps"], r["precision"], r["recall"], r["leak_frames"], r["tracks"]]
                for v, dd in res["e1_video"].items() for d, r in dd.items()]
        s += [table(["Clip", "Detector", "FPS", "Precision", "Recall", "Leak frames", "Track IDs"], rows), ""]

    if res.get("e2"):
        s += ["## E2 — detection interval sweep (LK tracking)", "",
              "![FPS vs interval](../eval/out/fps_vs_interval.png)",
              "![Leak frames vs interval](../eval/out/leakframes_vs_interval.png)", ""]
        for v, d in res["e2"].items():
            rows = [["per-frame, no tracking", d["off"]["detector_calls"], f"{d['off']['fps']:.1f} ± {d['off']['fps_std']:.1f}",
                     d["off"]["leak_frames"], d["off"]["mean_coverage"], d["off"]["tracks"]]]
            rows += [[f"every {r['interval']}", r["detector_calls"], f"{r['fps']:.1f} ± {r['fps_std']:.1f}",
                      r["leak_frames"], r["mean_coverage"], r["tracks"]] for r in d["lk"]]
            s += [f"**{v}**", "", table(["Detection", "Detector calls", "FPS", "Leak frames", "Mean coverage",
                                         "Track IDs"], rows), ""]

    if res.get("e3"):
        s += [f"## E3 — tracker comparison (detect every {exp['e3_interval']} frames)", ""]
        for v, d in res["e3"].items():
            rows = [[t, f"{r['fps']:.1f} ± {r['fps_std']:.1f}", r["leak_frames"], r["mean_coverage"],
                     r["id_consistency"], r["id_switches"], r["tracks"], r["mean_track_frames"]] for t, r in d.items()]
            s += [f"**{v}**", "", table(["Tracker", "FPS", "Leak frames", "Mean coverage", "ID consistency",
                                         "ID switches", "Track IDs", "Mean frames / track"], rows), ""]
        s += ["`off` = detect every frame with no memory; `none` = hold each box still between "
              "detections. ID switches count every time a ground-truth person's matched track ID "
              "changes, including re-acquisition after full occlusion.", ""]

    if res.get("e4") or res.get("e4_real"):
        s += ["## E4 — YuNet confidence threshold", "", "![PR curve](../eval/out/pr_curve.png)", ""]
        for label, key in (("Real photos", "e4_real"), ("Synthetic stills", "e4")):
            e4 = res.get(key)
            if not e4:
                continue
            s += [f"**{label}** (average precision {e4['ap']:.3f})", ""]
            rows = [[r["threshold"], r["precision"], r["recall"], r["f1"], r["fp"], r["fn"]] for r in e4["table"]]
            s += [table(["Threshold", "Precision", "Recall", "F1", "FP", "FN"], rows), ""]

    if res.get("e1_real"):
        s += ["## E5 — condition breakdown, real photos", ""]
        dets = list(res["e1_real"])
        groups = sorted(g for g in {g for d in res["e1_real"].values() for g in d["groups"]} if not g.startswith("source:"))
        rows = []
        for g in groups:
            row = [g.split(":", 1)[1] + (" (size)" if g.startswith("size:") else ""),
                   res["e1_real"][dets[0]]["groups"].get(g, {}).get("faces", 0)]
            for d in dets:
                gg = res["e1_real"][d]["groups"].get(g, {})
                row += [gg.get("recall", "-"), gg.get("coverage", "-")]
            rows.append(row)
        s += [table(["Group", "Faces"] + [f"{d} {k}" for d in dets for k in ("recall", "coverage")], rows), ""]
    if res.get("e1"):
        s += ["## E5 — condition breakdown, synthetic stills", "", "![Condition breakdown](../eval/out/condition_breakdown.png)", "",
              "Images, recall at IoU 0.5 and mean mask coverage at the default padding. "
              "Size groups by box width: small < 40 px, large > 100 px.", ""]
        dets = list(res["e1"])
        groups = sorted({g for d in res["e1"].values() for g in d["groups"]})
        rows = []
        for g in groups:
            row = [g.split(":", 1)[1] + (" (size)" if g.startswith("size:") else " (source)" if g.startswith("source:") else ""),
                   res["e1"][dets[0]]["groups"].get(g, {}).get("faces", 0)]
            for d in dets:
                gg = res["e1"][d]["groups"].get(g, {})
                row += [gg.get("recall", "-"), gg.get("coverage", "-")]
            rows.append(row)
        s += [table(["Group", "Faces"] + [f"{d} {k}" for d in dets for k in ("recall", "coverage")], rows), ""]
        if res.get("e3"):
            s += ["Video leak frames by segment:", ""]
            for v, d in res["e3"].items():
                segs = list(next(iter(d.values()))["leak_frames_by_segment"])
                if not segs:
                    continue
                rows = [[t] + [r["leak_frames_by_segment"].get(sg, 0) for sg in segs] for t, r in d.items()]
                s += [f"**{v}**", "", table(["Tracker"] + segs, rows), ""]

    if res.get("e6"):
        s += [f"## E6 — padding sweep (LK, detect every {exp['e3_interval']})", "", "![Padding sweep](../eval/out/padding_sweep.png)", "",
              "Mask area ratio = masked pixels / detector box area (what padding costs).", ""]
        for v, rows_ in res["e6"].items():
            rows = [[r["padding"], r["mean_coverage"], r["leak_frames"], r["mask_area_ratio"]] for r in rows_]
            s += [f"**{v}**", "", table(["Padding", "Mean coverage", "Leak frames", "Mask area ratio"], rows), ""]
    if res.get("e7"):
        s += [f"## E7 — early-detection policy (LK, detect every {exp['e3_interval']})", "",
              "Each row adds one rule for running the detector off schedule (DECISIONS D-017). "
              "The default is the last row with the configured interval "
              f"(`unconfirmed_interval` = {Config().unconfirmed_interval}).", ""]
        for v, rows_ in res["e7"].items():
            segs = list(rows_[0]["leak_frames_by_segment"])
            rows = [[r["policy"], f"{r['fps']:.1f} ± {r['fps_std']:.1f}", r["detector_calls"], r["early_detections"],
                     r["leak_frames"]] + [r["leak_frames_by_segment"].get(sg, 0) for sg in segs] for r in rows_]
            s += [f"**{v}**", "", table(["Policy", "FPS", "Detector calls", "Early", "Leak frames"] + segs, rows), ""]
    if res.get("e8"):
        s += [f"## E8 — mask shape (LK, detect every {exp['e3_interval']})", "",
              "Ellipse vs rectangle at each padding (DECISIONS D-020). The rectangle has no "
              "corner condition, so it reaches full coverage at lower padding, at the cost of "
              "hiding more area.", ""]
        for v, d in res["e8"].items():
            rows = [[shape, r["padding"], r["mean_coverage"], r["leak_frames"], r["mask_area_ratio"]]
                    for shape, rows_ in d.items() for r in rows_]
            s += [f"**{v}**", "", table(["Shape", "Padding", "Mean coverage", "Leak frames", "Mask area ratio"], rows), ""]
    if res.get("e9"):
        s += [f"## E9 — temporal smoothing (LK, detect every {exp['e3_interval']})", "",
              "Jitter = mean magnitude of the second difference of box centres (px). Smoothing is "
              "zero-phase and off by default (D-019).", ""]
        for v, rows_ in res["e9"].items():
            segs = list(rows_[0]["leak_frames_by_segment"])
            rows = [[r["window"], r["jitter_px"], r["leak_frames"], r["mean_coverage"]]
                    + [r["leak_frames_by_segment"].get(sg, 0) for sg in segs] for r in rows_]
            s += [f"**{v}**", "", table(["Window", "Jitter (px)", "Leak frames", "Mean coverage"] + segs, rows), ""]
    if res.get("e10"):
        s += [f"## E10 — coasting lost tracks (detect every {exp['e3_interval']})", "",
              "Freeze = a lost box stays where it was; coast = it keeps its last velocity and its mask "
              "grows (D-023).", ""]
        for v, rows_ in res["e10"].items():
            segs = list(rows_[0]["leak_frames_by_segment"])
            rows = [[r["tracker"], r["lost"], f"{r['fps']:.1f} ± {r['fps_std']:.1f}", r["leak_frames"], r["precision"]]
                    + [r["leak_frames_by_segment"].get(sg, 0) for sg in segs] for r in rows_]
            s += [f"**{v}**", "", table(["Tracker", "When lost", "FPS", "Leak frames", "Precision"] + segs, rows), ""]
    if res.get("e11"):
        s += ["## E11 — tiled detection for small faces (real photos)", "",
              "YuNet on the full frame alone vs also on n x n upscaled tiles (D-022).", ""]
        rows = [[r["tiles"], r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["ms_per_image"]]
                for r in res["e11"]]
        s += [table(["Tiles", "TP", "FP", "FN", "Precision", "Recall", "ms / image"], rows), ""]
    if res.get("e12"):
        s += [f"## E12 — backward backfill of new faces (LK, detect every {exp['e3_interval']})", "",
              "When a face is first detected it is tracked backwards over the last N buffered frames, "
              "so the frames before its first detection are masked too (D-026).", ""]
        for v, rows_ in res["e12"].items():
            segs = list(rows_[0]["leak_frames_by_segment"])
            rows = [[r["backfill"], f"{r['fps']:.1f} ± {r['fps_std']:.1f}", r["backfilled_frames"], r["leak_frames"],
                     r["precision"]] + [r["leak_frames_by_segment"].get(sg, 0) for sg in segs] for r in rows_]
            s += [f"**{v}**", "", table(["Backfill N", "FPS", "Frames filled", "Leak frames", "Precision"] + segs,
                                        rows), ""]
    if res.get("e13"):
        e = res["e13"]
        s += ["## E13 — attacking the redacted output (real photos)", "",
              "Each face is redacted, then the *output* is attacked. *Detectable*: YuNet still fires inside the "
              "hidden region. *Matchable*: SFace similarity between the original and the redacted face is above "
              "OpenCV's same-person threshold 0.363 (aligned with the original landmarks, the attacker's best "
              f"case). *Re-identified*: in a closed set of all {e['rows'][0]['gallery']} original faces, the "
              "redacted face's nearest neighbour is its own original (chance is 1 in the gallery size).", ""]
        rows = [[r["setting"], r["faces"], r["detectable_0.3"], r["detectable_0.6"], r["mean_similarity"],
                 r["matchable"], r["reidentified"]] for r in e["rows"]]
        s += [table(["Setting", "Faces", "Detectable (0.3)", "Detectable (0.6)", "Mean similarity",
                     "Matchable", "Re-identified (rank-1)"], rows), ""]
        if e.get("videos"):
            rows = [[v, d["frames"], d["hidden_face_frames"], d["residuals"], d["residual_frames"]]
                    for v, d in e["videos"].items()]
            s += ["Exported clips at the default configuration, every frame re-scanned with YuNet at 0.3:", "",
                  table(["Clip", "Frames", "Hidden face-frames", "Residual detections", "Frames with a residual"],
                        rows), ""]
    if res.get("e14"):
        s += [f"## E14 — association, re-ID linking and motion models (detect every {exp['e3_interval']})", "",
              "Each variant changes one thing from the default (D-029, D-030, D-031). *Links* = tracks merged by "
              "appearance.", ""]
        for v, rows_ in res["e14"].items():
            rows = [[r["tracker"], r["variant"], f"{r['fps']:.1f} ± {r['fps_std']:.1f}", r["leak_frames"],
                     r["precision"], r["tracks"], r["id_switches"], r["reid_links"]] for r in rows_]
            s += [f"**{v}**", "", table(["Tracker", "Variant", "FPS", "Leak frames", "Precision", "Tracks",
                                         "ID switches", "Links"], rows), ""]
    if res.get("e15"):
        s += ["## E15 — Haar profile-face fallback", "",
              "YuNet alone vs YuNet plus the profile cascade run on the frame and its mirror (D-032).", ""]
        rows = [[r["set"], r["profile"], r["tp"], r["fp"], r["fn"], r["precision"], r["recall"], r["ms_per_image"]]
                for r in res["e15"]]
        s += [table(["Set", "Profile", "TP", "FP", "FN", "Precision", "Recall", "ms / image"], rows), ""]
    if res.get("e16"):
        s += [f"## E16 — robustness to degraded video (LK, detect every {exp['e3_interval']})", "",
              f"Each clip re-encoded with one degradation: low light ({DARK_GAIN:.0%} brightness + noise "
              f"σ {DARK_NOISE:g}), JPEG quality {JPEG_QUALITY} per frame, {MOTION_BLUR_PX} px horizontal motion "
              f"blur, and half resolution (ground truth scaled). *Per-frame* columns are the no-tracking "
              "baseline on the same degraded clip.", ""]
        for v, rows_ in res["e16"].items():
            rows = [[r["condition"], f"{r['fps']:.1f}", r["leak_frames"], round(r["mean_coverage"], 4), r["recall"],
                     r.get("leak_frames_per_frame_detection", "—"), r.get("recall_per_frame_detection", "—")]
                    for r in rows_]
            s += [f"**{v}**", "", table(["Condition", "FPS", "Leak frames", "Mean coverage", "Box recall",
                                         "Leak frames (per-frame)", "Recall (per-frame)"], rows), ""]
    s.append(END)
    return "\n".join(s)


def write_results_md(path: Path, block: str) -> None:
    text = path.read_text() if path.exists() else "# FaceGuard — Results\n\n"
    if BEGIN in text and END in text:
        pre, rest = text.split(BEGIN, 1)
        post = rest.split(END, 1)[1]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def headline(res: dict, interval: int) -> list[str]:
    out = []
    for v, d in res.get("e2", {}).items():
        off = d["off"]
        r = next((r for r in d["lk"] if r["interval"] == interval), None)
        if r is None or off["fps"] <= 0:
            continue
        out.append(
            f"**{v}:** detecting every {interval} frames with Lucas-Kanade tracking ran at "
            f"**{r['fps'] / off['fps']:.2f}x** the throughput of per-frame detection "
            f"({r['fps']:.1f} vs {off['fps']:.1f} FPS) while changing leak frames from "
            f"**{off['leak_frames']}** to **{r['leak_frames']}**."
        )
    return out


# --- main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "eval" / "experiments.yaml"))
    ap.add_argument("--runs", type=int, help="override timing repeats (e.g. 1 for a quick check)")
    ap.add_argument("--out", help="output folder (default from config)")
    ap.add_argument("--results-md", help="RESULTS.md to update (default from config)")
    args = ap.parse_args(argv)

    exp = {**DEFAULTS, **(yaml.safe_load(Path(args.config).read_text()) or {})}
    if args.runs:
        exp["runs"] = args.runs
    out = Path(args.out or ROOT / exp.get("out", "eval/out"))
    out.mkdir(parents=True, exist_ok=True)
    results_md = Path(args.results_md or ROOT / exp.get("results_md", "docs/RESULTS.md"))
    base = Config(**exp.get("base_config", {}))
    todo = set(exp["experiments"])

    videos = load_videos(exp.get("videos", []))
    images = load_images(exp.get("images", {}))
    res: dict = {"environment": environment(base), "config": exp, "datasets": {
        "videos": [{"name": v.name, "synthetic": v.synthetic, "frames": v.frames,
                    "face_frames": sum(len(f) for f in v.truth.values()), "segments": list(v.segments)} for v in videos],
        "images": {"annotated": sum(i.source == "annotated" for i in images),
                   "synthetic": sum(i.source == "synthetic" for i in images),
                   "faces": sum(len(i.gts) for i in images)},
    }}
    print(f"{len(videos)} clip(s), {len(images)} image(s)", flush=True)

    with tempfile.TemporaryDirectory(prefix="faceguard-eval-") as tmp:
        runner = Runner(base, int(exp.get("runs", 3)), Path(tmp))
        real = [i for i in images if i.source == "annotated"]
        synth = [i for i in images if i.source == "synthetic"]
        if images and ({"e1", "e5"} & todo):
            print("E1 images", flush=True)
            if synth:
                res["e1"] = e1_images(synth, base, exp["e1_detectors"])
            if real:
                res["e1_real"] = e1_images(real, base, exp["e1_detectors"])
        if videos and "e1" in todo:
            print("E1 clips", flush=True)
            res["e1_video"] = {v.name: {d: video_row(runner.run(v, detector=d, tracker="off"))
                                        for d in exp.get("e1_detectors", ["yunet", "haar"])} for v in videos}
        if videos and "e2" in todo:
            print("E2 interval sweep", flush=True)
            res["e2"] = {v.name: {
                "off": video_row(runner.run(v, tracker="off")),
                "lk": [{"interval": n, **video_row(runner.run(v, tracker="lk", detect_interval=n))}
                       for n in exp.get("e2_intervals", [1, 5, 10, 15, 30])],
            } for v in videos}
            res["headline"] = headline(res, int(exp.get("e3_interval", 10)))
        if videos and ({"e3", "e5"} & todo):
            print("E3 trackers", flush=True)
            n = int(exp.get("e3_interval", 10))
            res["e3"] = {v.name: {t: video_row(runner.run(v, tracker=t, detect_interval=n))
                                  for t in exp.get("e3_trackers", ["lk", "csrt", "kcf", "none", "off"])} for v in videos}
        if images and "e4" in todo:
            print("E4 threshold sweep", flush=True)
            if synth:
                res["e4"] = e4_pr(synth, base, exp["e4_thresholds"])
            if real:
                res["e4_real"] = e4_pr(real, base, exp["e4_thresholds"])
        if videos and "e6" in todo:
            print("E6 padding sweep", flush=True)
            n = int(exp.get("e3_interval", 10))
            res["e6"] = {v.name: e6_padding(v, runner.run(v, tracker="lk", detect_interval=n), base,
                                            exp.get("e6_paddings", [0.0, 0.1, 0.2, 0.25, 0.3])) for v in videos}

        if videos and "e7" in todo:
            print("E7 refresh policy", flush=True)
            n = int(exp["e3_interval"])
            res["e7"] = {v.name: [{"policy": name, **video_row(runner.run(v, tracker="lk", detect_interval=n, **kw))}
                                  for name, kw in e7_policies(exp["e7_unconfirmed_intervals"])] for v in videos}

        n = int(exp["e3_interval"])
        if videos and "e8" in todo:
            print("E8 mask shape", flush=True)
            res["e8"] = {v.name: {shape: e6_padding(v, runner.run(v, tracker="lk", detect_interval=n),
                                                    replace(base, mask_shape=shape), exp["e8_paddings"])
                                  for shape in ("ellipse", "rect")} for v in videos}
        if videos and "e9" in todo:
            print("E9 smoothing", flush=True)
            res["e9"] = {v.name: e9_smoothing(v, runner.run(v, tracker="lk", detect_interval=n), base,
                                              exp["e9_windows"]) for v in videos}
        if videos and "e10" in todo:
            print("E10 coasting", flush=True)
            res["e10"] = {v.name: [{"tracker": t, "lost": "coast" if c else "freeze",
                                    **video_row(runner.run(v, tracker=t, detect_interval=n, coast=c))}
                                   for t in exp["e10_trackers"] for c in (False, True)] for v in videos}
        if real and "e11" in todo:
            print("E11 tiling", flush=True)
            rows = []
            for g in [0, *exp["e11_tile_grids"]]:
                r = e1_images(real, replace(base, tile_grid=g), ["yunet"])["yunet"]
                rows.append({"tiles": "off" if g == 0 else f"{g} x {g}", **{k: r[k] for k in
                             ("tp", "fp", "fn", "precision", "recall", "ms_per_image")}})
            res["e11"] = rows

        if videos and "e12" in todo:
            print("E12 backfill", flush=True)
            res["e12"] = {v.name: [{"backfill": b, **video_row(runner.run(v, tracker="lk", detect_interval=n,
                                                                          backfill_frames=b))}
                                   for b in exp["e12_backfill"]] for v in videos}

        if real and "e13" in todo:
            print("E13 output check", flush=True)
            res["e13"] = e13_output_check(real, base, exp["e13_settings"])
            if videos:
                res["e13"]["videos"] = e13_videos(videos, runner, n)

        if videos and "e14" in todo:
            print("E14 association, re-ID and motion models", flush=True)
            from core.recognition import available

            variants = [vt for vt in exp["e14_variants"] if not vt.get("reid_link") or available(base)]
            res["e14"] = {v.name: [{"tracker": t, "variant": vt["label"],
                                    **video_row(runner.run(v, tracker=t, detect_interval=n,
                                                           **{k: x for k, x in vt.items() if k != "label"}))}
                                   for t in exp["e14_trackers"] for vt in variants] for v in videos}
        if images and "e15" in todo:
            print("E15 profile fallback", flush=True)
            res["e15"] = []
            for src, items in (("real photos", real), ("synthetic stills", synth)):
                for prof in (False, True):
                    if not items:
                        continue
                    r = e1_images(items, replace(base, profile=prof), ["yunet"])["yunet"]
                    res["e15"].append({"set": src, "profile": "on" if prof else "off",
                                       **{k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "ms_per_image")}})

        if videos and "e16" in todo:
            print("E16 robustness", flush=True)
            res["e16"] = e16_robustness(videos, runner, n)

    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    (out / "raw_results.json").write_text(json.dumps(res, indent=1, default=str))
    plots = make_plots(res, out)
    write_results_md(results_md, report(res, exp))
    print(f"Wrote {out / 'raw_results.json'}, {len(plots)} plot(s), {results_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
