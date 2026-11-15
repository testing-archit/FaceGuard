"""Build the report figures and the demo clip from the public sample media.

    python scripts/make_demo.py

Writes (all regenerated, nothing hand-edited):
- docs/figures/failure_rotation.png   per-frame detection misses a rolled face; tracking covers it
- docs/figures/failure_entry.png      a face entering between detections is exposed (honest limitation)
- docs/figures/selective.png          selective redaction: one person deliberately left visible
- docs/figures/modes.png              solid / pixelate / blur on the same face
- out/demo/demo_side_by_side.mp4      15 s original | redacted, for the demo (not committed)

Needs scripts/fetch_samples.py and scripts/make_*_clip.py to have been run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.detectors import YuNetDetector  # noqa: E402
from core.pipeline import analyse_video, boxes_at  # noqa: E402
from core.redact import redact  # noqa: E402
from core.review import read_frame  # noqa: E402
from core.types import Box, Config  # noqa: E402
from eval.metrics import LEAK_COVERAGE, coverage, load_video_truth  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
FIGS = ROOT / "docs" / "figures"
DEMO = ROOT / "out" / "demo"
PANEL_W = 640  # width of each panel in the figures
DEMO_SECONDS = 15
RED, TEAL = (40, 40, 220), (150, 160, 15)  # BGR


def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Caption bar above a panel."""
    bar = np.full((34, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return cv2.vconcat([bar, img])


def fit(img: np.ndarray, w: int = PANEL_W) -> np.ndarray:
    return cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def outline(img: np.ndarray, boxes: list[Box], color, thick: int = 3) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        cv2.rectangle(out, (b.x - 6, b.y - 6), (b.x + b.w + 6, b.y + b.h + 6), color, thick, cv2.LINE_AA)
    return out


def leak_boxes(frame_gt: list[tuple[int, Box]], preds: list[Box], w: int, h: int, cfg: Config) -> list[Box]:
    return [g for _, g in frame_gt if coverage(g, preds, w, h, cfg) < LEAK_COVERAGE]


def worst_leak_frame(truth, tracks, w, h, cfg, lo: int, hi: int) -> int:
    """Frame in [lo, hi) where the most ground-truth faces are exposed."""
    best, best_n = lo, -1
    for f in range(lo, hi):
        n = len(leak_boxes(truth.get(f, []), boxes_at(tracks, f), w, h, cfg))
        if n > best_n:
            best, best_n = f, n
    return best


def failure_rotation(cfg: Config) -> None:
    clip = SAMPLES / "synthetic_er.mp4"
    entry = json.loads((SAMPLES / "synthetic_er.gt.json").read_text())["videos"][0]
    truth = load_video_truth(entry)
    off = analyse_video(clip, replace(cfg, tracker="off"))
    lk = analyse_video(clip, replace(cfg, tracker="lk"))
    w, h = off.info.width, off.info.height
    f = worst_leak_frame(truth, off.tracks, w, h, cfg, *entry["segments"]["rotation"])
    frame = read_frame(clip, f)
    a = redact(frame, boxes_at(off.tracks, f), cfg)
    b = redact(frame, boxes_at(lk.tracks, f), cfg)
    exposed_a = leak_boxes(truth[f], boxes_at(off.tracks, f), w, h, cfg)
    exposed_b = leak_boxes(truth[f], boxes_at(lk.tracks, f), w, h, cfg)
    panels = [
        label(fit(outline(a, exposed_a, RED)), f"Detect every frame, no tracking: {len(exposed_a)} face(s) exposed"),
        label(fit(outline(b, exposed_b, RED)), f"Detect every 10 frames + LK tracking: {len(exposed_b)} exposed"),
    ]
    cv2.imwrite(str(FIGS / "failure_rotation.png"), cv2.hconcat(panels))
    print(f"failure_rotation.png: frame {f}, exposed {len(exposed_a)} -> {len(exposed_b)}")


def failure_entry(cfg: Config) -> None:
    clip = SAMPLES / "synthetic_cross.mp4"
    entry = json.loads((SAMPLES / "synthetic_cross.gt.json").read_text())["videos"][0]
    truth = load_video_truth(entry)
    lk = analyse_video(clip, replace(cfg, tracker="lk"))
    w, h = lk.info.width, lk.info.height
    f = worst_leak_frame(truth, lk.tracks, w, h, cfg, *entry["segments"]["entry"])
    frame = read_frame(clip, f)
    out = redact(frame, boxes_at(lk.tracks, f), cfg)
    exposed = leak_boxes(truth[f], boxes_at(lk.tracks, f), w, h, cfg)
    nxt = next((g for g in range(f + 1, f + 60) if not leak_boxes(truth.get(g, []), boxes_at(lk.tracks, g), w, h, cfg)), None)
    later = redact(read_frame(clip, nxt), boxes_at(lk.tracks, nxt), cfg) if nxt is not None else out
    panels = [label(fit(outline(out, exposed, RED)), f"Frame {f}: a person enters between detections (exposed)"),
              label(fit(later), f"Frame {nxt}: next detection finds and hides them")]
    cv2.imwrite(str(FIGS / "failure_entry.png"), cv2.hconcat(panels))
    print(f"failure_entry.png: exposed at {f}, covered from {nxt}")


def selective(cfg: Config) -> None:
    img = cv2.imread(str(SAMPLES / "addams-family.png"))
    faces = YuNetDetector(cfg).detect(img)
    keep = max(faces, key=lambda d: d.box.y + d.box.h)  # the person lowest in frame
    hidden = [d.box for d in faces if d is not keep]
    out = outline(redact(img, hidden, replace(cfg, mode="fill")), [keep.box], TEAL)
    panels = [label(fit(img, 420), "Original"),
              label(fit(out, 420), f"{len(hidden)} hidden, 1 left visible on purpose")]
    cv2.imwrite(str(FIGS / "selective.png"), cv2.hconcat(panels))
    print("selective.png")


def modes(cfg: Config) -> None:
    img = cv2.imread(str(SAMPLES / "karen-and-rob.png"))
    faces = [d.box for d in YuNetDetector(cfg).detect(img)]
    panels = [label(fit(redact(img, faces, replace(cfg, mode=m)), 300), name)
              for m, name in (("fill", "Solid (strongest)"), ("pixelate", "Pixelate"), ("blur", "Blur (weakest)"))]
    cv2.imwrite(str(FIGS / "modes.png"), cv2.hconcat(panels))
    print("modes.png")


def demo_clip(cfg: Config) -> None:
    """Side-by-side original | redacted, first DEMO_SECONDS of the synthetic clip."""
    clip = SAMPLES / "synthetic_er.mp4"
    a = analyse_video(clip, replace(cfg, tracker="lk"))
    fps = a.info.fps
    n = int(DEMO_SECONDS * fps)
    DEMO.mkdir(parents=True, exist_ok=True)
    silent = DEMO / "demo.silent.mp4"
    w = PANEL_W
    h = round(a.info.height * w / a.info.width) + 34
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    cap = cv2.VideoCapture(str(clip))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        red = redact(frame, boxes_at(a.tracks, i), replace(cfg, mode="fill"))
        writer.write(cv2.hconcat([label(fit(frame), "Original"),
                                  label(fit(red), "FaceGuard (LK, detect every 10 frames)")]))
    cap.release()
    writer.release()
    out = DEMO / "demo_side_by_side.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-map_metadata", "-1", str(out)], check=True)
    silent.unlink()
    print(f"{out.relative_to(ROOT)}")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    failure_rotation(cfg)
    failure_entry(cfg)
    selective(cfg)
    modes(cfg)
    demo_clip(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
