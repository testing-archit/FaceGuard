"""Label ground-truth face boxes into data/ground_truth.json (EVALUATION.md section 2).

    # images: one window per image, drag a box per face, ENTER/SPACE after each
    python eval/annotate.py images data/samples/crowd_01.jpg data/samples/dim_02.jpg \\
        --conditions low_light

    # video: annotate every 5th frame; ids are carried over from the previous
    # keyframe by overlap and can be corrected in the terminal
    python eval/annotate.py video data/samples/campus_walk.mp4 --every 5

Controls (OpenCV `selectROIs`): drag a rectangle, press ENTER or SPACE to
accept it, repeat for every face, then ESC to finish the frame. Press `c`
while dragging to cancel the current rectangle. Previously saved boxes and
the previous keyframe's boxes are drawn for reference.

Boxes are labelled by hand on purpose. Do not seed ground truth from the
detector under test: its misses would silently become "no face here".
Paths are stored relative to data/, matching the documented format.
Record consent for anyone identifiable before committing annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.types import Box  # noqa: E402
from core.video_io import VideoReader  # noqa: E402

DEFAULT_TRUTH = ROOT / "data" / "ground_truth.json"
CARRY_IOU = 0.3  # min overlap to carry a person id over from the previous keyframe
REF_COLOR = (0, 200, 255)  # previous / saved boxes, drawn for reference
WINDOW = "FaceGuard annotate"


def load_doc(path: Path) -> dict:
    if path.exists():
        doc = json.loads(path.read_text())
    else:
        doc = {}
    doc.setdefault("images", [])
    doc.setdefault("videos", [])
    return doc


def save_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


def rel_to_data(media: Path, truth: Path) -> str:
    """Path stored in the truth file, relative to the truth file's folder."""
    media = media.resolve()
    try:
        return media.relative_to(truth.resolve().parent).as_posix()
    except ValueError:
        return str(media)


def upsert(entries: list[dict], entry: dict) -> None:
    """Replace the entry for the same file, or append."""
    for k, e in enumerate(entries):
        if e["file"] == entry["file"]:
            entries[k] = entry
            return
    entries.append(entry)


def assign_ids(prev: list[tuple[int, Box]], boxes: list[Box], next_id: int) -> list[int]:
    """Carry person ids over from the previous keyframe by IoU; new faces get fresh ids."""
    ids: list[int] = []
    taken: set[int] = set()
    for b in boxes:
        best, best_iou = None, CARRY_IOU
        for pid, pb in prev:
            v = b.iou(pb)
            if pid not in taken and v >= best_iou:
                best, best_iou = pid, v
        if best is None:
            best = next_id
            next_id += 1
        taken.add(best)
        ids.append(best)
    return ids


def parse_ids(text: str, n: int) -> list[int] | None:
    parts = text.replace(",", " ").split()
    if len(parts) != n or not all(p.lstrip("-").isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def draw_refs(img: np.ndarray, refs: list[tuple[int, Box]]) -> np.ndarray:
    out = img.copy()
    for pid, b in refs:
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), REF_COLOR, 1)
        cv2.putText(out, str(pid), (b.x, max(b.y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REF_COLOR, 1)
    return out


def select_boxes(img: np.ndarray, refs: list[tuple[int, Box]], title: str) -> list[Box]:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(WINDOW, title)
    rects = cv2.selectROIs(WINDOW, draw_refs(img, refs), showCrosshair=False)
    return [Box(*map(int, r)) for r in rects if r[2] > 0 and r[3] > 0]


def annotate_images(paths: list[Path], truth: Path, conditions: list[str]) -> None:
    doc = load_doc(truth)
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path}: not a readable image", file=sys.stderr)
            continue
        key = rel_to_data(path, truth)
        old = next((e for e in doc["images"] if e["file"] == key), None)
        refs = [(f["id"], Box(*f["box"])) for f in old["faces"]] if old else []
        boxes = select_boxes(img, refs, f"{path.name}: drag each face, ENTER after each, ESC when done")
        if not boxes and refs:
            print(f"{path.name}: nothing drawn, kept the {len(refs)} saved box(es)")
            continue
        entry = {"file": key, "faces": [{"id": i, "box": b.as_list()} for i, b in enumerate(boxes)]}
        if conditions:
            entry["conditions"] = conditions
        upsert(doc["images"], entry)
        save_doc(truth, doc)  # save after every image so a crash loses little
        print(f"{path.name}: {len(boxes)} face(s)")
    cv2.destroyAllWindows()


def annotate_video(path: Path, truth: Path, every: int, conditions: list[str]) -> None:
    doc = load_doc(truth)
    key = rel_to_data(path, truth)
    entry = next((e for e in doc["videos"] if e["file"] == key), None)
    with VideoReader(path) as reader:
        fps = reader.info.fps
        if entry is None:
            entry = {"file": key, "fps": round(fps, 3), "frames": {}, "interpolate": True}
        if conditions:
            entry["conditions"] = conditions
        frames: dict[str, list[dict]] = entry["frames"]
        prev: list[tuple[int, Box]] = []
        next_id = 1 + max((f["id"] for fs in frames.values() for f in fs), default=-1)
        for i, frame in enumerate(reader):
            if i % every:
                continue
            saved = [(f["id"], Box(*f["box"])) for f in frames.get(str(i), [])]
            boxes = select_boxes(frame, saved or prev,
                                 f"{path.name} frame {i}: drag each face, ENTER after each, ESC when done")
            if not boxes and saved:
                prev = saved
                continue
            ids = assign_ids(prev, boxes, next_id)
            if boxes:
                answer = input(f"frame {i}: ids {' '.join(map(str, ids))} - ENTER to accept or type ids: ")
                typed = parse_ids(answer, len(boxes)) if answer.strip() else None
                if answer.strip() and typed is None:
                    print("  could not parse, kept the suggested ids")
                ids = typed or ids
            next_id = max([next_id - 1, *ids]) + 1
            frames[str(i)] = [{"id": pid, "box": b.as_list()} for pid, b in zip(ids, boxes)]
            prev = list(zip(ids, boxes))
            upsert(doc["videos"], entry)
            save_doc(truth, doc)
    cv2.destroyAllWindows()
    print(f"{path.name}: {len(frames)} keyframe(s) saved to {truth}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    im = sub.add_parser("images", help="annotate one or more images")
    im.add_argument("paths", nargs="+", type=Path)
    vd = sub.add_parser("video", help="annotate a video at keyframes")
    vd.add_argument("path", type=Path)
    vd.add_argument("--every", type=int, default=5, help="annotate every Nth frame (default 5)")
    for p in (im, vd):
        p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
        p.add_argument("--conditions", default="",
                       help="comma-separated tags for E5, e.g. low_light,profile,occlusion,motion")
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.kind == "images":
        annotate_images(args.paths, args.truth, conditions)
    else:
        annotate_video(args.path, args.truth, args.every, conditions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
