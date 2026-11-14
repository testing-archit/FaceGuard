"""Generate a 15 s clip of faces crossing and entering, with exact ground truth.

Three face patches are cut from the er.png group photo and moved over a
plain, face-free background:

    A  walks left -> right
    B  walks right -> left, passes IN FRONT of A around the middle
       (occlusion + the ID-switch case from EVALUATION.md)
    C  enters from the right edge at 8 s and stops (new-face acquisition:
       exposure before the next detection frame is measured)

Patches are 2.4x the source so faces are ~80 px wide, and are blended
under an elliptical alpha so there is no rectangular edge for the trackers
to lock on to. A face is in the ground truth while at least half of its box
is visible (inside the frame and not hidden behind a patch drawn in front).

Writes data/samples/synthetic_cross.mp4 and data/samples/synthetic_cross.gt.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "samples" / "er.png"
OUT = ROOT / "data" / "samples" / "synthetic_cross.mp4"
GT = ROOT / "data" / "samples" / "synthetic_cross.gt.json"

W, H, FPS, SECONDS = 1280, 720, 30, 15
SCALE = 2.4  # patch magnification: ~33 px source faces -> ~80 px
CONTEXT = 1.6  # patch half-size as a multiple of the face box size
MIN_VISIBLE = 0.5
BOB_PX = 6  # vertical walking bob amplitude
BOB_PERIOD = 24  # frames per bob cycle

# Source face boxes on er.png (same hand-checked boxes as make_synthetic_clip.py).
FACE_A = (141, 256, 33, 42)
FACE_B = (406, 87, 33, 43)
FACE_C = (120, 48, 31, 40)

# id, source face, z-order (higher = in front), first frame, last frame,
# start centre, end centre (patch centre path, linear between the frames;
# the patch holds its last position afterwards)
ACTORS = [
    (0, FACE_A, 0, 0, 360, (200, 380), (1080, 380)),
    (1, FACE_B, 1, 0, 360, (1080, 400), (200, 400)),
    (2, FACE_C, 0, 240, 330, (1400, 200), (900, 200)),
]


def background() -> np.ndarray:
    """Smooth gradient with low-frequency texture: no faces, some LK features."""
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (H // 40, W // 40), dtype=np.uint8)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    ramp = np.linspace(70, 140, W, dtype=np.float32)[None, :].repeat(H, 0)
    g = np.clip(0.75 * ramp + 0.12 * noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def cut_patch(src: np.ndarray, face: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Scaled patch around a face, its elliptical alpha, and the face box inside it."""
    x, y, w, h = face
    cx, cy = x + w / 2, y + h / 2
    hw, hh = CONTEXT * w, CONTEXT * h
    x0, y0 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
    x1, y1 = int(min(cx + hw, src.shape[1])), int(min(cy + hh, src.shape[0]))
    patch = cv2.resize(src[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    ph, pw = patch.shape[:2]
    alpha = np.zeros((ph, pw), np.float32)
    cv2.ellipse(alpha, (pw // 2, ph // 2), (int(pw * 0.48), int(ph * 0.48)), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=pw * 0.03)
    face_in_patch = ((x - x0) * SCALE, (y - y0) * SCALE, w * SCALE, h * SCALE)
    return patch, alpha, face_in_patch


def position(actor: tuple, i: int) -> tuple[float, float] | None:
    _, _, _, f0, f1, (sx, sy), (ex, ey) = actor
    if i < f0:
        return None
    u = min((i - f0) / (f1 - f0), 1.0)
    bob = BOB_PX * np.sin(2 * np.pi * i / BOB_PERIOD) if u < 1.0 else 0.0
    return sx + (ex - sx) * u, sy + (ey - sy) * u + bob


def paste(frame: np.ndarray, patch: np.ndarray, alpha: np.ndarray, cx: float, cy: float) -> tuple[int, int]:
    ph, pw = patch.shape[:2]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + pw, W), min(y0 + ph, H)
    if fx1 > fx0 and fy1 > fy0:
        p = patch[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0].astype(np.float32)
        a = alpha[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0, None]
        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        frame[fy0:fy1, fx0:fx1] = (a * p + (1 - a) * roi).astype(np.uint8)
    return x0, y0


def visible_fraction(box: tuple[float, float, float, float], occluders: list[np.ndarray]) -> float:
    """Share of a face box inside the frame and not under an opaque front patch."""
    x, y, w, h = box
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    total = max((x1 - x0) * (y1 - y0), 1)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    hidden = np.zeros((cy1 - cy0, cx1 - cx0), bool)
    for occ in occluders:
        hidden |= occ[cy0:cy1, cx0:cx1] >= 0.5
    return float((~hidden).sum()) / total


def main() -> int:
    src = cv2.imread(str(SRC))
    if src is None:
        print("data/samples/er.png missing; run scripts/fetch_samples.py first", file=sys.stderr)
        return 1
    bg = background()
    patches = {a[0]: cut_patch(src, a[1]) for a in ACTORS}
    order = sorted(ACTORS, key=lambda a: a[2])  # back to front
    silent = OUT.with_name(OUT.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames: dict[str, list[dict]] = {}
    crossing: list[int] = []

    for i in range(FPS * SECONDS):
        frame = bg.copy()
        placed = []  # (actor, face box in frame, alpha footprint)
        for actor in order:
            pos = position(actor, i)
            if pos is None:
                continue
            patch, alpha, (fx, fy, fw, fh) = patches[actor[0]]
            x0, y0 = paste(frame, patch, alpha, *pos)
            foot = np.zeros((H, W), np.float32)
            ph, pw = patch.shape[:2]
            ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + ph, H)
            if ax1 > ax0 and ay1 > ay0:
                foot[ay0:ay1, ax0:ax1] = alpha[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
            placed.append((actor, (x0 + fx, y0 + fy, fw, fh), foot))
        gts = []
        for k, (actor, box, _) in enumerate(placed):
            front = [f for a2, _, f in placed[k + 1 :] if a2[2] > actor[2]]
            if visible_fraction(box, front) >= MIN_VISIBLE:
                gts.append({"id": actor[0], "box": [round(v) for v in box]})
            if front and any((f[int(max(box[1], 0)) : int(box[1] + box[3]), int(max(box[0], 0)) : int(box[0] + box[2])] > 0.5).any() for f in front):
                crossing.append(i)
        writer.write(frame)
        frames[str(i)] = gts
    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT)],
        check=True,
    )
    silent.unlink()
    cross = [min(crossing), max(crossing) + 1] if crossing else [0, 0]
    GT.write_text(json.dumps({
        "videos": [{
            "file": f"samples/{OUT.name}",
            "fps": FPS,
            "synthetic": True,
            "interpolate": False,
            "segments": {"approach": [0, cross[0]], "crossing": cross, "separate": [cross[1], 240],
                         "entry": [240, 330], "static": [330, FPS * SECONDS]},
            "frames": frames,
        }]
    }))
    print(f"Wrote {OUT} and {GT}; crossing frames {cross}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
