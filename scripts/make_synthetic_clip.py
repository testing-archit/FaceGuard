"""Generate a 20 s test clip with exact per-frame face ground truth.

A real group photo (er.png, 6 faces) is animated across a 1280x720 canvas
through segments that stress different parts of the pipeline:

    0-4 s    static                       easy baseline
    4-8 s    slow pan                     tracking should be near perfect
    8-12 s   in-plane rotation to ±40°    YuNet misses rolled faces
    12-14 s  fast whip pan                large inter-frame motion, LK pyramid test
    15-18 s  low light with sensor noise  detector confidence drops
    18-20 s  zoom out to ~17 px faces     below YuNet's reliable size

Because the motion is generated, the face boxes in every frame are known
exactly. The six source boxes were detected by YuNet on the still photo and
checked by eye. A 440 Hz tone is muxed in so audio preservation can be tested.

Writes data/samples/synthetic_er.mp4 and data/samples/synthetic_er.gt.json.
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
OUT = ROOT / "data" / "samples" / "synthetic_er.mp4"
GT = ROOT / "data" / "samples" / "synthetic_er.gt.json"

W, H, FPS, SECONDS = 1280, 720, 30, 20
TONE_HZ = 440
NOISE_SIGMA = 6.0  # grey levels, low-light segment only
MIN_VISIBLE = 0.5  # drop GT faces less than half inside the frame

# Hand-checked YuNet boxes on er.png, [x, y, w, h].
FACES = [
    [340, 35, 31, 40],
    [141, 256, 33, 42],
    [68, 127, 30, 39],
    [120, 48, 31, 40],
    [259, 229, 31, 40],
    [406, 87, 33, 43],
]

# time (s): scale, angle (deg), tx, ty (px from canvas centre), brightness
KEYS = [
    (0.0, 1.4, 0, 0, 0, 1.0),
    (4.0, 1.4, 0, 0, 0, 1.0),
    (6.0, 1.4, 0, -160, 0, 1.0),
    (8.0, 1.4, 0, 160, 0, 1.0),
    (8.5, 1.4, 0, 0, 0, 1.0),
    (9.5, 1.4, 40, 0, 0, 1.0),
    (11.0, 1.4, -40, 0, 0, 1.0),
    (12.0, 1.4, 0, 0, 0, 1.0),
    (12.5, 1.4, 0, 0, 0, 1.0),
    (13.0, 1.4, 0, 300, 0, 1.0),
    (13.5, 1.4, 0, -300, 0, 1.0),
    (14.0, 1.4, 0, 0, 0, 1.0),
    (15.0, 1.4, 0, 0, 0, 1.0),
    (15.5, 1.4, 0, 0, 0, 0.3),
    (17.5, 1.4, 0, 0, 0, 0.3),
    (18.0, 1.4, 0, 0, 0, 1.0),
    (20.0, 0.55, 0, 0, 0, 1.0),
]


def params_at(t: float) -> tuple[float, ...]:
    """Smoothstep interpolation between keyframes."""
    for (t0, *a), (t1, *b) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            u = u * u * (3 - 2 * u)
            return tuple(x + (y - x) * u for x, y in zip(a, b))
    return tuple(KEYS[-1][1:])


def affine(scale: float, angle: float, tx: float, ty: float, src_w: int, src_h: int) -> np.ndarray:
    m = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), angle, scale)
    m[0, 2] += W / 2 - src_w / 2 + tx
    m[1, 2] += H / 2 - src_h / 2 + ty
    return m


def gt_boxes(m: np.ndarray, scale: float) -> list[dict]:
    """Face boxes in canvas coordinates: centre follows the affine, size follows scale."""
    out = []
    for fid, (x, y, w, h) in enumerate(FACES):
        cx, cy = m @ np.array([x + w / 2, y + h / 2, 1.0])
        bw, bh = w * scale, h * scale
        x0, y0 = cx - bw / 2, cy - bh / 2
        vis_w = min(x0 + bw, W) - max(x0, 0)
        vis_h = min(y0 + bh, H) - max(y0, 0)
        if vis_w <= 0 or vis_h <= 0 or vis_w * vis_h < MIN_VISIBLE * bw * bh:
            continue
        out.append({"id": fid, "box": [round(x0), round(y0), round(bw), round(bh)]})
    return out


def main() -> int:
    src = cv2.imread(str(SRC))
    if src is None:
        print("data/samples/er.png missing; run scripts/fetch_samples.py first", file=sys.stderr)
        return 1
    sh, sw = src.shape[:2]
    rng = np.random.default_rng(0)
    silent = OUT.with_name(OUT.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames: dict[str, list[dict]] = {}

    for i in range(FPS * SECONDS):
        scale, angle, tx, ty, bright = params_at(i / FPS)
        m = affine(scale, angle, tx, ty, sw, sh)
        frame = cv2.warpAffine(src, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))
        if bright < 1.0:
            f = frame.astype(np.float32) * bright + rng.normal(0, NOISE_SIGMA, frame.shape)
            frame = np.clip(f, 0, 255).astype(np.uint8)
        writer.write(frame)
        frames[str(i)] = gt_boxes(m, scale)
    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
         "-f", "lavfi", "-i", f"sine=frequency={TONE_HZ}:duration={SECONDS}",
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", "-c:a", "aac", "-shortest", str(OUT)],
        check=True,
    )
    silent.unlink()

    GT.write_text(json.dumps({
        "videos": [{
            "file": f"samples/{OUT.name}",
            "fps": FPS,
            "synthetic": True,
            "interpolate": False,
            "segments": {"static": [0, 120], "slow_pan": [120, 240], "rotation": [240, 360],
                         "fast_pan": [360, 420], "low_light": [450, 540], "small_faces": [540, 600]},
            "frames": frames,
        }]
    }))
    print(f"Wrote {OUT} and {GT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
