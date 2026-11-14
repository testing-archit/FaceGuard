"""Temporal smoothing of track boxes to reduce mask jitter.

Detector boxes wobble by a few pixels from frame to frame and tracks jump
when a detection re-anchors them, so the mask visibly shimmers. Because
analysis stores every box before anything is rendered, smoothing can be
zero-phase: a centred moving average uses past and future frames, so it
does not lag behind motion the way a causal filter (e.g. an exponential
moving average) would.

Smoothing is a low-pass filter on the box centre and size, applied per
contiguous run of frames (a gap in a track is never bridged). It trades a
little coverage at sudden motion for a steadier mask; E8 measures both.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.types import Box, Track


def _runs(frames: list[int]) -> list[list[int]]:
    """Split sorted frame numbers into runs of consecutive frames."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f == runs[-1][-1] + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0; the window shrinks symmetrically at the ends.

    Near the ends of a run the half-width is cut to the distance from the
    end, so the window stays centred: constant-velocity motion passes
    through unchanged and the first and last boxes are not dragged toward
    their neighbours. The very first and last boxes are kept as they are.
    """
    n = len(values)
    if window <= 1 or n <= 1:
        return values.astype(np.float64)
    idx = np.arange(n)
    half = np.minimum(np.minimum(idx, n - 1 - idx), window // 2)
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    lo = idx - half
    hi = idx + half + 1
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def smooth_history(history: dict[int, Box], window: int) -> dict[int, Box]:
    """Smooth centre (cx, cy) and size (w, h) of a track's boxes, run by run."""
    if window <= 1 or len(history) < 2:
        return dict(history)
    out: dict[int, Box] = {}
    for run in _runs(sorted(history)):
        boxes = [history[f] for f in run]
        arr = np.array([[b.x + b.w / 2, b.y + b.h / 2, b.w, b.h] for b in boxes], np.float64)
        sm = moving_average(arr, window)
        for f, (cx, cy, w, h) in zip(run, sm):
            out[f] = Box.from_xywh(cx - w / 2, cy - h / 2, w, h)
    return out


def smooth_tracks(tracks: list[Track], window: int) -> list[Track]:
    """Copies of `tracks` with smoothed boxes. Manual tracks are left exactly
    where the user put them."""
    if window <= 1:
        return tracks
    return [
        t if t.source == "manual" else replace(t, history=smooth_history(t.history, window))
        for t in tracks
    ]


def jitter_px(tracks: list[Track]) -> float:
    """Mean magnitude of the second difference of box centres, in px/frame².

    Zero for a box moving at constant velocity; large when the mask shakes.
    Computed over consecutive-frame triples only.
    """
    acc: list[float] = []
    for t in tracks:
        for run in _runs(sorted(t.history)):
            if len(run) < 3:
                continue
            c = np.array([t.history[f].center for f in run])
            acc.extend(np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1).tolist())
    return float(np.mean(acc)) if acc else 0.0
