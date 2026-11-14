import pytest

from core.redact import build_mask
from core.smoothing import jitter_px, moving_average, smooth_history, smooth_tracks
from core.types import Box, Config, Track

import numpy as np


def test_moving_average_preserves_linear_motion_and_ends():
    v = np.arange(10, dtype=float)[:, None]
    assert np.allclose(moving_average(v, 5), v)  # centred window: no lag on constant velocity


def test_smoothing_removes_jitter_without_bridging_gaps():
    hist = {f: Box(100 + 3 * f + (4 if f % 2 else -4), 50, 40, 40) for f in range(20)}
    hist.update({f: Box(500, 50, 40, 40) for f in range(30, 35)})
    t = Track(id=1, box=hist[0], history=hist)
    s = smooth_tracks([t], 5)[0]
    assert sorted(s.history) == sorted(hist)  # same frames, gap kept
    assert s.history[32] == Box(500, 50, 40, 40)
    assert jitter_px([s]) < jitter_px([t]) / 3


def test_manual_tracks_are_not_smoothed():
    hist = {f: Box(10 * (f % 2), 0, 20, 20) for f in range(10)}
    t = Track(id=1, box=hist[0], history=hist, source="manual")
    assert smooth_tracks([t], 5)[0].history == hist


def test_window_one_is_identity():
    hist = {0: Box(0, 0, 10, 10), 1: Box(5, 0, 10, 10)}
    assert smooth_history(hist, 1) == hist


def test_rect_mask_covers_whole_padded_box():
    face = Box(100, 100, 40, 40)
    cfg = Config(mask_shape="rect", padding=0.0)
    roi, a = build_mask(face, 640, 480, cfg)
    inside = a[100 - roi.y : 140 - roi.y, 100 - roi.x : 140 - roi.x]
    assert inside.min() == 1.0  # corners included, unlike the ellipse at padding 0
    _, e = build_mask(face, 640, 480, Config(padding=0.0))
    assert e.min() < 1.0
    with pytest.raises(ValueError):
        Config(mask_shape="star")
