import cv2
import numpy as np
import pytest

from core.tracking import LKTracker, NullTracker, OpenCVTracker, _median_scale, tracker_factory
from core.types import Box, Config


def textured(seed=0, size=(480, 640)):
    """Smooth random texture: plenty of Shi-Tomasi corners, LK-friendly."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size[0] // 8, size[1] // 8), dtype=np.uint8)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0)


def warp(gray, dx=0.0, dy=0.0, scale=1.0, center=(320, 240)):
    m = cv2.getRotationMatrix2D(center, 0, scale)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REFLECT)


def bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_lk_recovers_translation():
    g0 = textured()
    g1 = warp(g0, dx=6, dy=-4)
    t = LKTracker(Config())
    t.init(bgr(g0), g0, Box(270, 190, 100, 100))
    b = t.update(bgr(g1), g1)
    assert b is not None
    assert (b.x, b.y) == pytest.approx((276, 186), abs=1)
    assert (b.w, b.h) == pytest.approx((100, 100), abs=2)


def test_lk_recovers_scale_about_box_centre():
    g0 = textured(1)
    g1 = warp(g0, scale=1.1, center=(320, 240))
    t = LKTracker(Config())
    t.init(bgr(g0), g0, Box(270, 190, 100, 100))
    b = t.update(bgr(g1), g1)
    assert b.w == pytest.approx(110, abs=3)
    assert b.center == pytest.approx((320, 240), abs=2)


def test_lk_large_motion_handled_by_pyramid():
    g0 = textured(2)
    g1 = warp(g0, dx=30, dy=0)
    t = LKTracker(Config())
    t.init(bgr(g0), g0, Box(270, 190, 100, 100))
    b = t.update(bgr(g1), g1)
    assert b is not None and b.x == pytest.approx(300, abs=2)


def test_lk_reports_lost_when_scene_changes():
    g0, g1 = textured(3), textured(4)
    t = LKTracker(Config())
    t.init(bgr(g0), g0, Box(270, 190, 100, 100))
    assert t.update(bgr(g1), g1) is None


def test_lk_textureless_box_is_lost():
    flat = np.full((480, 640), 128, np.uint8)
    t = LKTracker(Config())
    t.init(bgr(flat), flat, Box(270, 190, 100, 100))
    assert t.update(bgr(flat), flat) is None


def test_median_scale():
    old = np.array([[0, 0], [10, 0], [0, 10]], float)
    assert _median_scale(old, old * 1.5) == pytest.approx(1.5)


ER = "data/samples/er.png"


@pytest.mark.skipif(not __import__("os").path.exists(ER), reason="run scripts/fetch_samples.py")
@pytest.mark.parametrize("kind", ["CSRT", "KCF"])
def test_opencv_trackers_follow_translation(kind):
    # Correlation-filter trackers need real image structure; smoothed noise
    # gives KCF a flat response and it never moves.
    g0 = cv2.cvtColor(cv2.imread(ER), cv2.COLOR_BGR2GRAY)
    face = Box(340, 35, 31, 40)
    t = OpenCVTracker(kind)
    t.init(bgr(g0), g0, face)
    # 1 px per frame for 8 frames. KCF cannot follow a single 4 px jump on a
    # 31 px face (its HOG cells quantise position), but follows gradual motion.
    for k in range(1, 9):
        g = warp(g0, dx=k, dy=k / 2, center=(250, 250))
        b = t.update(bgr(g), g)
        assert b is not None
    cx, cy = face.center
    assert b.center == pytest.approx((cx + 8, cy + 4), abs=3)


def test_null_tracker_holds_box():
    g = textured()
    t = NullTracker()
    t.init(bgr(g), g, Box(1, 2, 3, 4))
    assert t.update(bgr(g), g) == Box(1, 2, 3, 4)


@pytest.mark.parametrize("name", ["lk", "csrt", "kcf", "none", "off"])
def test_factory_builds_every_tracker(name):
    assert tracker_factory(Config(tracker=name))() is not None
