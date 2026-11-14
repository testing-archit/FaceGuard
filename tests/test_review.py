import cv2
import numpy as np
import pytest

from core.errors import FaceGuardError
from core.review import (
    delete_track, detections_to_tracks, draw_overlay, interpolate_boxes, manual_track,
    new_job_dir, next_track_id, propagate, read_frame, remove_job_dir, remove_stale_jobs, set_redact,
)
from core.types import Box, Config, Detection
from core.pipeline import boxes_at


def textured(seed=0, size=(240, 320)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size[0] // 8, size[1] // 8), dtype=np.uint8)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 0)


def moving_clip(path, frames=20, step=3):
    """Texture translating right by `step` px per frame."""
    base = textured()
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for i in range(frames):
        m = np.float32([[1, 0, i * step], [0, 1, 0]])
        g = cv2.warpAffine(base, m, (320, 240), borderMode=cv2.BORDER_REFLECT)
        w.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    w.release()


def test_image_detections_become_numbered_tracks():
    tracks = detections_to_tracks([Detection(Box(0, 0, 10, 10), 0.9), Detection(Box(20, 20, 10, 10), 0.7)])
    assert [t.id for t in tracks] == [1, 2]
    assert tracks[1].history == {0: Box(20, 20, 10, 10)}
    assert next_track_id(tracks) == 3


def test_untick_delete_and_manual_change_what_is_rendered():
    tracks = detections_to_tracks([Detection(Box(i * 20, 0, 10, 10), 0.9) for i in range(4)])
    set_redact(tracks, 2, False)
    tracks = delete_track(tracks, 4)
    tracks.append(manual_track(next_track_id(tracks), {0: Box(100, 100, 30, 30)}))
    assert boxes_at(tracks, 0) == [Box(0, 0, 10, 10), Box(40, 0, 10, 10), Box(100, 100, 30, 30)]
    assert tracks[-1].source == "manual"


def test_interpolate_endpoints_and_midpoint():
    h = interpolate_boxes(10, Box(0, 0, 10, 10), 20, Box(100, 50, 30, 30))
    assert sorted(h) == list(range(10, 21))
    assert h[10] == Box(0, 0, 10, 10) and h[20] == Box(100, 50, 30, 30)
    assert h[15] == Box(50, 25, 20, 20)


def test_propagate_follows_motion(tmp_path):
    clip = tmp_path / "m.avi"
    moving_clip(clip)
    h = propagate(clip, 5, Box(100, 80, 60, 60), Config(tracker="lk"))
    assert min(h) == 5 and max(h) == 19
    assert h[5] == Box(100, 80, 60, 60)
    assert h[19].x == pytest.approx(100 + 14 * 3, abs=3)


def test_propagate_respects_end_and_hold(tmp_path):
    clip = tmp_path / "m.avi"
    moving_clip(clip)
    h = propagate(clip, 2, Box(100, 80, 60, 60), Config(tracker="none"), end=8)
    assert sorted(h) == list(range(2, 9))
    assert set(h.values()) == {Box(100, 80, 60, 60)}


def test_propagate_rejects_bad_input(tmp_path):
    clip = tmp_path / "m.avi"
    moving_clip(clip)
    with pytest.raises(FaceGuardError):
        propagate(clip, 0, Box(0, 0, 0, 10), Config())
    with pytest.raises(FaceGuardError):
        propagate(clip, 500, Box(0, 0, 10, 10), Config())


def test_read_frame_matches_sequential(tmp_path):
    clip = tmp_path / "m.avi"
    moving_clip(clip)
    cap = cv2.VideoCapture(str(clip))
    for _ in range(7):
        ok, f = cap.read()
    cap.release()
    assert np.array_equal(read_frame(clip, 6), f)


def test_overlay_does_not_touch_input():
    img = np.zeros((100, 100, 3), np.uint8)
    tracks = detections_to_tracks([Detection(Box(10, 10, 30, 30), 0.9)])
    out = draw_overlay(img, tracks, 0, draft=Box(50, 50, 20, 20), cfg=Config())
    assert not img.any() and out.any()


def test_job_dirs_and_stale_cleanup(tmp_path):
    d = new_job_dir(tmp_path)
    (d / "upload.mp4").write_bytes(b"x")
    assert remove_stale_jobs(tmp_path, max_age_s=3600) == 0
    assert remove_stale_jobs(tmp_path, max_age_s=-1) == 1
    assert not d.exists()
    d2 = new_job_dir(tmp_path)
    remove_job_dir(d2)
    assert not d2.exists()
