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


def test_runs_stops_and_stepping():
    from core.review import flag_runs, frame_runs, review_stops, step_to
    from core.types import Track

    assert frame_runs([1, 2, 3, 7, 8, 10]) == [(1, 3), (7, 8), (10, 10)]
    t = Track(id=1, box=Box(0, 0, 1, 1), flags={4: "lost", 5: "lost", 6: "unconfirmed", 20: "lost"})
    assert flag_runs(t) == [(4, 5, "lost"), (6, 6, "unconfirmed"), (20, 20, "lost")]
    stops = review_stops([t])
    assert stops == [4, 6, 20]
    assert step_to(stops, 4, +1) == 6 and step_to(stops, 4, -1) is None and step_to(stops, 25, -1) == 20


def test_thumbnail_collector_keeps_largest_confirmed_box():
    from core.review import ThumbnailCollector
    from core.types import Track

    frame = np.zeros((200, 200, 3), np.uint8)
    t = Track(id=1, box=Box(10, 10, 20, 20), history={0: Box(10, 10, 20, 20), 1: Box(10, 10, 60, 60),
                                                    2: Box(10, 10, 90, 90)}, flags={2: "lost"})
    c = ThumbnailCollector(size=48)
    for i in range(3):
        c(i, frame, [t])
    assert max(c.thumbs[1].shape[:2]) == 48 and c._area[1] == 3600  # frame 2 is flagged, skipped


def test_redaction_log_has_no_pixels_and_summarises():
    import json

    from core.review import redaction_log
    from core.types import Track

    tracks = [Track(id=1, box=Box(0, 0, 5, 5), history={0: Box(0, 0, 5, 5), 1: Box(0, 0, 5, 5)}, flags={1: "lost"}),
              Track(id=2, box=Box(9, 9, 5, 5), history={0: Box(9, 9, 5, 5)}, redact=False, source="manual")]
    log = redaction_log(tracks, Config(), {"name": "x.mp4", "frames": 2})
    json.dumps(log)
    assert log["summary"] == {"regions": 2, "redacted": 1, "left_visible": [2], "manual": [2], "frames_to_review": 1}
    assert log["regions"][0]["frames"] == [[0, 1]] and log["regions"][0]["flagged"] == [[1, 1, "lost"]]
    assert "model_path" not in log["settings"]


def test_session_round_trip_and_binding(tmp_path):
    from core.errors import FaceGuardError
    from core.review import session_from_json, session_to_json
    from core.types import Box, Track

    src = tmp_path / "a.mp4"
    src.write_bytes(b"video bytes")
    tracks = [Track(1, Box(1, 2, 3, 4), redact=False, history={0: Box(1, 2, 3, 4), 1: Box(2, 2, 3, 4)},
                    flags={1: "lost"}), Track(2, Box(5, 5, 5, 5), source="manual", history={3: Box(5, 5, 5, 5)})]
    data = session_to_json(tracks, src, {"detector": "yunet"})
    back, used = session_from_json(data, src)
    assert [(t.id, t.redact, t.source, t.history, t.flags) for t in back] == \
           [(t.id, t.redact, t.source, t.history, t.flags) for t in tracks]
    assert used == {"detector": "yunet"}
    other = tmp_path / "b.mp4"
    other.write_bytes(b"other video")
    with pytest.raises(FaceGuardError, match="different file"):
        session_from_json(data, other)
    with pytest.raises(FaceGuardError, match="Not a FaceGuard"):
        session_from_json({"tracks": []}, src)
