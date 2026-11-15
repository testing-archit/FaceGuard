import json
from pathlib import Path

import pytest

from core.backfill import BufferedFrame, FrameBuffer, backfill_track, visible_fraction
from core.pipeline import analyse_video, process_video, tracks_to_json
from core.tracking import LKTracker, NullTracker
from core.types import FLAG_BACKFILL, Box, Config, Track
from core.video_io import VideoReader
from eval.metrics import evaluate_video, load_video_truth
from tests.test_tracking import bgr, textured, warp

CROSS = Path("data/samples/synthetic_cross.mp4")
needs_cross = pytest.mark.skipif(not CROSS.exists(), reason="run scripts/make_crossing_clip.py")


def test_frame_buffer_keeps_last_n_oldest_first():
    buf = FrameBuffer(3)
    out = [buf.push(BufferedFrame(i, None, None)) for i in range(5)]
    assert [o.index if o else None for o in out] == [None, None, None, 0, 1]
    assert [b.index for b in buf.newest_first()] == [4, 3, 2]
    assert [b.index for b in buf.drain()] == [2, 3, 4] and len(buf) == 0


def test_visible_fraction():
    assert visible_fraction(Box(0, 0, 10, 10), 100, 100) == 1.0
    assert visible_fraction(Box(-5, 0, 10, 10), 100, 100) == 0.5
    assert visible_fraction(Box(200, 0, 10, 10), 100, 100) == 0.0


def test_backfill_follows_motion_backwards_with_lk():
    """Texture moves +5 px/frame; the box found at frame 4 is traced back to frame 0."""
    g0 = textured(7)
    frames = [warp(g0, dx=5 * k) for k in range(5)]
    birth_box = Box(290, 190, 100, 100)
    t = Track(id=1, box=birth_box, history={4: birth_box})
    past = [BufferedFrame(k, bgr(frames[k]), frames[k]) for k in (3, 2, 1, 0)]
    n = backfill_track(t, BufferedFrame(4, bgr(frames[4]), frames[4]), past, LKTracker(Config()), Config())
    assert n == 4
    assert t.history[0].x == pytest.approx(270, abs=2)
    assert all(t.flags[k] == FLAG_BACKFILL for k in range(4))


def test_backfill_stops_at_floor_existing_box_and_frame_edge():
    cfg = Config()
    frame = bgr(textured(1))
    past = [BufferedFrame(k, frame, None) for k in (9, 8, 7, 6)]
    t = Track(id=1, box=Box(10, 10, 20, 20), history={10: Box(10, 10, 20, 20)})
    assert backfill_track(t, BufferedFrame(10, frame, None), past, NullTracker(), cfg, floor=8) == 2
    t = Track(id=1, box=Box(10, 10, 20, 20), history={10: Box(10, 10, 20, 20), 8: Box(0, 0, 1, 1)})
    assert backfill_track(t, BufferedFrame(10, frame, None), past, NullTracker(), cfg) == 1
    off_edge = Box(-15, 10, 20, 20)  # 25% inside the frame: the face had not entered yet
    t = Track(id=1, box=off_edge, history={10: off_edge})
    assert backfill_track(t, BufferedFrame(10, frame, None), past, NullTracker(), cfg) == 0


@needs_cross
def test_backfill_removes_entry_exposure(tmp_path):
    entry = json.loads(CROSS.with_suffix(".gt.json").read_text())["videos"][0]
    truth = load_video_truth(entry)
    with VideoReader(CROSS) as r:
        w, h = r.info.width, r.info.height
    leaks = {}
    for n in (0, 10):
        cfg = Config(backfill_frames=n)
        a = analyse_video(CROSS, cfg)
        leaks[n] = evaluate_video(a.tracks, truth, w, h, cfg, entry["segments"]).leak_frames_by_segment["entry"]
    assert leaks[0] > 0 and leaks[10] == 0


@needs_cross
def test_delayed_writer_matches_two_pass_with_backfill(tmp_path):
    cfg = Config(backfill_frames=10)
    single = process_video(CROSS, tmp_path / "a.mp4", cfg)
    assert tracks_to_json(single.tracks) == tracks_to_json(analyse_video(CROSS, cfg).tracks)
    with VideoReader(tmp_path / "a.mp4") as r:
        assert r.info.frame_count == single.stats.frames
    assert single.stats.backfilled_frames > 0
