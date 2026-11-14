from pathlib import Path

import cv2

import pytest

from core.pipeline import analyse_video, process_video, tracks_from_json, tracks_to_json
from core.types import Config

SYNTH = Path("data/samples/synthetic_er.mp4")
needs_clip = pytest.mark.skipif(not SYNTH.exists(), reason="run scripts/make_synthetic_clip.py")


@needs_clip
def test_two_pass_matches_single_pass(tmp_path):
    """analyse (UI pass 1) must produce exactly the boxes process_video redacts."""
    cfg = Config(tracker="lk")
    single = process_video(SYNTH, tmp_path / "a.mp4", cfg)
    analysed = analyse_video(SYNTH, cfg)
    assert tracks_to_json(single.tracks) == tracks_to_json(analysed.tracks)


@needs_clip
def test_detector_runs_every_interval(tmp_path):
    fixed = Config(tracker="lk", detect_interval=10, refresh_on_lost=False, unconfirmed_interval=0)
    a = analyse_video(SYNTH, fixed)
    assert a.stats.detector_calls == 60 and a.stats.early_detections == 0
    a = analyse_video(SYNTH, Config(tracker="lk", detect_interval=10))
    assert a.stats.detector_calls == 60 + a.stats.early_detections
    off = analyse_video(SYNTH, Config(tracker="off", detect_interval=10))
    assert off.stats.detector_calls == 600  # off forces per-frame detection


@needs_clip
def test_tracks_json_round_trip():
    a = analyse_video(SYNTH, Config(tracker="lk"))
    assert tracks_to_json(tracks_from_json(tracks_to_json(a.tracks))) == tracks_to_json(a.tracks)


def test_scene_cut_needs_both_histogram_and_layout_change():
    import numpy as np
    from core.pipeline import frame_signature, is_scene_cut

    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 256, (180, 320, 3), dtype=np.uint8), (0, 0), 6)
    cfg = Config()
    sa = frame_signature(a, cfg)
    darker = (a.astype(np.float32) * 0.3).astype(np.uint8)  # fade: histogram changes, layout does not
    assert not is_scene_cut(sa, frame_signature(darker, cfg), cfg)
    shifted = np.roll(a, 40, axis=1)  # pan: layout changes, histogram does not
    assert not is_scene_cut(sa, frame_signature(shifted, cfg), cfg)
    other = np.full_like(a, 230)
    other[:90] = 20  # a different shot: both change
    assert is_scene_cut(sa, frame_signature(other, cfg), cfg)
    assert not is_scene_cut(sa, frame_signature(other, cfg), Config(scene_cut_corr=0))


def test_scene_cut_ends_tracks(tmp_path):
    import numpy as np

    clip = tmp_path / "cut.avi"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    img = cv2.imread("data/samples/er.png") if Path("data/samples/er.png").exists() else None
    if img is None:
        pytest.skip("run scripts/fetch_samples.py")
    shot1 = cv2.resize(img, (320, 240))
    shot2 = np.full_like(shot1, 200)
    shot2[:120] = 30
    for i in range(20):
        w.write(shot1 if i < 10 else shot2)
    w.release()
    a = analyse_video(clip, Config(tracker="lk", detect_interval=30))
    assert a.stats.scene_cuts == 1
    assert {e.frame for e in a.events if e.kind == "scene_cut"} == {10}
    assert all(max(t.history) < 10 for t in a.tracks)  # no ID crosses the cut
