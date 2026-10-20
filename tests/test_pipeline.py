from pathlib import Path

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
    a = analyse_video(SYNTH, Config(tracker="lk", detect_interval=10))
    assert a.stats.detector_calls == 60
    off = analyse_video(SYNTH, Config(tracker="off", detect_interval=10))
    assert off.stats.detector_calls == 600  # off forces per-frame detection


@needs_clip
def test_tracks_json_round_trip():
    a = analyse_video(SYNTH, Config(tracker="lk"))
    assert tracks_to_json(tracks_from_json(tracks_to_json(a.tracks))) == tracks_to_json(a.tracks)
