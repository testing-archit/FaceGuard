from pathlib import Path

import pytest

from core.parallel import analyse_parallel, chunks, stitch
from core.pipeline import Analysis, RunStats, analyse_video, tracks_to_json
from core.types import Box, Config, Track
from core.video_io import VideoInfo, VideoReader

SYNTH = Path("data/samples/synthetic_er.mp4")


def test_chunks_cover_the_range_without_gaps():
    assert chunks(1000, 4, 150) == [(0, 250), (250, 500), (500, 750), (750, 1000)]
    assert chunks(200, 8, 150) == [(0, 200)]  # too short to split
    assert chunks(450, 8, 150) == [(0, 150), (150, 300), (300, 450)]


def _analysis(tracks):
    return Analysis(info=VideoInfo(10, 10, 30.0, 20), tracks=tracks, events=[], stats=RunStats())


def test_stitch_continues_tracks_and_drops_warmup_only_tracks():
    b = Box(10, 10, 20, 20)
    a1 = Track(1, b, history={f: b for f in range(0, 10)})
    gone = Track(2, b, history={f: Box(80, 80, 5, 5) for f in range(0, 4)})
    # chunk 2 = [10, 20), analysed from frame 7 (warm-up)
    c1 = Track(1, b, history={f: b for f in range(7, 20)})
    warm_only = Track(2, b, history={f: Box(50, 50, 5, 5) for f in range(7, 9)})
    new = Track(3, b, history={f: Box(60, 60, 9, 9) for f in range(12, 20)})
    tracks, _ = stitch([((0, 10), _analysis([a1, gone])), ((10, 20), _analysis([c1, warm_only, new]))], Config())
    assert [t.id for t in tracks] == [1, 2, 3]
    assert set(tracks[0].history) == set(range(20))  # continued across the boundary
    assert min(tracks[2].history) == 12


def test_skip_is_exact():
    if not SYNTH.exists():
        pytest.skip("run scripts/make_synthetic_clip.py")
    with VideoReader(SYNTH) as r:
        frames = [f for _, f in zip(range(40), r)]
    with VideoReader(SYNTH) as r:
        assert r.skip(37)
        assert (next(iter(r)) == frames[37]).all()


@pytest.mark.skipif(not SYNTH.exists(), reason="run scripts/make_synthetic_clip.py")
def test_parallel_matches_sequential_on_the_er_clip():
    cfg = Config()
    seq = analyse_video(SYNTH, cfg)
    par = analyse_parallel(SYNTH, cfg, workers=4)
    assert par.stats.frames == seq.stats.frames == 600
    assert len(par.tracks) == len(seq.tracks)
    assert {f for t in par.tracks for f in t.history} == {f for t in seq.tracks for f in t.history}
