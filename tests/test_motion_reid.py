import itertools
from pathlib import Path

import numpy as np
import pytest

from core.association import greedy_match, hungarian, hungarian_match
from core.detectors import ProfileFallback, make_detector
from core.motion import KalmanCV
from core.reid import link_tracks
from core.types import Box, Config, Detection, Track


def test_hungarian_is_optimal_against_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n, m = (int(x) for x in rng.integers(1, 5, 2))
        c = rng.random((n, m))
        got = sum(c[r, k] for r, k in hungarian(c))
        t = c if n <= m else c.T  # brute force over the shorter side
        best = min(sum(t[i, p[i]] for i in range(t.shape[0]))
                   for p in itertools.permutations(range(t.shape[1]), t.shape[0]))
        assert len(hungarian(c)) == min(n, m)
        assert got == pytest.approx(best)


def test_hungarian_beats_greedy_on_the_classic_case():
    """Greedy takes the single best pair (0.6) and strands the other track;
    optimal assignment matches both (0.55 + 0.55 > 0.6 + 0)."""
    tracks = [Box(0, 0, 100, 100), Box(60, 0, 100, 100)]
    dets = [Box(30, 0, 100, 100), Box(110, 0, 100, 100)]
    g_pairs, _, _ = greedy_match(tracks, dets, 0.3)
    h_pairs, ut, ud = hungarian_match(tracks, dets, 0.3)
    assert len(h_pairs) == 2 and not ut and not ud
    assert len(h_pairs) >= len(g_pairs)


def test_hungarian_match_respects_threshold():
    pairs, ut, ud = hungarian_match([Box(0, 0, 10, 10)], [Box(100, 100, 10, 10)], 0.3)
    assert pairs == [] and ut == [0] and ud == [0]


def test_kalman_learns_velocity_and_uncertainty_grows_when_coasting():
    kf = KalmanCV(0, 0, q=1.0, r=2.0)
    for k in range(1, 30):
        kf.predict()
        kf.update(5.0 * k, 0.0)
    assert kf.x[2] == pytest.approx(5.0, abs=0.3)
    s0 = kf.position_std
    for _ in range(10):
        kf.predict()
    assert kf.x[0] == pytest.approx(5.0 * 29 + 50, abs=5)
    assert kf.position_std > 3 * s0


def _track(tid, frames, x=0):
    return Track(id=tid, box=Box(x, 0, 10, 10), history={f: Box(x, 0, 10, 10) for f in frames})


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_link_tracks_merges_reappearing_face_only():
    a, b, c = _track(1, range(0, 10)), _track(2, range(30, 40)), _track(3, range(30, 40), x=500)
    emb = {1: _unit([1, 0, 0]), 2: _unit([0.95, 0.1, 0]), 3: _unit([0, 1, 0])}
    tracks, links = link_tracks([a, b, c], emb, Config(reid_threshold=0.5, reid_max_gap=50))
    assert [t.id for t in tracks] == [1, 3]
    assert links[0].kept == 1 and links[0].merged == 2 and links[0].gap == 21 and links[0].frame == 30
    assert set(tracks[0].history) == set(range(0, 10)) | set(range(30, 40))


def test_link_tracks_never_links_overlapping_or_distant_tracks():
    a, b = _track(1, range(0, 10)), _track(2, range(5, 15))
    far = _track(3, range(500, 510))
    emb = {1: _unit([1, 0]), 2: _unit([1, 0]), 3: _unit([1, 0])}
    tracks, links = link_tracks([a, b, far], emb, Config(reid_max_gap=150))
    assert links == [] and len(tracks) == 3


def test_link_tracks_keeps_hidden_if_either_was_hidden():
    a, b = _track(1, range(0, 10)), _track(2, range(20, 30))
    a.redact = False
    tracks, _ = link_tracks([a, b], {1: _unit([1, 0]), 2: _unit([1, 0])}, Config())
    assert tracks[0].redact


class _Fixed:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, gray=None):
        return list(self.dets)


@pytest.mark.skipif(not Path("models/haarcascade_profileface.xml").exists(), reason="profile cascade missing")
def test_profile_fallback_keeps_base_detection_on_overlap(monkeypatch):
    cfg = Config(profile=True)
    base = _Fixed([Detection(Box(10, 10, 50, 50), 0.9)])
    pf = ProfileFallback(base, cfg)
    monkeypatch.setattr(pf, "_profiles", lambda gray: [Box(12, 12, 50, 50), Box(200, 10, 40, 40)])
    out = pf.detect(np.zeros((100, 300, 3), np.uint8))
    assert sorted((d.box.x, d.score) for d in out) == [(10, 0.9), (200, cfg.profile_score)]
    assert isinstance(make_detector(cfg), ProfileFallback)


def test_link_tracks_uses_last_confirmed_frame_as_the_end():
    """An occluded face's track is dragged on (flagged guesses) past the moment
    its face re-emerges as a new track; they still link."""
    a, b = _track(1, range(0, 20)), _track(2, range(15, 30), x=40)
    a.flags = {f: "unconfirmed" for f in range(10, 20)}
    tracks, links = link_tracks([a, b], {1: _unit([1, 0]), 2: _unit([1, 0])}, Config())
    assert len(tracks) == 1 and links[0].gap == 6
    assert all(tracks[0].history[f].x == 40 for f in range(15, 30))
