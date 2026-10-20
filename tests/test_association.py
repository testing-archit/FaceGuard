import numpy as np

from core.association import TrackManager, greedy_match
from core.tracking import NullTracker
from core.types import Box, Config, Detection

FRAME = np.zeros((240, 320, 3), np.uint8)
GRAY = np.zeros((240, 320), np.uint8)


def det(x, y, w=40, h=40):
    return Detection(Box(x, y, w, h), 0.9)


def test_greedy_prefers_highest_iou():
    tracks = [Box(0, 0, 10, 10), Box(100, 0, 10, 10)]
    dets = [Box(101, 0, 10, 10), Box(1, 0, 10, 10)]
    pairs, ut, ud = greedy_match(tracks, dets, 0.3)
    assert sorted(pairs) == [(0, 1), (1, 0)] and ut == [] and ud == []


def test_greedy_rejects_below_threshold():
    pairs, ut, ud = greedy_match([Box(0, 0, 10, 10)], [Box(8, 8, 10, 10)], 0.3)
    assert pairs == [] and ut == [0] and ud == [0]


def test_greedy_one_detection_two_tracks():
    pairs, ut, ud = greedy_match([Box(0, 0, 10, 10), Box(2, 0, 10, 10)], [Box(1, 0, 10, 10)], 0.3)
    assert len(pairs) == 1 and len(ut) == 1 and ud == []


def run(cfg, detections_per_frame):
    tm = TrackManager(cfg, NullTracker)
    present = []
    for i, dets in enumerate(detections_per_frame):
        present.append([t.id for t in tm.step(i, FRAME, GRAY, dets)])
    return tm, present


def test_ids_are_stable_across_frames():
    tm, present = run(Config(), [[det(10, 10), det(200, 10)], [det(12, 10), det(202, 10)], [det(14, 10), det(204, 10)]])
    assert present == [[1, 2], [1, 2], [1, 2]]


def test_grace_period_keeps_track_redacted_then_deletes():
    cfg = Config(max_misses=3, tracker="none")
    frames = [[det(10, 10)]] + [[] for _ in range(5)]
    tm, present = run(cfg, frames)
    # frames 1-3 are within grace (i - 0 <= 3), frame 4 exceeds it
    assert present == [[1], [1], [1], [1], [], []]
    assert tm.all_tracks()[0].history.keys() == {0, 1, 2, 3}


def test_grace_measured_in_frames_not_detector_runs():
    cfg = Config(max_misses=15, detect_interval=10, tracker="none")
    frames = [[det(10, 10)]] + [None] * 9 + [[]] + [None] * 9 + [[]]
    tm, present = run(cfg, frames)
    assert present[10] == [1]  # 10 frames since seen: within 15
    assert present[20] == []  # 20 frames: deleted


def test_off_mode_has_no_memory():
    tm, present = run(Config(tracker="off"), [[det(10, 10)], [], [det(10, 10)]])
    assert present == [[1], [], [2]]  # a miss ends the track; no grace


def test_rematch_resets_grace():
    cfg = Config(max_misses=2, tracker="none")
    tm, present = run(cfg, [[det(10, 10)], [], [], [det(10, 10)], [], []])
    assert present == [[1], [1], [1], [1], [1], [1]]


def test_duplicates_merge_into_older_id_and_are_logged():
    cfg = Config(tracker="none", iou_threshold=0.9, duplicate_iou=0.7)
    # second detection overlaps the first too little to match (iou 0.9) but
    # enough to be the same face (iou > 0.7)
    tm, present = run(cfg, [[det(10, 10)], [det(10, 10), det(12, 10)]])
    assert present[1] == [1]
    assert len(tm.events) == 1 and tm.events[0].kept == 1 and tm.events[0].dropped == 2


class FollowingTracker(NullTracker):
    """Stands in for LK/CSRT: always reports the face as still there."""


class LosingTracker(NullTracker):
    """Reports the target lost from the first update."""

    def update(self, frame, gray):
        return None


def run_with(cfg, factory, detections_per_frame):
    tm = TrackManager(cfg, factory)
    return [[t.id for t in tm.step(i, FRAME, GRAY, d)] for i, d in enumerate(detections_per_frame)]


def test_followed_track_outlives_max_misses_until_unconfirmed_cap():
    """Detector misses a rotated face, but the tracker still has it (D-012)."""
    cfg = Config(tracker="lk", max_misses=3, max_unconfirmed=6)
    present = run_with(cfg, FollowingTracker, [[det(10, 10)]] + [[] for _ in range(8)])
    assert present == [[1]] * 7 + [[], []]  # kept through frame 6, gone at 7


def test_lost_track_gets_short_grace_from_moment_lost():
    cfg = Config(tracker="lk", max_misses=2, max_unconfirmed=60)
    present = run_with(cfg, LosingTracker, [[det(10, 10)]] + [[] for _ in range(5)])
    # lost on frame 1; kept while i - 1 <= 2
    assert present == [[1], [1], [1], [1], [], []]
