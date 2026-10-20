import pytest

from core.types import Box, Config, Track
from eval.metrics import coverage, evaluate_video, load_video_truth

W, H = 640, 480


def test_coverage_perfect_box_at_default_padding_is_full():
    g = Box(100, 100, 60, 60)
    assert coverage(g, [g], W, H, Config()) == 1.0


def test_coverage_no_prediction_is_zero():
    assert coverage(Box(100, 100, 60, 60), [], W, H, Config()) == 0.0


def test_coverage_zero_padding_caps_at_ellipse_area():
    g = Box(100, 100, 200, 200)
    c = coverage(g, [g], W, H, Config(padding=0.0))
    assert c == pytest.approx(3.14159 / 4, abs=0.02)


def test_coverage_union_of_two_half_boxes():
    g = Box(100, 100, 100, 100)
    left, right = Box(100, 100, 50, 100), Box(150, 100, 50, 100)
    assert coverage(g, [left, right], W, H, Config()) > coverage(g, [left], W, H, Config())


def track(tid, boxes):
    return Track(id=tid, box=next(iter(boxes.values())), history=boxes)


def test_leak_frames_counted_per_frame():
    gt = {f: [(0, Box(100, 100, 50, 50))] for f in range(10)}
    t = track(1, {f: Box(100, 100, 50, 50) for f in range(10) if f not in (3, 7)})
    m = evaluate_video([t], gt, W, H, Config())
    assert m.leak_frames == 2 and m.leak_frame_ids == [3, 7]
    assert m.recall == pytest.approx(0.8)
    assert m.precision == 1.0


def test_untoggled_track_does_not_count_as_coverage():
    gt = {0: [(0, Box(100, 100, 50, 50))]}
    t = track(1, {0: Box(100, 100, 50, 50)})
    t.redact = False
    assert evaluate_video([t], gt, W, H, Config()).leak_frames == 1


def test_id_switch_and_consistency():
    gt = {f: [(0, Box(100, 100, 50, 50))] for f in range(4)}
    a = track(1, {0: Box(100, 100, 50, 50), 1: Box(100, 100, 50, 50)})
    b = track(2, {2: Box(100, 100, 50, 50), 3: Box(100, 100, 50, 50)})
    m = evaluate_video([a, b], gt, W, H, Config())
    assert m.id_switches == 1
    assert m.id_consistency == 1.0  # each track only ever covered person 0


def test_interpolation_between_keyframes():
    entry = {
        "interpolate": True,
        "frames": {"0": [{"id": 0, "box": [0, 0, 10, 10]}], "4": [{"id": 0, "box": [40, 0, 10, 10]}]},
    }
    gt = load_video_truth(entry)
    assert gt[2] == [(0, Box(20, 0, 10, 10))]
    assert sorted(gt) == [0, 1, 2, 3, 4]
