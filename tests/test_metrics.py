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
    # Tiny feather isolates the geometry; the default outward feather pushes
    # the alpha >= 0.5 edge a few pixels beyond the ellipse (~0.83 here).
    g = Box(100, 100, 200, 200)
    c = coverage(g, [g], W, H, Config(padding=0.0, feather_frac=0.005))
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


def test_match_boxes_one_to_one():
    from eval.metrics import count_detections, match_boxes
    g = [Box(0, 0, 10, 10), Box(100, 100, 10, 10)]
    p = [Box(1, 0, 10, 10), Box(0, 1, 10, 10), Box(300, 300, 5, 5)]
    assert len(match_boxes(g, p)) == 1
    c = count_detections(g, p)
    assert (c.tp, c.fp, c.fn) == (1, 2, 1)
    assert c.precision == pytest.approx(1 / 3) and c.recall == pytest.approx(0.5)


def test_pr_curve_ranks_by_score():
    from eval.metrics import pr_curve
    g = [Box(0, 0, 10, 10), Box(50, 50, 10, 10)]
    imgs = [(g, [(Box(0, 0, 10, 10), 0.9), (Box(200, 200, 10, 10), 0.8), (Box(50, 50, 10, 10), 0.3)])]
    scores, prec, rec, ap = pr_curve(imgs)
    assert list(scores) == [0.9, 0.8, 0.3]
    assert list(rec) == [0.5, 0.5, 1.0]
    assert prec[0] == 1.0 and prec[1] == 0.5 and prec[2] == pytest.approx(2 / 3)
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_size_bucket():
    from eval.metrics import size_bucket
    assert size_bucket(Box(0, 0, 20, 20)) == "small"
    assert size_bucket(Box(0, 0, 60, 60)) == "medium"
    assert size_bucket(Box(0, 0, 120, 120)) == "large"
