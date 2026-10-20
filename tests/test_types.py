import pytest

from core.types import Box, Config


def test_iou_identical_boxes_is_one():
    b = Box(10, 10, 50, 50)
    assert b.iou(b) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert Box(0, 0, 10, 10).iou(Box(20, 20, 10, 10)) == 0.0


def test_iou_touching_edges_is_zero():
    assert Box(0, 0, 10, 10).iou(Box(10, 0, 10, 10)) == 0.0


def test_iou_half_overlap():
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert Box(0, 0, 10, 10).iou(Box(5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_iou_contained_box():
    assert Box(0, 0, 10, 10).iou(Box(0, 0, 5, 5)) == pytest.approx(0.25)


def test_iou_is_symmetric():
    a, b = Box(3, 4, 20, 15), Box(10, 2, 9, 30)
    assert a.iou(b) == pytest.approx(b.iou(a))


def test_iou_zero_area_boxes():
    assert Box(0, 0, 0, 0).iou(Box(0, 0, 0, 0)) == 0.0


def test_pad_is_per_side():
    assert Box(100, 100, 40, 20).pad(0.25, 1000, 1000) == Box(90, 95, 60, 30)


def test_pad_zero_is_identity():
    b = Box(5, 6, 7, 8)
    assert b.pad(0.0, 100, 100) == b


def test_pad_clips_at_top_left_border():
    assert Box(2, 3, 40, 40).pad(0.25, 640, 480) == Box(0, 0, 52, 53)


def test_pad_clips_at_bottom_right_border():
    padded = Box(600, 440, 40, 40).pad(0.25, 640, 480)
    assert padded == Box(590, 430, 50, 50)
    assert padded.x + padded.w == 640 and padded.y + padded.h == 480


def test_pad_box_larger_than_frame_fills_frame():
    assert Box(0, 0, 100, 100).pad(0.5, 80, 60) == Box(0, 0, 80, 60)


def test_clip_box_fully_outside_collapses():
    assert Box(700, 500, 10, 10).clip(640, 480).area == 0


def test_center():
    assert Box(10, 20, 30, 40).center == (25.0, 40.0)


def test_from_xywh_rounds():
    assert Box.from_xywh(1.4, 1.6, 10.5, 9.49) == Box(1, 2, 10, 9)


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Config(mode="sharpen")


def test_config_rejects_zero_interval():
    with pytest.raises(ValueError):
        Config(detect_interval=0)
