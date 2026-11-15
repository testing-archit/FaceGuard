from pathlib import Path

import cv2
import numpy as np
import pytest

from core.pipeline import render_frame, text_boxes_at, boxes_at
from core.text import TEXT_SOURCE, TextDetector, TextTracks
from core.types import Box, Config, Track

MODEL = Path(Config().text_model_path)
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/download_model.py")


def plate_image():
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(img, (100, 180), (460, 250), (255, 255, 255), -1)
    cv2.putText(img, "MH 12 AB 1234", (110, 232), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


@needs_model
def test_detector_finds_the_plate_and_maps_back_to_frame_coordinates():
    det = TextDetector(Config())
    assert det.input_size(1920, 1080) == (1280, 704)  # long side capped, multiples of 32
    boxes = det.detect(plate_image())
    assert len(boxes) == 1
    img = plate_image()
    ink = (img == 0).all(axis=2)  # every pixel of every glyph
    out = render_frame(img, [], boxes, Config(mode="fill", fill_color=(255, 0, 0)))
    assert (out[ink] == (255, 0, 0)).all()  # the padded rectangle covers all of them


@needs_model
def test_rendered_text_is_unreadable_and_rectangular():
    img = plate_image()
    boxes = TextDetector(Config()).detect(img)
    out = render_frame(img, [], boxes, Config(mode="fill"))
    assert TextDetector(Config()).detect(out) == []
    b = boxes[0]
    assert (out[b.y, b.x] == 0).all() and (out[b.y + b.h - 1, b.x + b.w - 1] == 0).all()  # corners too


class _FakeDet:
    def __init__(self, seq):
        self.seq = iter(seq)

    def detect(self, frame):
        return next(self.seq)


@needs_model
def test_text_tracks_hold_match_and_stamp_back():
    ids = iter(range(100, 200))
    tt = TextTracks(Config(), lambda: next(ids))
    tt._det = _FakeDet([[Box(10, 10, 50, 20)], [Box(12, 10, 50, 20), Box(300, 300, 40, 20)]])
    tt.step(0, None, True, [])
    for i in range(1, 10):
        tt.step(i, None, False, [])
    tt.step(10, None, True, list(range(9, -1, -1)))
    a, b = sorted(tt.all_tracks(), key=lambda t: t.id)
    assert a.id == 100 and set(a.history) == set(range(11))  # held, then matched (same ID)
    assert b.source == TEXT_SOURCE and set(b.history) == set(range(1, 11))  # stamped back to the last detection


def test_face_and_text_boxes_are_kept_apart():
    tracks = [Track(1, Box(0, 0, 10, 10), history={0: Box(0, 0, 10, 10)}),
              Track(2, Box(20, 0, 10, 10), history={0: Box(20, 0, 10, 10)}, source=TEXT_SOURCE)]
    assert boxes_at(tracks, 0) == [Box(0, 0, 10, 10)]
    assert text_boxes_at(tracks, 0) == [Box(20, 0, 10, 10)]
