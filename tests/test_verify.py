import json
from pathlib import Path

import pytest

from core.pipeline import analyse_image, process_video, read_image, render_image
from core.types import Box, Config
from core.verify import VerifyReport, check_frame, verify_image, verify_video

PHOTO = Path("data/samples/addams-family.png")
CROSS = Path("data/samples/synthetic_cross.mp4")


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame, gray=None):
        from core.types import Detection

        return [Detection(b, 0.9) for b in self.boxes]


def test_check_frame_splits_residual_and_unmasked_by_centre():
    det = FakeDetector([Box(10, 10, 20, 20), Box(200, 200, 20, 20)])
    r, u = check_frame(None, [Box(0, 0, 50, 50)], det, frame=3)
    assert [h.box for h in r] == [Box(10, 10, 20, 20)] and r[0].frame == 3
    assert [h.box for h in u] == [Box(200, 200, 20, 20)]


def test_report_summary():
    assert VerifyReport(frames_checked=2).ok
    assert "no face" in VerifyReport(frames_checked=2).summary()


@pytest.mark.skipif(not PHOTO.exists(), reason="run scripts/fetch_samples.py")
def test_fill_hides_and_unredacted_is_caught():
    img = read_image(PHOTO)
    boxes = [d.box for d in analyse_image(img, Config())]
    assert verify_image(render_image(img, boxes, Config(mode="fill")), boxes, Config()).ok
    weak = verify_image(img, boxes, Config())  # nothing redacted: every face is a residual
    assert len(weak.residuals) == len(boxes)


@pytest.mark.skipif(not CROSS.exists(), reason="run scripts/make_crossing_clip.py")
def test_verify_video_default_export_is_clean(tmp_path):
    a = process_video(CROSS, tmp_path / "o.mp4", Config())
    rep = verify_video(tmp_path / "o.mp4", a.tracks, Config(), every=5)
    assert rep.frames_checked == 90 and rep.hidden_faces > 0 and rep.ok
