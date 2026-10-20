from pathlib import Path

import cv2
import numpy as np
import pytest

from core.detectors import YuNetDetector
from core.types import Config

SAMPLES = Path("data/samples")


def test_blank_frame_returns_empty_list():
    det = YuNetDetector(Config())
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_missing_model_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        YuNetDetector(Config(model_path="models/nope.onnx"))


@pytest.mark.skipif(not (SAMPLES / "messi5.jpg").exists(), reason="run scripts/fetch_samples.py")
def test_input_size_follows_resolution():
    """Same image at two resolutions must give proportionally scaled boxes."""
    det = YuNetDetector(Config())
    img = cv2.imread(str(SAMPLES / "messi5.jpg"))
    big = cv2.resize(img, None, fx=2, fy=2)
    (small_face,) = det.detect(img)
    (big_face,) = det.detect(big)
    # A stale input size would put the box somewhere else entirely; box width
    # itself varies ~10% across scales, which is normal regression noise.
    (sx, sy), (bx, by) = small_face.box.center, big_face.box.center
    assert bx == pytest.approx(2 * sx, rel=0.03)
    assert by == pytest.approx(2 * sy, rel=0.03)
    assert big_face.box.w == pytest.approx(2 * small_face.box.w, rel=0.15)
    assert small_face.landmarks.shape == (5, 2)
    assert 0.0 < small_face.score <= 1.0
