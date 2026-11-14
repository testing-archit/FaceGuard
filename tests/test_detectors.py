from pathlib import Path

import cv2
import numpy as np
import pytest

from core.detectors import YuNetDetector
from core.types import Box, Config, Detection

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


def test_haar_blank_frame_returns_empty_list():
    from core.detectors import HaarDetector

    assert HaarDetector(Config(detector="haar")).detect(np.zeros((240, 320, 3), np.uint8)) == []


@pytest.mark.skipif(not (SAMPLES / "addams-family.png").exists(), reason="run scripts/fetch_samples.py")
def test_haar_same_interface_and_finds_faces():
    from core.detectors import make_detector

    det = make_detector(Config(detector="haar"))
    found = det.detect(cv2.imread(str(SAMPLES / "addams-family.png")))
    assert len(found) >= 3
    assert all(d.score == 1.0 and d.landmarks is None for d in found)


def test_tile_grid_covers_frame_with_overlap():
    from core.detectors import tile_grid

    tiles = tile_grid(100, 60, 2, 0.25)
    assert len(tiles) == 4
    covered = np.zeros((60, 100), bool)
    for x, y, w, h in tiles:
        covered[y : y + h, x : x + w] = True
    assert covered.all()
    (x0, _, w0, _), (x1, _, _, _) = tiles[0], tiles[1]
    assert x0 + w0 > x1  # neighbouring tiles overlap


def test_tiled_detector_maps_tile_boxes_back_and_merges():
    from core.detectors import TiledDetector

    class Fake:
        """Finds a 'face' only when it is at least 20 px wide: a bright square."""

        def detect(self, frame, gray=None):
            m = frame[..., 0] > 200
            ys, xs = np.nonzero(m)
            if len(xs) == 0 or xs.max() - xs.min() + 1 < 20:
                return []
            return [Detection(Box(int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)), 0.9)]

    frame = np.zeros((200, 200, 3), np.uint8)
    frame[150:162, 30:42] = 255  # 12 px face: too small for the base detector
    assert Fake().detect(frame) == []
    dets = TiledDetector(Fake(), Config(tile_grid=2, tile_scale=2.0)).detect(frame)
    assert len(dets) == 1  # found in one tile (or two overlapping ones, merged by NMS)
    b = dets[0].box
    assert abs(b.x - 30) <= 1 and abs(b.y - 150) <= 1 and abs(b.w - 12) <= 1


def test_nms_keeps_best_of_overlaps():
    from core.detectors import nms

    a, b, c = (Detection(Box(0, 0, 10, 10), 0.6), Detection(Box(1, 0, 10, 10), 0.9),
               Detection(Box(50, 50, 10, 10), 0.5))
    assert [d.score for d in nms([a, b, c], 0.3)] == [0.9, 0.5]
