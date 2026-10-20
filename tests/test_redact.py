import numpy as np
import pytest

from core.redact import blur_kernel, build_mask, odd, redact
from core.types import Box, Config

W, H = 320, 240


def noise_frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def face_region(img: np.ndarray, b: Box) -> np.ndarray:
    return img[b.y : b.y + b.h, b.x : b.x + b.w]


@pytest.mark.parametrize("mode", ["blur", "pixelate", "fill"])
def test_redact_does_not_modify_input(mode):
    frame = noise_frame()
    before = frame.copy()
    out = redact(frame, [Box(100, 80, 60, 70)], Config(mode=mode))
    assert np.array_equal(frame, before)
    assert out.shape == frame.shape and out.dtype == np.uint8


@pytest.mark.parametrize("mode", ["blur", "pixelate", "fill"])
def test_pixels_far_from_face_untouched(mode):
    frame = noise_frame()
    out = redact(frame, [Box(100, 80, 60, 70)], Config(mode=mode))
    assert np.array_equal(out[:, :40], frame[:, :40])
    assert np.array_equal(out[:, 260:], frame[:, 260:])


def test_fill_covers_entire_face_box_at_default_padding():
    """Every pixel of the face box, corners included, is fully opaque."""
    face = Box(100, 80, 60, 70)
    out = redact(noise_frame(), [face], Config(mode="fill", fill_color=(0, 0, 0)))
    assert face_region(out, face).max() == 0


@pytest.mark.parametrize("face", [Box(0, 0, 60, 70), Box(W - 50, H - 60, 50, 60), Box(-20, 100, 60, 60)])
def test_fill_covers_faces_at_frame_border(face):
    out = redact(noise_frame(), [face], Config(mode="fill"))
    visible = face.clip(W, H)
    assert face_region(out, visible).max() == 0


def test_zero_padding_cannot_cover_face_box_corners():
    """Documents the geometry behind D-006: inscribed ellipse misses corners."""
    face = Box(100, 80, 60, 60)
    roi, alpha = build_mask(face, W, H, Config(padding=0.0))
    assert alpha[face.y - roi.y, face.x - roi.x] < 0.5


def test_mask_is_opaque_inside_and_transparent_outside():
    face = Box(100, 80, 60, 60)
    roi, alpha = build_mask(face, W, H, Config())
    cx, cy = face.x + 30 - roi.x, face.y + 30 - roi.y
    assert alpha[cy, cx] == 1.0
    assert alpha[0, 0] < 0.05  # ROI corner is outside the ellipse


def test_pixelate_reduces_detail():
    face = Box(100, 80, 80, 80)
    frame = noise_frame()
    out = redact(frame, [face], Config(mode="pixelate", pixel_blocks=8))
    inner = face_region(out, face).reshape(-1, 3)
    assert len(np.unique(inner, axis=0)) < 100  # noise has ~6400 unique colours


def test_blur_kernel_scales_with_face_and_is_odd():
    cfg = Config(blur_strength=0.35, blur_min_kernel=11)
    small, large = blur_kernel(Box(0, 0, 10, 10), cfg), blur_kernel(Box(0, 0, 200, 200), cfg)
    assert small == 11
    assert large == 71
    assert small % 2 == 1 and large % 2 == 1


def test_odd():
    assert [odd(x) for x in (0, 1, 2, 2.1, 9, 10)] == [1, 1, 3, 3, 9, 11]


def test_empty_face_list_returns_identical_copy():
    frame = noise_frame()
    assert np.array_equal(redact(frame, [], Config()), frame)
