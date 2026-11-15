import shutil
import subprocess

import numpy as np
import pytest

from core.errors import FaceGuardError
from core.redact import redact
from core.types import Box, Config
from core.vault import mask_regions, restore, seal
from core.video_io import voice_filter

PW = "a long passphrase"


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_vault_round_trip_is_exact():
    img, cfg = _img(), Config(mode="fill")
    boxes = [Box(20, 20, 40, 40), Box(100, 50, 30, 30)]
    red = redact(img, boxes, cfg)
    assert not np.array_equal(red, img)
    v = seal(img, red, mask_regions(boxes, 160, 120, cfg), PW)
    assert np.array_equal(restore(red, v, PW), img)


def test_vault_rejects_wrong_passphrase_other_image_and_tampering():
    img, cfg = _img(), Config(mode="fill")
    boxes = [Box(20, 20, 40, 40)]
    red = redact(img, boxes, cfg)
    v = seal(img, red, mask_regions(boxes, 160, 120, cfg), PW)
    with pytest.raises(FaceGuardError, match="passphrase"):
        restore(red, v, "not the passphrase")
    other = red.copy()
    other[0, 0] ^= 1
    with pytest.raises(FaceGuardError, match="different image"):
        restore(other, v, PW)
    tampered = v.replace(b'"ciphertext": "', b'"ciphertext": "AAAA')
    with pytest.raises(FaceGuardError):
        restore(red, tampered, PW)


def test_vault_needs_a_real_passphrase_and_hides_the_pixels():
    img, cfg = _img(), Config(mode="fill")
    boxes = [Box(20, 20, 40, 40)]
    with pytest.raises(FaceGuardError, match="at least"):
        seal(img, redact(img, boxes, cfg), mask_regions(boxes, 160, 120, cfg), "short")
    v = seal(img, redact(img, boxes, cfg), mask_regions(boxes, 160, 120, cfg), PW)
    assert img[20:60, 20:60].tobytes()[:32] not in v  # no plaintext pixels in the file


def test_voice_filters():
    assert voice_filter("keep", 0.75, 48000) is None
    assert voice_filter("mute", 0.75, None) == "volume=0"
    assert voice_filter("pitch", 0.75, 48000) == "asetrate=36000,aresample=48000,atempo=1.333333"
    assert voice_filter("pitch", 0.75, None) is None  # caller falls back to mute


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_pitch_shift_moves_a_tone_and_mute_silences(tmp_path):
    from core.pipeline import process_video

    src = tmp_path / "tone.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x120:d=1:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)

    def peak(path):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                              "-ar", "48000", "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, np.float32)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        return np.fft.rfftfreq(len(x), 1 / 48000)[spec.argmax()], float(np.abs(x).max())

    for voice, expect in (("keep", 440), ("pitch", 330)):
        out = tmp_path / f"{voice}.mp4"
        process_video(src, out, Config(keep_audio=True, voice=voice, voice_pitch=0.75))
        f, _ = peak(out)
        assert f == pytest.approx(expect, abs=8)
    out = tmp_path / "mute.mp4"
    process_video(src, out, Config(keep_audio=True, voice="mute"))
    assert peak(out)[1] < 1e-3
