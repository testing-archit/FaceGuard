from pathlib import Path

import cv2
import numpy as np
import pytest

import cli

SAMPLE = Path("data/samples/addams-family.png")


def test_unsupported_input_is_a_message_not_a_traceback(tmp_path, capsys):
    src = tmp_path / "notes.txt"
    src.write_text("hi")
    assert cli.main(["--input", str(src), "--output", str(tmp_path / "o.png")]) == 1
    assert "Unsupported file type" in capsys.readouterr().err


def test_refuses_to_overwrite_source(tmp_path, capsys):
    src = tmp_path / "a.png"
    cv2.imwrite(str(src), np.zeros((8, 8, 3), np.uint8))
    assert cli.main(["--input", str(src), "--output", str(src)]) == 1
    assert "never modifies the source" in capsys.readouterr().err


@pytest.mark.skipif(not SAMPLE.exists(), reason="run scripts/fetch_samples.py")
@pytest.mark.parametrize("mode", ["blur", "pixelate", "fill"])
def test_image_export_keeps_resolution(tmp_path, mode):
    out = tmp_path / f"out_{mode}.png"
    assert cli.main(["--input", str(SAMPLE), "--output", str(out), "--mode", mode]) == 0
    assert cv2.imread(str(out)).shape == cv2.imread(str(SAMPLE)).shape


SYNTH = Path("data/samples/synthetic_er.mp4")


@pytest.mark.skipif(not SYNTH.exists(), reason="run scripts/make_synthetic_clip.py")
def test_video_export_keeps_size_fps_and_audio(tmp_path):
    from core.video_io import VideoReader, has_audio

    out = tmp_path / "out.mp4"
    assert cli.main(["--input", str(SYNTH), "--output", str(out), "--keep-audio"]) == 0
    with VideoReader(SYNTH) as a, VideoReader(out) as b:
        assert (a.info.width, a.info.height, a.info.fps) == (b.info.width, b.info.height, b.info.fps)
        assert b.info.frame_count == a.info.frame_count
    assert has_audio(out)


def test_batch_folder(tmp_path):
    import cv2
    import numpy as np

    from cli import main

    src = tmp_path / "in"
    src.mkdir()
    for name in ("a.png", "b.jpg"):
        cv2.imwrite(str(src / name), np.full((60, 80, 3), 128, np.uint8))
    (src / "notes.txt").write_text("ignored")
    (src / "broken.png").write_bytes(b"not an image")
    dst = tmp_path / "out"
    assert main(["--input", str(src), "--output", str(dst)]) == 1  # broken.png fails, rest continue
    assert sorted(p.name for p in dst.iterdir()) == ["a.png", "b.jpg"]


def test_batch_refuses_same_folder(tmp_path):
    from cli import main

    (tmp_path / "a.png").write_bytes(b"x")
    assert main(["--input", str(tmp_path), "--output", str(tmp_path)]) == 1
