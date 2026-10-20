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
