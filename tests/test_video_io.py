import shutil
import subprocess

import cv2
import numpy as np
import pytest

from core import video_io
from core.video_io import VideoInfo, VideoReader, VideoWriter, finalise, has_audio, limit_warnings

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_clip(path, frames=12, size=(64, 48), fps=12.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    w.release()


def add_tone(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst)],
        check=True,
    )


def test_reader_streams_all_frames(tmp_path):
    clip = tmp_path / "c.mp4"
    make_clip(clip)
    with VideoReader(clip) as r:
        assert (r.info.width, r.info.height) == (64, 48)
        assert r.info.fps == pytest.approx(12.0)
        assert sum(1 for _ in r) == 12


def test_reader_rejects_garbage_with_message(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(video_io.FaceGuardError, match="Could not open video"):
        VideoReader(bad)


def test_limit_warnings():
    assert limit_warnings(VideoInfo(1280, 720, 30.0, 600)) == []
    assert len(limit_warnings(VideoInfo(3840, 2160, 30.0, 3000))) == 2


@needs_ffmpeg
def test_finalise_keeps_audio_and_size(tmp_path):
    src = tmp_path / "src.mp4"
    make_clip(tmp_path / "raw.mp4")
    add_tone(tmp_path / "raw.mp4", src)
    dst = tmp_path / "out.mp4"
    with VideoReader(src) as r, VideoWriter(dst, r.info) as w:
        for f in r:
            w.write(f)
    assert finalise(w.path, dst, src) == []
    assert has_audio(dst)
    assert not w.path.exists()
    with VideoReader(dst) as r:
        assert (r.info.width, r.info.height) == (64, 48)


@needs_ffmpeg
def test_finalise_without_keep_audio_is_silent(tmp_path):
    src = tmp_path / "src.mp4"
    make_clip(tmp_path / "raw.mp4")
    add_tone(tmp_path / "raw.mp4", src)
    dst = tmp_path / "out.mp4"
    with VideoReader(src) as r, VideoWriter(dst, r.info) as w:
        for f in r:
            w.write(f)
    finalise(w.path, dst, None)
    assert not has_audio(dst)


def test_finalise_without_ffmpeg_warns_and_keeps_file(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io, "ffmpeg_available", lambda: False)
    silent = tmp_path / "o.silent.mp4"
    make_clip(silent)
    dst = tmp_path / "o.mp4"
    warnings = finalise(silent, dst, audio_src=tmp_path / "whatever.mp4")
    assert dst.exists()
    assert any("audio was not preserved" in w for w in warnings)
