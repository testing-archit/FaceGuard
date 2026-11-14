"""NFR-3: a full pipeline run makes zero network connections.

Every way of opening a socket is replaced with one that fails the test, then
the CLI processes an image and a video end to end (detect, track, redact,
encode). ffmpeg runs as a subprocess on local files only; its command line is
checked to contain no URL.
"""

import socket
import subprocess
from pathlib import Path

import pytest

import cli

PHOTO = Path("data/samples/er.png")
CLIP = Path("data/samples/synthetic_cross.mp4")


class NetworkUsed(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    real_run = subprocess.run
    commands = []

    def checked_run(cmd, *a, **kw):
        commands.append(cmd)
        if any("://" in str(part) for part in cmd):
            raise NetworkUsed(f"subprocess given a URL: {cmd}")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", checked_run)
    return commands


@pytest.mark.skipif(not PHOTO.exists(), reason="run scripts/fetch_samples.py")
def test_image_pipeline_is_offline(no_network, tmp_path):
    assert cli.main(["--input", str(PHOTO), "--output", str(tmp_path / "o.png"), "--mode", "fill"]) == 0


@pytest.mark.skipif(not CLIP.exists(), reason="run scripts/make_crossing_clip.py")
def test_video_pipeline_is_offline(no_network, tmp_path):
    out = tmp_path / "o.mp4"
    assert cli.main(["--input", str(CLIP), "--output", str(out), "--tracker", "lk", "--keep-audio"]) == 0
    assert out.stat().st_size > 0
