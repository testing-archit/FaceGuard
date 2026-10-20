"""Streaming video IO and ffmpeg finalisation.

Frames are decoded one at a time and never accumulated (NFR-4).
`cv2.VideoWriter` writes video only, and its portable mp4v codec does not play
in browsers, so output is written to a temporary silent file and ffmpeg
re-encodes it to H.264, optionally muxing the original audio back in.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.errors import FaceGuardError

# FR-4: warn, but proceed, above these.
WARN_SECONDS = 60
WARN_PIXELS = 1920 * 1080
FALLBACK_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int  # container estimate; may be off by a few frames

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Iterate BGR frames from a video file, one at a time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FaceGuardError(
                f"Could not open video {self.path}. The file may be corrupted or use a "
                "codec OpenCV cannot decode; try re-encoding it to H.264 MP4."
            )
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps and fps > 0 else FALLBACK_FPS,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def limit_warnings(info: VideoInfo) -> list[str]:
    """FR-4: long or very large videos are processed, but the user is told."""
    warnings = []
    if info.duration > WARN_SECONDS:
        warnings.append(f"Video is {info.duration:.0f} s long; processing may be slow.")
    if info.width * info.height > WARN_PIXELS:
        warnings.append(f"Video is {info.width}x{info.height}, above 1080p; processing may be slow.")
    return warnings


def silent_path(dst: str | Path) -> Path:
    """Temporary file the OpenCV writer produces before ffmpeg finalises it."""
    dst = Path(dst)
    return dst.with_name(dst.stem + ".silent.mp4")


class VideoWriter:
    """Write BGR frames to a temporary silent mp4v file."""

    def __init__(self, dst: str | Path, info: VideoInfo) -> None:
        self.path = silent_path(dst)
        self._size = (info.width, info.height)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, self._size
        )
        if not self._writer.isOpened():
            raise FaceGuardError(f"Could not create output video at {self.path}.")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def has_audio(path: str | Path) -> bool:
    """True if the file has an audio stream (requires ffprobe; False if absent)."""
    if shutil.which("ffprobe") is None:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def finalise(silent: str | Path, dst: str | Path, audio_src: str | Path | None) -> list[str]:
    """Turn the silent mp4v file into the final output.

    Re-encodes to H.264 / yuv420p so it plays in browsers, and maps the first
    audio stream of `audio_src` if given (the trailing `?` makes a missing
    audio stream non-fatal). Without ffmpeg, keeps the mp4v file and warns.
    Returns warnings for the user.
    """
    silent, dst = Path(silent), Path(dst)
    if not ffmpeg_available():
        silent.replace(dst)
        warnings = ["ffmpeg not found: exported mp4v video (plays in VLC, may not preview in a browser)."]
        if audio_src is not None:
            warnings.append("ffmpeg not found: audio was not preserved.")
        return warnings

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent)]
    if audio_src is not None:
        cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FaceGuardError(f"ffmpeg failed while writing {dst}: {result.stderr.strip()[-400:]}")
    silent.unlink(missing_ok=True)

    if audio_src is not None and not has_audio(audio_src):
        return ["Input has no audio track; exported silent video."]
    return []
