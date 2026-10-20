"""User-facing errors. Anything raised as FaceGuardError is shown as a message, never a traceback."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".avi", ".mov")


class FaceGuardError(Exception):
    """An expected failure with a message meant for the user."""


def media_kind(path: str | Path) -> str:
    """Return "image" or "video" from the file extension, or raise FaceGuardError."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    supported = ", ".join(IMAGE_EXTS + VIDEO_EXTS)
    raise FaceGuardError(f"Unsupported file type {ext or '(none)'!r}. Supported: {supported}")
