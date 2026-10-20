"""Fetch the YuNet face detector from OpenCV Zoo and verify it.

This is the only part of FaceGuard that touches the network, and it runs once
during setup. The model is also committed to `models/`, so this script only
matters if that file is missing or corrupted.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Pinned to an exact opencv_zoo commit so the bytes can never change under us.
ZOO_COMMIT = "26cc381e4d2594bb9f47a26eb8fd96c94a13660d"
MODEL_NAME = "face_detection_yunet_2026may.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{ZOO_COMMIT}/models/face_detection_yunet/{MODEL_NAME}"
)
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
MODEL_SIZE = 229_738

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path) -> bool:
    """True when the file exists and matches the pinned size and hash."""
    return (
        path.is_file()
        and path.stat().st_size == MODEL_SIZE
        and sha256_of(path) == MODEL_SHA256
    )


def main() -> int:
    if is_valid(MODEL_PATH):
        print(f"{MODEL_PATH.name} already present and verified.")
        return 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    print(f"Downloading {MODEL_NAME} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    if not is_valid(tmp):
        tmp.unlink(missing_ok=True)
        print(
            "Downloaded file failed the SHA-256 check. Not installing it.",
            file=sys.stderr,
        )
        return 1

    tmp.replace(MODEL_PATH)
    print(f"Saved and verified {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
