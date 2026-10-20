"""Fetch FaceGuard's two detector models and verify them.

This is the only part of FaceGuard that touches the network, and it runs once
during setup. Both files are also committed to `models/`, so this script only
matters if one is missing or corrupted.

- YuNet ONNX from OpenCV Zoo (MIT licence)
- Haar frontal-face cascade from the OpenCV repo (Intel licence, see the
  header of the XML). OpenCV 5 wheels no longer bundle the cascades.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str  # pinned to an exact commit so the bytes can never change
    sha256: str
    size: int


ASSETS = [
    Asset(
        name="face_detection_yunet_2026may.onnx",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/"
        "face_detection_yunet_2026may.onnx",
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        size=229_738,
    ),
    Asset(
        name="haarcascade_frontalface_default.xml",
        url="https://raw.githubusercontent.com/opencv/opencv/"
        "b7553d4e2eddc5645023a94a850fe092dd9f223e/data/haarcascades/"
        "haarcascade_frontalface_default.xml",
        sha256="0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
        size=930_127,
    ),
]


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, asset: Asset) -> bool:
    """True when the file exists and matches the pinned size and hash."""
    return path.is_file() and path.stat().st_size == asset.size and sha256_of(path) == asset.sha256


def fetch(asset: Asset) -> bool:
    dest = MODELS / asset.name
    if is_valid(dest, asset):
        print(f"{asset.name} already present and verified.")
        return True

    tmp = dest.with_suffix(".part")
    print(f"Downloading {asset.name} ...")
    try:
        urllib.request.urlretrieve(asset.url, tmp)
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return False
    if not is_valid(tmp, asset):
        tmp.unlink(missing_ok=True)
        print(f"{asset.name} failed the SHA-256 check. Not installing it.", file=sys.stderr)
        return False
    tmp.replace(dest)
    print(f"Saved and verified {dest}")
    return True


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    results = [fetch(a) for a in ASSETS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
