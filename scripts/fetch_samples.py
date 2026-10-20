"""Fetch public OpenCV test media into data/samples/ for development.

These are the classic MIT+CMU frontal-face test photos and OpenCV sample clips
that ship with OpenCV's own test data. They are fetched, not committed:
data/samples/ is git-ignored because real project clips will contain people.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"

_EXTRA = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/cascadeandhog/images/"
_CORE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"

FILES = {
    "addams-family.png": _EXTRA,  # 7 faces, mixed pose
    "class57.png": _EXTRA,  # 57 small faces, crowd case
    "er.png": _EXTRA,  # 6 faces
    "rehg-thanksgiving-1994.png": _EXTRA,  # 7 faces
    "karen-and-rob.png": _EXTRA,  # 2 faces
    "waynesworld2.png": _EXTRA,  # 2 faces
    "churchill-downs.png": _EXTRA,  # tiny faces, hard case
    "messi5.jpg": _CORE,  # 1 face
    "vtest.avi": _CORE,  # pedestrians, small faces, no audio
}


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, base in FILES.items():
        dest = SAMPLES / name
        if dest.exists():
            continue
        print(f"Fetching {name}")
        try:
            urllib.request.urlretrieve(base + name, dest)
        except OSError as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
