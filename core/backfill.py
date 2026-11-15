"""Backward backfill: cover the frames before a new face was first detected.

A face that enters between two scheduled detections is visible, unmasked,
until the detector runs again (the "entry exposure" measured in E3). The
detection that finally finds it fixes the future, but the frames already
seen are still exposed.

Because those frames are still held in a short buffer, the tracker can be
run *backwards in time*: initialise it on the new detection, then feed it
the buffered frames newest-first. Optical flow does not care about the
direction of time (brightness constancy holds just as well from frame t to
t-1), so the box follows the face back towards where it entered, until the
tracker loses it, the face leaves the frame, or a scene cut is reached.

Backfilled boxes are tracker guesses with no detection behind them, so they
are flagged `FLAG_BACKFILL` for review (DECISIONS D-026).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from core.tracking import Tracker
from core.types import FLAG_BACKFILL, Box, Config, Track


@dataclass
class BufferedFrame:
    index: int
    frame: np.ndarray
    gray: np.ndarray | None


class FrameBuffer:
    """The last `size` decoded frames, oldest first."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._q: deque[BufferedFrame] = deque()

    def push(self, item: BufferedFrame) -> BufferedFrame | None:
        """Add the newest frame; return the frame that fell out, if any."""
        self._q.append(item)
        return self._q.popleft() if len(self._q) > self.size else None

    def drain(self) -> list[BufferedFrame]:
        out = list(self._q)
        self._q.clear()
        return out

    def newest_first(self) -> list[BufferedFrame]:
        return list(reversed(self._q))

    def __len__(self) -> int:
        return len(self._q)


def visible_fraction(box: Box, frame_w: int, frame_h: int) -> float:
    """Share of the box that lies inside the frame (1 = fully inside)."""
    return box.clip(frame_w, frame_h).area / box.area if box.area > 0 else 0.0


def backfill_track(
    track: Track,
    birth: BufferedFrame,
    past: list[BufferedFrame],
    tracker: Tracker,
    cfg: Config,
    floor: int = 0,
) -> int:
    """Run `tracker` backwards from the track's first frame over `past` (newest first).

    Stops at the first frame where the tracker loses the face, the box is
    mostly outside the frame (the face had not entered yet), the frame is
    before `floor` (the last scene cut), or the track already has a box.
    Returns the number of frames filled.
    """
    h, w = birth.frame.shape[:2]
    tracker.init(birth.frame, birth.gray, track.box)
    filled = 0
    for bf in past:
        if bf.index < floor or bf.index in track.history:
            break
        box = tracker.update(bf.frame, bf.gray)
        if box is None or visible_fraction(box, w, h) < cfg.backfill_min_visible:
            break
        track.history[bf.index] = box.clip(w, h)
        track.flags[bf.index] = FLAG_BACKFILL
        filled += 1
    return filled
