"""Data association: which detection belongs to which existing track.

Greedy IoU matching plus the track lifecycle (create, re-anchor, grace
period, delete, merge duplicates). Stable track IDs are what make selective
redaction work: the user toggles person 3 once and the choice follows them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.tracking import Tracker, TrackerFactory
from core.types import Box, Config, Detection, Track


def greedy_match(
    track_boxes: list[Box], det_boxes: list[Box], iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match tracks to detections, highest IoU first.

    Returns (pairs of (track_idx, det_idx), unmatched track indices,
    unmatched detection indices). Pairs below `iou_threshold` never match.
    Greedy rather than Hungarian: no scipy dependency, and with the handful
    of faces per frame here the two almost always agree (DECISIONS D-009).
    """
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    iou = np.array([[t.iou(d) for d in det_boxes] for t in track_boxes])
    pairs: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        ti, di = divmod(int(flat), len(det_boxes))
        if iou[ti, di] < iou_threshold:
            break
        if ti in used_t or di in used_d:
            continue
        pairs.append((ti, di))
        used_t.add(ti)
        used_d.add(di)
    unmatched_t = [i for i in range(len(track_boxes)) if i not in used_t]
    unmatched_d = [i for i in range(len(det_boxes)) if i not in used_d]
    return pairs, unmatched_t, unmatched_d


@dataclass(frozen=True)
class IdEvent:
    """A suspected identity problem, logged for the evaluation report."""

    frame: int
    kind: str  # "duplicate": two tracks converged on one face; the younger was dropped
    kept: int
    dropped: int


class TrackManager:
    """Owns every track and its tracker for one video.

    Call `step` once per frame, in order. On detection frames pass the
    detections; on other frames pass None.
    """

    def __init__(self, cfg: Config, make_tracker: TrackerFactory) -> None:
        self._cfg = cfg
        self._make_tracker = make_tracker
        self._hold = cfg.tracker != "off"  # "off" = no memory between frames
        self.live: list[Track] = []
        self.finished: list[Track] = []
        self.events: list[IdEvent] = []
        self._trackers: dict[int, Tracker] = {}
        self._next_id = 1

    def step(
        self, i: int, frame: np.ndarray, gray: np.ndarray, detections: list[Detection] | None
    ) -> list[Track]:
        """Advance all tracks to frame `i`; return the tracks present on it."""
        # 1. Propagate every track to this frame first (DECISIONS D-008), so
        #    detections are matched against where faces are now.
        for t in self.live:
            box = self._trackers[t.id].update(frame, gray)
            if box is None:
                t.lost = True  # box stays frozen at its last position
            else:
                t.box = box

        matched_ids: set[int] = set()
        if detections is not None:
            matched_ids = self._associate(i, frame, gray, detections)

        if self._hold:
            present = list(self.live)
        else:
            present = [t for t in self.live if t.id in matched_ids]
        for t in present:
            t.history[i] = t.box
        return present

    def _associate(
        self, i: int, frame: np.ndarray, gray: np.ndarray, detections: list[Detection]
    ) -> set[int]:
        cfg = self._cfg
        pairs, unmatched_t, unmatched_d = greedy_match(
            [t.box for t in self.live], [d.box for d in detections], cfg.iou_threshold
        )
        matched_ids = set()
        for ti, di in pairs:
            t = self.live[ti]
            t.box = detections[di].box  # re-anchor: resets accumulated drift
            t.last_seen_frame = i
            t.lost = False
            self._trackers[t.id].init(frame, gray, t.box)
            matched_ids.add(t.id)

        grace = cfg.max_misses if self._hold else 0
        expired = {self.live[ti].id for ti in unmatched_t if i - self.live[ti].last_seen_frame > grace}

        for di in unmatched_d:
            t = Track(id=self._next_id, box=detections[di].box, last_seen_frame=i)
            self._next_id += 1
            tracker = self._make_tracker()
            tracker.init(frame, gray, t.box)
            self._trackers[t.id] = tracker
            self.live.append(t)
            matched_ids.add(t.id)

        expired |= self._merge_duplicates(i)
        for t in [t for t in self.live if t.id in expired]:
            self._retire(t)
        return matched_ids - expired

    def _merge_duplicates(self, i: int) -> set[int]:
        """Two tracks on one face: keep the older ID, drop the younger, log it."""
        dropped: set[int] = set()
        by_id = sorted(self.live, key=lambda t: t.id)
        for a_idx, a in enumerate(by_id):
            if a.id in dropped:
                continue
            for b in by_id[a_idx + 1 :]:
                if b.id not in dropped and a.box.iou(b.box) > self._cfg.duplicate_iou:
                    dropped.add(b.id)
                    self.events.append(IdEvent(frame=i, kind="duplicate", kept=a.id, dropped=b.id))
        return dropped

    def _retire(self, t: Track) -> None:
        self.live.remove(t)
        self._trackers.pop(t.id, None)
        self.finished.append(t)

    def all_tracks(self) -> list[Track]:
        """Every track seen so far, live or finished, by ID."""
        return sorted(self.live + self.finished, key=lambda t: t.id)
