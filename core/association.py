"""Data association: which detection belongs to which existing track.

Greedy IoU matching plus the track lifecycle (create, re-anchor, grace
period, delete, merge duplicates). Stable track IDs are what make selective
redaction work: the user toggles person 3 once and the choice follows them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.tracking import Tracker, TrackerFactory
from core.types import FLAG_LOST, FLAG_UNCONFIRMED, Box, Config, Detection, Track


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
    kind: str  # "duplicate": two tracks converged on one face, the younger was dropped
    #            "scene_cut": the scene changed, every live track was ended
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
        # Only real trackers produce evidence that a face is still there;
        # "none" just holds a stale box, so it gets the short lost-track grace.
        self._follows_motion = cfg.tracker in ("lk", "csrt", "kcf")
        self.live: list[Track] = []
        self.finished: list[Track] = []
        self.events: list[IdEvent] = []
        self._trackers: dict[int, Tracker] = {}
        self._next_id = 1
        self._last_detection: int | None = None
        self._velocity: dict[int, tuple[float, float]] = {}
        self.born: list[Track] = []  # tracks created on the latest step (for backfill)

    def step(
        self, i: int, frame: np.ndarray, gray: np.ndarray, detections: list[Detection] | None
    ) -> list[Track]:
        """Advance all tracks to frame `i`; return the tracks present on it."""
        self.born = []
        # 1. Propagate every track to this frame first (DECISIONS D-008), so
        #    detections are matched against where faces are now.
        for t in self.live:
            box = self._trackers[t.id].update(frame, gray)
            if box is None:
                if not t.lost:
                    t.lost, t.lost_since = True, i  # box frozen at its last position ...
                if self._cfg.coast and self._follows_motion:
                    t.box = self._coast(t)  # ... unless coasting carries it on
            elif not t.lost:
                self._update_velocity(t, box)
                t.box = box

        matched_ids: set[int] = set()
        if detections is not None:
            self._last_detection = i
            matched_ids = self._associate(i, frame, gray, detections)

        if self._hold:
            present = list(self.live)
        else:
            present = [t for t in self.live if t.id in matched_ids]
        for t in present:
            t.history[i] = self._mask_box(t, i, frame.shape)
            if t.lost:
                t.flags[i] = FLAG_LOST
            elif self._last_detection is not None and t.last_seen_frame < self._last_detection:
                t.flags[i] = FLAG_UNCONFIRMED
        return present

    @property
    def any_lost(self) -> bool:
        """True if a tracker has lost a face that is still being held (frozen box)."""
        return any(t.lost for t in self.live)

    @property
    def any_unconfirmed(self) -> bool:
        """True if the last detector run missed a face a tracker still reports.

        Tracker and detector disagree, so one of them is wrong: the detector
        missed a hard pose, or the tracker has been dragged off the face (an
        occluded face's points latch onto whatever passes in front, without
        the track ever being "lost"). Detecting more often until they agree
        again resolves it either way.
        """
        if self._last_detection is None:
            return False
        return any(t.last_seen_frame < self._last_detection for t in self.live)

    def scene_cut(self, i: int) -> None:
        """End every live track: a track ID never crosses a cut (motion is meaningless
        across it, and an ID is a motion track, not an identity)."""
        for t in list(self.live):
            self.events.append(IdEvent(frame=i, kind="scene_cut", kept=0, dropped=t.id))
            self._retire(t)

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
            t.landmarks = detections[di].landmarks
            t.last_seen_frame = i
            t.lost = False
            self._trackers[t.id].init(frame, gray, t.box)
            matched_ids.add(t.id)

        expired = {self.live[ti].id for ti in unmatched_t if self._expired(self.live[ti], i)}

        for di in unmatched_d:
            t = Track(id=self._next_id, box=detections[di].box, last_seen_frame=i,
                      landmarks=detections[di].landmarks)
            self._next_id += 1
            tracker = self._make_tracker()
            tracker.init(frame, gray, t.box)
            self._trackers[t.id] = tracker
            self.live.append(t)
            self.born.append(t)
            matched_ids.add(t.id)

        expired |= self._merge_duplicates(i)
        for t in [t for t in self.live if t.id in expired]:
            self._retire(t)
        return matched_ids - expired

    def _expired(self, t: Track, i: int) -> bool:
        """Lifecycle rule for a track the detector did not confirm on frame i.

        - tracker still following: keep up to `max_unconfirmed` frames since the
          last detector match; this caps drift (DECISIONS D-012)
        - tracker lost, or no tracker ("none"): keep the frozen box for
          `max_misses` frames
        - "off": no memory at all
        """
        cfg = self._cfg
        if not self._hold:
            return True
        if self._follows_motion and not t.lost:
            return i - t.last_seen_frame > cfg.max_unconfirmed
        since = t.lost_since if (self._follows_motion and t.lost) else t.last_seen_frame
        return i - since > cfg.max_misses or i - t.last_seen_frame > cfg.max_unconfirmed

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

    # --- coasting (D-023) ---------------------------------------------------

    def _update_velocity(self, t: Track, box: Box) -> None:
        """Exponential moving average of the box centre's per-frame displacement."""
        (cx0, cy0), (cx1, cy1) = t.box.center, box.center
        a = self._cfg.velocity_smoothing
        vx, vy = self._velocity.get(t.id, (cx1 - cx0, cy1 - cy0))
        self._velocity[t.id] = (a * (cx1 - cx0) + (1 - a) * vx, a * (cy1 - cy0) + (1 - a) * vy)

    def _coast(self, t: Track) -> Box:
        """Constant-velocity prediction with decay: where the face probably went.

        The box keeps its size so it can still be matched by IoU when the
        detector re-finds the face; only the recorded mask grows (`_mask_box`).
        """
        vx, vy = self._velocity.get(t.id, (0.0, 0.0))
        d = self._cfg.coast_decay
        self._velocity[t.id] = (vx * d, vy * d)
        return Box.from_xywh(t.box.x + vx, t.box.y + vy, t.box.w, t.box.h)

    def _mask_box(self, t: Track, i: int, shape: tuple[int, ...]) -> Box:
        """Box recorded for redaction on frame i: grown while a coasting track is lost.

        Uncertainty about a lost face's position increases every frame, so
        the mask grows with it, up to `coast_max_scale` times the box.
        """
        cfg = self._cfg
        if not (cfg.coast and t.lost and self._follows_motion):
            return t.box
        scale = min(1.0 + cfg.coast_grow * (i - t.lost_since + 1), cfg.coast_max_scale)
        cx, cy = t.box.center
        w, h = t.box.w * scale, t.box.h * scale
        return Box.from_xywh(cx - w / 2, cy - h / 2, w, h).clip(shape[1], shape[0])

    def _retire(self, t: Track) -> None:
        self.live.remove(t)
        self._trackers.pop(t.id, None)
        self.finished.append(t)

    def all_tracks(self) -> list[Track]:
        """Every track seen so far, live or finished, by ID."""
        return sorted(self.live + self.finished, key=lambda t: t.id)
