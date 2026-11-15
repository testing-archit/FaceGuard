"""Opt-in allow-list: leave chosen people visible, hide everyone else.

This is face *verification* against references the user supplies on this
computer ("is this the same person as in my reference photo?"), not
identification: there is no database of people, no names, and embeddings
live only in memory for the session.

Concept: SFace maps an aligned 112x112 face to a 128-d embedding trained so
that photos of the same person are close in angle (cosine similarity) and
different people are far apart. Alignment uses YuNet's five landmarks to
warp eyes, nose and mouth to canonical positions, which removes in-plane
rotation and scale before the comparison.

Privacy direction matters: a false *match* leaves a stranger visible, while
a false non-match only hides someone who wanted to be shown. So the
threshold is stricter than OpenCV's verification default (0.363), and the
UI shows each match score for the user to confirm (D-025).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.errors import FaceGuardError
from core.types import Config, Detection, Track

EMBED_SAMPLES = 3  # embeddings kept per track (largest confirmed boxes), averaged


def _row(d: Detection) -> np.ndarray:
    """YuNet's 15-value row (box, 5 landmarks, score), as SFace's alignCrop expects."""
    return np.array([*d.box.as_list(), *np.asarray(d.landmarks).ravel(), d.score], np.float32)


def available(cfg: Config) -> bool:
    return Path(cfg.recognizer_path).is_file()


class FaceMatcher:
    """SFace embeddings and cosine matching."""

    def __init__(self, cfg: Config) -> None:
        if not available(cfg):
            raise FaceGuardError(
                "The allow-list needs the SFace model. Run "
                "`python scripts/download_model.py --with-allowlist` once (38.7 MB)."
            )
        self._rec = cv2.FaceRecognizerSF.create(cfg.recognizer_path, "")
        self.threshold = cfg.allow_threshold

    def embed(self, frame: np.ndarray, det: Detection) -> np.ndarray | None:
        """L2-normalised embedding of one detected face, or None without landmarks (Haar)."""
        if det.landmarks is None:
            return None
        aligned = self._rec.alignCrop(frame, _row(det))
        f = self._rec.feature(aligned).ravel().astype(np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 0 else None

    def references(self, images: list[np.ndarray], detector) -> list[np.ndarray]:
        """One embedding per reference photo: its largest face."""
        refs = []
        for img in images:
            dets = [d for d in detector.detect(img) if d.landmarks is not None]
            if not dets:
                continue
            e = self.embed(img, max(dets, key=lambda d: d.box.area))
            if e is not None:
                refs.append(e)
        return refs


def similarity(e: np.ndarray, refs: list[np.ndarray]) -> float:
    """Best cosine similarity of an embedding to any reference (-1..1)."""
    if e is None or not refs:
        return -1.0
    return float(max(np.dot(e, r) for r in refs))


@dataclass
class AllowMatch:
    track_id: int
    score: float


def apply_allow_list(
    tracks: list[Track], embeddings: dict[int, np.ndarray], refs: list[np.ndarray], threshold: float
) -> list[AllowMatch]:
    """Leave matching tracks visible (redact=False). Returns every track's best score.

    Tracks without an embedding (never confirmed by a landmark detector,
    or drawn by hand) are never matched, so they stay hidden.
    """
    out = []
    for t in tracks:
        score = similarity(embeddings.get(t.id), refs)
        out.append(AllowMatch(t.id, score))
        if score >= threshold:
            t.redact = False
    return out


class EmbeddingCollector:
    """Per-frame pipeline hook: embeds each track's largest detector-confirmed faces.

    Runs only on frames where a detection was just matched to the track (so
    the box and landmarks are a real detection, not a tracker estimate) and
    keeps the `EMBED_SAMPLES` largest, averaged and re-normalised.
    """

    def __init__(self, matcher: FaceMatcher) -> None:
        self._m = matcher
        self._samples: dict[int, list[tuple[int, np.ndarray]]] = {}

    def __call__(self, i: int, frame: np.ndarray, present: list[Track]) -> None:
        for t in present:
            if t.last_seen_frame != i or t.landmarks is None or i in t.flags:
                continue
            got = self._samples.setdefault(t.id, [])
            if len(got) >= EMBED_SAMPLES and t.box.area <= min(a for a, _ in got):
                continue
            e = self._m.embed(frame, Detection(t.box, 1.0, t.landmarks))
            if e is None:
                continue
            got.append((t.box.area, e))
            got.sort(key=lambda x: -x[0])
            del got[EMBED_SAMPLES:]

    @property
    def embeddings(self) -> dict[int, np.ndarray]:
        out = {}
        for tid, got in self._samples.items():
            m = np.mean([e for _, e in got], axis=0)
            out[tid] = m / np.linalg.norm(m)
        return out
