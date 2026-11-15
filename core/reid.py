"""Re-identification linking: give a face that reappears its old track ID.

Track IDs come from motion. When a face is fully occluded, or leaves and
re-enters, the motion track ends and a new ID starts, so a choice like
"leave person 2 visible" silently stops applying to them (REPORT section 8).

Linking compares appearance instead of motion: each track's SFace embedding
(core/recognition.py, the mean of its largest detector-confirmed faces) is
matched against tracks that ended *before* it started. If the cosine
similarity is at least `reid_threshold` and the gap is at most
`reid_max_gap` frames, the later track is merged into the earlier one.

This is verification between two tracks of the same video, nothing is
stored beyond the session and no identity is named (CLAUDE.md, D-025).
Opt-in, and only meaningful with the SFace model present (D-031).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import Config, Track


@dataclass(frozen=True)
class Link:
    kept: int  # earlier track ID, which survives
    merged: int  # later track ID, folded into `kept`
    similarity: float
    gap: int  # frames between the end of `kept` and the start of `merged`
    frame: int  # first frame of `merged`, where the face reappeared


def link_tracks(tracks: list[Track], embeddings: dict[int, np.ndarray], cfg: Config) -> tuple[list[Track], list[Link]]:
    """Merge later tracks into earlier ones they match. Returns (tracks, links).

    Candidates are processed in order of the later track's start, best
    similarity first, so a chain A -> B -> C becomes one track A. A track
    can absorb at most one successor at a time (its new end then moves on),
    which stops two simultaneous people from both being merged into it.
    Manual tracks are never linked.
    """
    by_id = {t.id: t for t in tracks}
    # A track "ends" at its last box the detector backed up. Frames after that
    # are tracker guesses (flagged): an occluded face's track is often dragged
    # along by the occluder for a while before it expires, overlapping the
    # new track of the same face when it re-emerges.
    span = {t.id: (min(t.history), _confirmed_end(t)) for t in tracks if t.history}
    alive = {t.id for t in tracks}
    links: list[Link] = []
    for later in sorted((t for t in tracks if t.id in span), key=lambda t: span[t.id][0]):
        if later.source != "detector" or later.id not in embeddings or later.id not in alive:
            continue
        start = span[later.id][0]
        best: tuple[float, Track] | None = None
        for earlier in tracks:
            if earlier.id == later.id or earlier.id not in alive or earlier.source != "detector":
                continue
            if earlier.id not in embeddings or earlier.id not in span:
                continue
            end = span[earlier.id][1]
            if not 0 < start - end <= cfg.reid_max_gap:
                continue
            sim = float(np.dot(embeddings[earlier.id], embeddings[later.id]))
            if sim >= cfg.reid_threshold and (best is None or sim > best[0]):
                best = (sim, earlier)
        if best is None:
            continue
        sim, earlier = best
        links.append(Link(earlier.id, later.id, round(sim, 4), start - span[earlier.id][1], start))
        # Guessed boxes of `earlier` after its confirmed end give way to the
        # later track's boxes on frames both have.
        for f in [f for f in earlier.history if f > span[earlier.id][1] and f in later.history]:
            del earlier.history[f]
            earlier.flags.pop(f, None)
        earlier.history.update(later.history)
        earlier.flags.update(later.flags)
        earlier.box, earlier.last_seen_frame = later.box, later.last_seen_frame
        earlier.redact = earlier.redact or later.redact  # never expose by merging
        span[earlier.id] = (span[earlier.id][0], _confirmed_end(earlier))
        # The merged appearance is the mean of both, re-normalised.
        e = embeddings[earlier.id] + embeddings[later.id]
        embeddings[earlier.id] = e / np.linalg.norm(e)
        alive.discard(later.id)
    return [by_id[i] for i in sorted(alive)], links


def _confirmed_end(t: Track) -> int:
    confirmed = [f for f in t.history if f not in t.flags]
    return max(confirmed) if confirmed else min(t.history)
