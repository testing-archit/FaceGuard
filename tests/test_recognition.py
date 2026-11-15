"""Allow-list: verification against user-supplied reference faces (needs the SFace model)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.detectors import YuNetDetector
from core.pipeline import analyse_video
from core.recognition import EmbeddingCollector, FaceMatcher, apply_allow_list, available, similarity
from core.types import Box, Config, Track

PHOTO = Path("data/samples/addams-family.png")
CLIP = Path("data/samples/synthetic_cross.mp4")
SRC = Path("data/samples/er.png")
cfg = Config()
needs_model = pytest.mark.skipif(not available(cfg), reason="python scripts/download_model.py --with-allowlist")
needs_photo = pytest.mark.skipif(not PHOTO.exists(), reason="run scripts/fetch_samples.py")


@needs_model
@needs_photo
def test_same_person_matches_and_others_do_not():
    m, det = FaceMatcher(cfg), YuNetDetector(cfg)
    img = cv2.imread(str(PHOTO))
    faces = det.detect(img)
    embs = [m.embed(img, d) for d in faces]
    other = cv2.resize(cv2.flip(img, 1), None, fx=0.8, fy=0.8)  # same people, new pose and scale
    for d in det.detect(other):
        e = m.embed(other, d)
        scores = sorted((float(np.dot(e, x)) for x in embs), reverse=True)
        assert scores[0] >= cfg.allow_threshold > scores[1]


@needs_model
@needs_photo
def test_reference_photo_leaves_only_that_person_visible(tmp_path):
    from cli import main

    img = cv2.imread(str(PHOTO))
    face = YuNetDetector(cfg).detect(img)[0].box.pad(1.0, img.shape[1], img.shape[0])
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), img[face.y : face.y + face.h, face.x : face.x + face.w])
    out = tmp_path / "out.png"
    assert main(["--input", str(PHOTO), "--output", str(out), "--mode", "fill", "--keep-visible", str(ref)]) == 0
    res = cv2.imread(str(out))
    dets = YuNetDetector(cfg).detect(img)
    b = dets[0].box
    assert np.array_equal(res[b.y : b.y + b.h, b.x : b.x + b.w], img[b.y : b.y + b.h, b.x : b.x + b.w])
    b = dets[1].box
    assert not np.array_equal(res[b.y : b.y + b.h, b.x : b.x + b.w], img[b.y : b.y + b.h, b.x : b.x + b.w])


@needs_model
@pytest.mark.skipif(not (CLIP.exists() and SRC.exists()), reason="run scripts/make_crossing_clip.py")
def test_video_tracks_get_embeddings_and_match_reference():
    m = FaceMatcher(cfg)
    col = EmbeddingCollector(m)
    a = analyse_video(CLIP, cfg, on_frame=col)
    assert set(col.embeddings) == {t.id for t in a.tracks}
    # reference: person A's face from the source photo (FACE_A in make_crossing_clip.py)
    src = cv2.imread(str(SRC))
    refs = m.references([src[200:340, 90:230]], YuNetDetector(cfg))
    matches = apply_allow_list(a.tracks, col.embeddings, refs, cfg.allow_threshold)
    visible = {t.id for t in a.tracks if not t.redact}
    assert len(visible) >= 1 and len(visible) < len(a.tracks)
    assert max(x.score for x in matches) >= cfg.allow_threshold


def test_tracks_without_embeddings_stay_hidden():
    t = Track(id=1, box=Box(0, 0, 10, 10))
    apply_allow_list([t], {}, [np.ones(128, np.float32) / np.sqrt(128)], 0.5)
    assert t.redact and similarity(None, []) == -1.0
