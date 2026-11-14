"""Drive the Streamlit app headlessly through the M4 acceptance flow."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from core.review import new_job_dir, remove_job_dir  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"
PHOTO = Path("data/samples/er.png")  # 6 faces
needs_photo = pytest.mark.skipif(not PHOTO.exists(), reason="run scripts/fetch_samples.py")


def seeded_app(src: Path, kind: str):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    job = new_job_dir()
    dst = job / ("input" + src.suffix)
    shutil.copy(src, dst)
    s = at.session_state
    s.job, s.upload_id, s.src, s.kind, s.name = job, "seed", dst, kind, src.stem
    s.tracks, s.analysed_with, s.events, s.stats, s.export, s.editor_version = None, None, [], None, None, 0
    return at, job


def button(at, label):
    return next(b for b in at.button if b.label == label)


@needs_photo
def test_image_untick_delete_manual_export():
    at, job = seeded_app(PHOTO, "image")
    try:
        at.run()
        assert not at.exception
        button(at, "Detect faces").click().run()
        tracks = at.session_state.tracks
        assert len(tracks) >= 4

        # untick person 2, delete person 4 (stand-in for a false positive)
        tracks[1].redact = False
        at.session_state.tracks = [t for t in tracks if t.id != 4]
        at.run()

        # manual box over a "missed" face
        at.toggle(key="manual_on").set_value(True).run()
        at.number_input(key="mx").set_value(10)
        at.number_input(key="my").set_value(10)
        at.number_input(key="mw").set_value(40)
        at.number_input(key="mh").set_value(40)
        button(at, "Add box").click().run()
        assert not at.exception
        manual = [t for t in at.session_state.tracks if t.source == "manual"]
        assert len(manual) == 1 and manual[0].history[0].as_list() == [10, 10, 40, 40]

        button(at, "Render and export").click().run()
        assert not at.exception
        out, _ = at.session_state.export
        img, src = cv2.imread(str(out)), cv2.imread(str(PHOTO))
        assert img.shape == src.shape  # original resolution
        t2 = next(t for t in at.session_state.tracks if t.id == 2).box
        region = (slice(t2.y, t2.y + t2.h), slice(t2.x, t2.x + t2.w))
        assert np.array_equal(img[region], src[region])  # unticked face left visible
        assert not np.array_equal(img[10:50, 10:50], src[10:50, 10:50])  # manual box redacted
    finally:
        remove_job_dir(job)


def test_no_upload_shows_prompt():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Upload" in i.value for i in at.info)
    remove_job_dir(at.session_state.job)


CLIP = Path("data/samples/synthetic_er.mp4")


@pytest.mark.skipif(not CLIP.exists(), reason="run scripts/make_synthetic_clip.py")
def test_video_scrub_manual_track_export():
    at, job = seeded_app(CLIP, "video")
    try:
        at.run()
        button(at, "Detect faces").click().run()
        assert not at.exception
        assert len(at.session_state.tracks) == 6
        at.slider(key="scrub").set_value(100).run()
        assert not at.exception
        at.toggle(key="manual_on").set_value(True).run()
        at.number_input(key="mx").set_value(600)
        at.number_input(key="my").set_value(300)
        button(at, "Add box").click().run()  # default: follow with tracker
        manual = [t for t in at.session_state.tracks if t.source == "manual"]
        assert len(manual) == 1 and min(manual[0].history) == 100 and max(manual[0].history) > 100
        button(at, "Render and export").click().run()
        assert not at.exception
        out, mime = at.session_state.export
        assert mime == "video/mp4" and Path(out).stat().st_size > 0
    finally:
        remove_job_dir(job)
