"""FaceGuard Streamlit UI: upload, analyse, review and correct, preview, export.

    streamlit run ui/app.py

No CV logic lives here. This file only collects settings, calls `core/`, and
converts BGR to RGB at the display boundary. All pixels stay on this machine:
`.streamlit/config.toml` binds the server to localhost and disables usage
statistics, and uploads live in a private temporary folder that Reset deletes.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.detectors import make_detector  # noqa: E402
from core.errors import IMAGE_EXTS, VIDEO_EXTS, FaceGuardError, media_kind  # noqa: E402
from core.pipeline import analyse_video, boxes_at, render_image, render_video  # noqa: E402
from core.review import (  # noqa: E402
    delete_track,
    detections_to_tracks,
    draw_overlay,
    interpolate_boxes,
    manual_track,
    new_job_dir,
    next_track_id,
    propagate,
    read_frame,
    remove_job_dir,
    remove_stale_jobs,
    tracks_on_frame,
)
from core.types import DETECTORS, MODES, Box, Config, Track  # noqa: E402
from core.video_io import VideoReader, limit_warnings  # noqa: E402

# Settings that change what analysis finds; changing them needs a re-run.
ANALYSIS_KEYS = ("detector", "conf_threshold", "tracker", "detect_interval")
UI_TRACKERS = ("lk", "csrt", "kcf", "none", "off")
TRACKER_LABELS = {
    "lk": "Lucas-Kanade optical flow (recommended)",
    "csrt": "CSRT (accurate, slow)",
    "kcf": "KCF (fast, loses faces)",
    "none": "Hold box still between detections",
    "off": "Detect every frame, no tracking (baseline)",
}
MODE_LABELS = {"fill": "Solid fill (strongest)", "pixelate": "Pixelate", "blur": "Gaussian blur (weakest)"}

st.set_page_config(page_title="FaceGuard", layout="wide")


# --- state -----------------------------------------------------------------


def state() -> st.session_state:
    s = st.session_state
    if "job" not in s:
        remove_stale_jobs()  # sessions that crashed without Reset
        s.job = new_job_dir()
        s.upload_id = None
        s.src = None
        s.kind = None
        s.tracks = None
        s.analysed_with = None
        s.events = []
        s.stats = None
        s.export = None
        s.editor_version = 0
    return s


def reset() -> None:
    remove_job_dir(st.session_state.get("job"))
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def rgb(img: np.ndarray) -> np.ndarray:
    """The one BGR -> RGB conversion, at the Streamlit boundary."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analysis_key(cfg: Config) -> tuple:
    return tuple(getattr(cfg, k) for k in ANALYSIS_KEYS)


def tracks_changed(s: st.session_state) -> None:
    """Anything that edits tracks invalidates the export and the editor widget."""
    s.export = None
    s.editor_version += 1


# --- sidebar ---------------------------------------------------------------


def sidebar() -> Config:
    d = Config()
    sb = st.sidebar
    sb.title("FaceGuard")
    sb.caption("Offline face redaction. Nothing leaves this computer.")

    sb.subheader("Redaction")
    mode = sb.radio("Mode", ("fill", "pixelate", "blur"), index=("fill", "pixelate", "blur").index(d.mode),
                    format_func=MODE_LABELS.get)
    sb.info(
        "**Privacy:** solid fill destroys the face. Pixelation can be attacked when the "
        "candidate set of people is small. Blur is the weakest: deblurring models can "
        "partially recover faces. Use fill when it matters."
    )
    padding = sb.slider("Padding per side", 0.0, 0.6, d.padding, 0.01,
                        help="Grow each box by this fraction of its size on every side. "
                        "Below 0.21 the elliptical mask cannot cover the box corners.")
    blur_strength = d.blur_strength
    pixel_blocks = d.pixel_blocks
    if mode == "blur":
        blur_strength = sb.slider("Blur strength", 0.1, 1.0, d.blur_strength, 0.05,
                                  help="Gaussian kernel size as a fraction of face width.")
    elif mode == "pixelate":
        pixel_blocks = sb.slider("Pixel blocks across face", 3, 20, d.pixel_blocks,
                                 help="Fewer blocks = coarser and stronger.")

    sb.subheader("Detection")
    detector = sb.selectbox("Detector", DETECTORS, format_func=lambda k: {"yunet": "YuNet CNN", "haar": "Haar cascade (baseline)"}[k])
    conf = sb.slider("Confidence threshold", 0.1, 0.95, d.conf_threshold, 0.05, disabled=detector != "yunet",
                     help="Lower finds more faces and more false positives. Missing a face is worse.")

    sb.subheader("Video tracking")
    tracker = sb.selectbox("Tracker", UI_TRACKERS, format_func=TRACKER_LABELS.get)
    interval = sb.slider("Detect every N frames", 1, 30, d.detect_interval, disabled=tracker == "off")
    keep_audio = sb.checkbox("Keep original audio", value=d.keep_audio,
                             help="Off by default. Audio is copied unchanged and voices can identify people.")
    if keep_audio:
        sb.warning("Audio is exported unchanged. Voices may reveal identity.")

    sb.divider()
    if sb.button("Reset session", help="Deletes the upload, previews and exports from this computer."):
        reset()
        st.rerun()
    sb.caption("Reset deletes temporary files (ordinary deletion, not secure erasure).")

    return replace(
        d, mode=mode, padding=padding, blur_strength=blur_strength, pixel_blocks=pixel_blocks,
        detector=detector, conf_threshold=conf, tracker=tracker, detect_interval=interval,
        keep_audio=keep_audio,
    )


# --- upload and analysis ---------------------------------------------------


def handle_upload(s: st.session_state) -> None:
    exts = [e.lstrip(".") for e in IMAGE_EXTS + VIDEO_EXTS]
    up = st.file_uploader("Image or short video", type=exts,
                          help="JPG/PNG images, MP4/AVI/MOV video. Up to about 2 minutes of video.")
    if up is None:
        return
    if up.file_id == s.upload_id:
        return
    try:
        kind = media_kind(up.name)
    except FaceGuardError as exc:
        st.error(str(exc))
        return
    remove_job_dir(s.job)
    s.job = new_job_dir()
    src = s.job / ("input" + Path(up.name).suffix.lower())
    src.write_bytes(up.getbuffer())
    s.upload_id, s.src, s.kind = up.file_id, src, kind
    s.name = Path(up.name).stem
    s.tracks, s.analysed_with, s.events, s.stats, s.export = None, None, [], None, None
    s.editor_version += 1


def run_analysis(s: st.session_state, cfg: Config) -> None:
    try:
        if s.kind == "image":
            img = load_image(s.src)
            s.tracks = detections_to_tracks(make_detector(cfg).detect(img))
            s.events, s.stats = [], None
        else:
            bar = st.progress(0.0, text="Analysing video...")

            def progress(done: int, total: int) -> None:
                if total and (done % 5 == 0 or done == total):
                    bar.progress(min(done / total, 1.0), text=f"Analysing: frame {done} / {total}")

            a = analyse_video(s.src, cfg, progress)
            bar.empty()
            s.tracks, s.events, s.stats = a.tracks, a.events, a.stats
            for w in a.stats.warnings:
                st.warning(w)
        s.analysed_with = analysis_key(cfg)
        tracks_changed(s)
        st.rerun()  # redraw without the button that started the run
    except (FaceGuardError, FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))


@st.cache_data(show_spinner=False, max_entries=4)
def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # applies EXIF orientation
    if img is None:
        raise FaceGuardError("Could not decode this image. Is the file corrupted?")
    return img


@st.cache_data(show_spinner=False, max_entries=64)
def load_frame(path: Path, index: int) -> np.ndarray:
    return read_frame(path, index)


def video_info(path: Path):
    with VideoReader(path) as r:
        return r.info


# --- review ----------------------------------------------------------------


def track_table(s: st.session_state, frame: int, n_frames: int) -> None:
    tracks: list[Track] = s.tracks
    st.subheader(f"Faces ({len(tracks)})")
    if not tracks:
        st.info("No faces detected. Add a manual box below if a face was missed.")
        return
    only_here = False
    if s.kind == "video":
        only_here = st.toggle("Only faces on the current frame", value=len(tracks) > 12)
    shown = tracks_on_frame(tracks, frame) if only_here else tracks
    rows = []
    for t in shown:
        frames = sorted(t.history)
        row = {"Person": t.id, "Redact": t.redact, "Delete": False,
               "Source": t.source}
        if s.kind == "video":
            row |= {"From": frames[0], "To": frames[-1], "Frames": len(frames)}
        else:
            row |= {"Score": None if t.score is None else round(t.score, 2)}
        rows.append(row)
    edited = st.data_editor(
        rows, key=f"tracks-{s.editor_version}", hide_index=True, width="stretch",
        disabled=[c for c in (rows[0].keys() if rows else []) if c not in ("Redact", "Delete")],
        column_config={
            "Redact": st.column_config.CheckboxColumn(help="Untick to leave this person visible"),
            "Delete": st.column_config.CheckboxColumn(help="Tick to remove a false positive"),
        },
    )
    changed = False
    by_id = {t.id: t for t in tracks}
    for row in edited:
        t = by_id.get(row["Person"])
        if t is not None and t.redact != row["Redact"]:
            t.redact = row["Redact"]
            changed = True
    to_delete = [row["Person"] for row in edited if row["Delete"]]
    if to_delete:
        for tid in to_delete:
            s.tracks = delete_track(s.tracks, tid)
        changed = True
    if changed:
        tracks_changed(s)
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("Redact all"):
        for t in tracks:
            t.redact = True
        tracks_changed(s)
        st.rerun()
    if c2.button("Redact none"):
        for t in tracks:
            t.redact = False
        tracks_changed(s)
        st.rerun()
    if s.events:
        with st.expander(f"Suspected ID switches ({len(s.events)})"):
            st.caption("Two tracks converged on one face; the younger one was merged into the older.")
            st.table([{"frame": e.frame, "kept": e.kept, "dropped": e.dropped} for e in s.events])


def manual_box_form(s: st.session_state, cfg: Config, frame_img: np.ndarray, frame: int, n_frames: int) -> Box | None:
    """Coordinate entry for a missed face. Returns the draft box for the preview."""
    h, w = frame_img.shape[:2]
    if not st.toggle("Add a manual box over a missed face", key="manual_on"):
        return None
    with st.container(border=True):
        st.caption("Enter the box in pixels; it is drawn in magenta on the preview so you can line it up.")
        c = st.columns(4)
        bw = c[2].number_input("Width", 1, w, min(w, max(24, w // 8)), key="mw")
        bh = c[3].number_input("Height", 1, h, min(h, max(24, h // 6)), key="mh")
        x = c[0].number_input("Left (x)", 0, w - 1, max(0, (w - bw) // 2), key="mx")
        y = c[1].number_input("Top (y)", 0, h - 1, max(0, (h - bh) // 2), key="my")
        draft = Box(int(x), int(y), int(bw), int(bh)).clip(w, h)

        how, end, end_box = "single", None, None
        if s.kind == "video":
            how = st.radio(
                "Over which frames?",
                ("track", "hold", "interpolate"),
                format_func={
                    "track": "Follow it with the tracker from this frame",
                    "hold": "Hold still from this frame to an end frame",
                    "interpolate": "Move linearly to a box at an end frame",
                }.get,
                horizontal=True,
            )
            if how in ("hold", "interpolate"):
                end = st.number_input("End frame", frame, n_frames - 1, min(n_frames - 1, frame + 30))
            if how == "interpolate":
                e = st.columns(4)
                ex = e[0].number_input("End x", 0, w - 1, int(x))
                ey = e[1].number_input("End y", 0, h - 1, int(y))
                ew = e[2].number_input("End width", 1, w, int(bw))
                eh = e[3].number_input("End height", 1, h, int(bh))
                end_box = Box(int(ex), int(ey), int(ew), int(eh)).clip(w, h)
                st.caption("Motion is rarely linear: scrub through the interval before exporting.")

        if st.button("Add box", type="primary"):
            try:
                if s.kind == "image":
                    history = {0: draft}
                elif how == "track":
                    with st.spinner("Tracking the box forward..."):
                        tracker_cfg = cfg if cfg.tracker in ("lk", "csrt", "kcf") else replace(cfg, tracker="lk")
                        history = propagate(s.src, frame, draft, tracker_cfg)
                elif how == "hold":
                    history = {f: draft for f in range(frame, int(end) + 1)}
                else:
                    history = interpolate_boxes(frame, draft, int(end), end_box)
                tid = next_track_id(s.tracks)
                s.tracks = s.tracks + [manual_track(tid, history)]
                tracks_changed(s)
                span = f" on frames {min(history)}-{max(history)}" if s.kind == "video" else ""
                st.session_state.flash = f"Added manual box {tid}{span}."
                st.rerun()
            except FaceGuardError as exc:
                st.error(str(exc))
        return draft


def preview(s: st.session_state, cfg: Config) -> None:
    if s.kind == "image":
        base = load_image(s.src)
        frame, n_frames = 0, 1
    else:
        info = video_info(s.src)
        n_frames = max(info.frame_count, 1)
        frame = st.slider("Frame", 0, n_frames - 1, 0, key="scrub",
                          help=f"{info.width}x{info.height} at {info.fps:.1f} fps, {info.duration:.1f} s")
        try:
            base = load_frame(s.src, frame)
        except FaceGuardError as exc:
            st.error(str(exc))
            return

    left, right = st.columns([3, 2])
    with right:
        track_table(s, frame, n_frames)
        draft = manual_box_form(s, cfg, base, frame, n_frames)
    with left:
        a, b = st.columns(2)
        a.image(rgb(draw_overlay(base, s.tracks, frame, draft=draft, cfg=cfg)),
                caption="Detections (green = redact, orange = visible, blue = manual)",
                width="stretch")
        b.image(rgb(render_image(base, boxes_at(s.tracks, frame), cfg)),
                caption=f"Redacted preview ({cfg.mode})", width="stretch")
        if s.kind == "video":
            here = tracks_on_frame(s.tracks, frame)
            st.caption(f"{len(here)} region(s) on frame {frame}, "
                       f"{sum(t.redact for t in here)} redacted.")


# --- export ----------------------------------------------------------------


def export(s: st.session_state, cfg: Config) -> None:
    st.subheader("Export")
    ext = s.src.suffix if s.kind == "image" else ".mp4"
    out = s.job / f"{s.name}_redacted{ext}"
    if st.button("Render and export", type="primary"):
        try:
            if s.kind == "image":
                img = render_image(load_image(s.src), boxes_at(s.tracks, 0), cfg)
                ok, buf = cv2.imencode(ext, img)  # fresh encode: no EXIF copied
                if not ok:
                    raise FaceGuardError("Could not encode the image.")
                out.write_bytes(buf.tobytes())
                s.export = (out, "image/png" if ext == ".png" else "image/jpeg")
            else:
                bar = st.progress(0.0, text="Rendering...")

                def progress(done: int, total: int) -> None:
                    if total and (done % 5 == 0 or done == total):
                        bar.progress(min(done / total, 1.0), text=f"Rendering: frame {done} / {total}")

                stats = render_video(s.src, out, s.tracks, cfg, progress)
                bar.empty()
                for w in stats.warnings:
                    st.warning(w)
                s.export = (out, "video/mp4")
        except FaceGuardError as exc:
            st.error(str(exc))

    if s.export and Path(s.export[0]).exists():
        path, mime = s.export
        if s.kind == "video":
            st.video(str(path))
        st.download_button("Download redacted file", Path(path).read_bytes(), file_name=Path(path).name, mime=mime)
        st.caption("Only the redacted copy is offered. The original is never modified.")


# --- main ------------------------------------------------------------------


def main() -> None:
    s = state()
    cfg = sidebar()
    st.title("FaceGuard")
    st.caption("Detect faces, choose who to hide, fix misses, preview, export. Runs fully offline.")
    handle_upload(s)
    if s.src is None:
        st.info("Upload a photo or a short video to start.")
        return
    if s.kind == "video":
        info = video_info(s.src)
        for w in limit_warnings(info):
            st.warning(w)

    if s.tracks is None:
        if st.button("Detect faces", type="primary"):
            run_analysis(s, cfg)
    elif s.analysed_with != analysis_key(cfg):
        st.warning("Detection settings changed. Re-run detection to apply them "
                   "(this discards manual edits).")
        if st.button("Re-run detection"):
            run_analysis(s, cfg)
    if s.tracks is None:
        return
    if s.stats is not None:
        st.caption(f"Analysed {s.stats.frames} frames at {s.stats.fps:.0f} FPS, "
                   f"{s.stats.detector_calls} detector calls, {len(s.tracks)} track(s).")
    if msg := st.session_state.pop("flash", None):
        st.success(msg)
    preview(s, cfg)
    export(s, cfg)


main()
