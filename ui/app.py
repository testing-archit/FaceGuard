"""FaceGuard Streamlit UI: upload, detect, review and correct, export.

    streamlit run ui/app.py

No CV logic lives here. This file collects settings, calls `core/`, and
converts BGR to RGB at the display boundary. All pixels stay on this machine:
`.streamlit/config.toml` binds the server to localhost and disables usage
statistics, and uploads live in a private temporary folder that Reset deletes.
Visual design lives in `ui/style.css` and the theme in `.streamlit/config.toml`.
"""

from __future__ import annotations

import html
import io
import json
import sys
import zipfile
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
    ThumbnailCollector,
    crop_thumbnail,
    delete_track,
    detections_to_tracks,
    flagged_frames,
    interpolate_boxes,
    manual_track,
    new_job_dir,
    next_track_id,
    propagate,
    read_frame,
    redaction_log,
    remove_job_dir,
    remove_stale_jobs,
    review_stops,
    step_to,
    tracks_on_frame,
)
from core.types import DETECTORS, Box, Config, Track  # noqa: E402
from core.video_io import VideoReader, limit_warnings  # noqa: E402
from ui.components import face_canvas, timeline  # noqa: E402

# Settings that change what analysis finds; changing them needs a re-run.
ANALYSIS_KEYS = ("detector", "conf_threshold", "tile_grid", "tracker", "detect_interval", "smooth_window")
SMALL_FACE_TILES = 2  # tile grid used by the "Search for small faces" switch
TRACKERS = ("lk", "csrt", "kcf", "none", "off")
TRACKER_LABELS = {
    "lk": "Lucas-Kanade optical flow",
    "csrt": "CSRT correlation filter",
    "kcf": "KCF correlation filter",
    "none": "Hold box between detections",
    "off": "None (detect every frame)",
}
DETECTOR_LABELS = {"yunet": "YuNet (neural network)", "haar": "Haar cascade (classical)"}
MODES = ("fill", "pixelate", "blur")
MODE_LABELS = {"fill": "Solid", "pixelate": "Pixelate", "blur": "Blur"}
MODE_HELP = {
    "fill": "Strongest. The face is replaced; nothing can be recovered.",
    "pixelate": "Strong, but can be matched against a small set of known faces.",
    "blur": "Weakest. Deblurring models can partially recover blurred faces.",
}
SHAPE_LABELS = {"ellipse": "Ellipse", "rect": "Rectangle"}
PRESETS = {
    "Maximum privacy": dict(mode="fill", mask_shape="rect", padding=0.30, detect_interval=5, conf_threshold=0.5),
    "Balanced": dict(mode="pixelate", mask_shape="ellipse", padding=0.25, detect_interval=10, conf_threshold=0.6),
    "Fast": dict(mode="pixelate", mask_shape="ellipse", padding=0.25, detect_interval=15, conf_threshold=0.6),
}
PRESET_KEYS = {"mode": "set_mode", "mask_shape": "set_shape", "padding": "set_padding",
               "detect_interval": "set_interval", "conf_threshold": "set_conf"}
PEOPLE_PANEL_PX = 470  # max scrollable height of the people list
PERSON_ROW_PX = 74  # approximate height of one person row, to size the list
THUMB_SHOW_PX = 44
THUMB_PX, THUMB_PAD = 96, 0.3  # stored crop size and context, as in core.review
FRAME_JUMP = 10  # frames moved by the jump buttons
BATCH_THUMB_PX = 360
BATCH_COLS = 4

SHIELD_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3Z" fill="currentColor" opacity=".14"/>'
    '<path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3Z" stroke="currentColor" stroke-width="1.6"/>'
    '<rect x="8.2" y="9.4" width="7.6" height="3.6" rx="1" fill="currentColor"/></svg>'
)

st.set_page_config(page_title="FaceGuard", page_icon=":material/shield_person:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)


# --- state -----------------------------------------------------------------


def state():
    s = st.session_state
    if "job" not in s:
        remove_stale_jobs()  # sessions that crashed without Reset
        s.job = new_job_dir()
        s.upload_id = None
        s.src = None
        s.kind = None
        s.name = None
        s.tracks = None
        s.analysed_with = None
        s.events = []
        s.stats = None
        s.export = None
        s.thumbs = {}
        s.editor_version = 0
        s.draft = None
        s.batch = None
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


def tracks_changed(s) -> None:
    """Anything that edits tracks invalidates the export and per-track widgets."""
    s.export = None
    s.editor_version += 1


def by_id(s, tid: int) -> Track | None:
    return next((t for t in s.tracks if t.id == tid), None)


def section(title: str, note: str = "") -> None:
    """Small uppercase section label, used in the sidebar and panels."""
    extra = f'<span class="fg-section-note">{html.escape(note)}</span>' if note else ""
    st.markdown(f'<div class="fg-section">{html.escape(title)}{extra}</div>', unsafe_allow_html=True)


# --- sidebar ---------------------------------------------------------------


def apply_preset() -> None:
    name = st.session_state.get("preset")
    for field, value in PRESETS.get(name, {}).items():
        st.session_state[PRESET_KEYS[field]] = value


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def sidebar() -> Config:
    d = Config()
    ss = st.session_state
    for field, key in PRESET_KEYS.items():
        ss.setdefault(key, getattr(d, field))
    with st.sidebar:
        st.markdown(f'<div class="fg-brand">{SHIELD_SVG}<span>FaceGuard</span></div>', unsafe_allow_html=True)

        section("Preset")
        st.selectbox("Preset", ["Custom", *PRESETS], key="preset", on_change=apply_preset,
                     label_visibility="collapsed",
                     help="Sets several options at once. Each can still be changed below.")

        section("Redaction")
        mode = st.segmented_control("Effect", MODES, key="set_mode", format_func=MODE_LABELS.get,
                                    width="stretch") or d.mode
        st.caption(MODE_HELP[mode])
        shape = st.segmented_control("Mask shape", ("ellipse", "rect"), key="set_shape",
                                     format_func=SHAPE_LABELS.get, width="stretch",
                                     help="Rectangle hides the whole padded box.") or d.mask_shape
        padding = st.slider("Padding", 0.0, 0.6, step=0.01, key="set_padding",
                            help="Grows each box by this fraction of its size on every side. With the "
                            "ellipse, values below 0.21 leave the corners of the face box uncovered.")
        blur_strength, pixel_blocks, fill = d.blur_strength, d.pixel_blocks, d.fill_color
        if mode == "blur":
            blur_strength = st.slider("Blur strength", 0.1, 1.0, d.blur_strength, 0.05,
                                      help="Gaussian kernel size as a fraction of face width.")
        elif mode == "pixelate":
            pixel_blocks = st.slider("Blocks across face", 3, 20, d.pixel_blocks,
                                     help="Fewer blocks give a coarser, stronger effect.")
        else:
            fill = hex_to_bgr(st.color_picker("Fill colour", "#000000"))

        section("Detection")
        detector = st.selectbox("Detector", DETECTORS, format_func=DETECTOR_LABELS.get)
        conf = st.slider("Confidence threshold", 0.1, 0.95, step=0.05, key="set_conf",
                         disabled=detector != "yunet",
                         help="Lower finds more faces and more false positives. A missed face is worse.")
        small = st.toggle("Search for small faces", value=False,
                          help="Also scans enlarged sections of the image, for crowds and distant people. "
                          "About 5x slower; best for photos.")

        section("Video")
        tracker = st.selectbox("Tracking", TRACKERS, format_func=TRACKER_LABELS.get)
        interval = st.slider("Detect every N frames", 1, 30, key="set_interval", disabled=tracker == "off",
                             help="Lower finds faces entering the shot sooner, at some speed cost.")
        smooth = st.slider("Box smoothing (frames)", 1, 9, d.smooth_window, step=2,
                           help="Steadier masks, but boxes can lag fast motion. 1 turns it off. "
                           "In testing, 3 added no exposed frames; 5 and above did.")
        keep_audio = st.toggle("Keep original audio", value=d.keep_audio,
                               help="Audio is copied unchanged. Voices can identify people.")

        st.divider()
        st.button("Reset session", icon=":material/restart_alt:", width="stretch", on_click=reset,
                  help="Deletes the upload, previews and exports from this computer.")
        st.markdown('<div class="fg-foot">Runs locally. No uploads, no account, no network.</div>',
                    unsafe_allow_html=True)

    return replace(
        d, mode=mode, mask_shape=shape, padding=padding, blur_strength=blur_strength,
        pixel_blocks=pixel_blocks, fill_color=fill, detector=detector, conf_threshold=conf,
        tile_grid=SMALL_FACE_TILES if small else 0,
        tracker=tracker, detect_interval=interval, smooth_window=smooth, keep_audio=keep_audio,
    )


# --- header ----------------------------------------------------------------


def header(s) -> None:
    if s.src is None:
        now = 0
    elif s.tracks is None:
        now = 1
    elif s.export is None:
        now = 2
    else:
        now = 3
    steps = ["Upload", "Detect", "Review", "Export"]
    items = "".join(
        f'<li class="{"now" if k == now else "done" if k < now else ""}"><span>{k + 1}</span>{name}</li>'
        for k, name in enumerate(steps)
    )
    st.markdown(
        f"""
<div class="fg-header">
  <div class="fg-title"><h1>Face redaction</h1>
    <p>Find faces in a photo or video, choose who to hide, and export a redacted copy.</p></div>
  <ol class="fg-steps">{items}</ol>
</div>
""",
        unsafe_allow_html=True,
    )


# --- upload and analysis ---------------------------------------------------


def handle_upload(s) -> None:
    exts = [e.lstrip(".") for e in IMAGE_EXTS + VIDEO_EXTS]
    up = st.file_uploader("Photo or video", type=exts, label_visibility="collapsed",
                          help="JPG or PNG images; MP4, AVI or MOV video up to about 2 minutes.")
    if up is None or up.file_id == s.upload_id:
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
    s.upload_id, s.src, s.kind, s.name = up.file_id, src, kind, Path(up.name).stem
    s.tracks, s.analysed_with, s.events, s.stats, s.export, s.thumbs, s.draft = None, None, [], None, None, {}, None
    s.pop("scrub", None)
    s.editor_version += 1


def run_analysis(s, cfg: Config) -> None:
    try:
        if s.kind == "image":
            img = load_image(s.src)
            s.tracks = detections_to_tracks(make_detector(cfg).detect(img))
            s.thumbs = {t.id: crop_thumbnail(img, t.box, THUMB_PX, THUMB_PAD) for t in s.tracks}
            s.events, s.stats = [], None
        else:
            bar = st.progress(0.0, text="Detecting and tracking faces")

            def progress(done: int, total: int) -> None:
                if total and (done % 5 == 0 or done == total):
                    bar.progress(min(done / total, 1.0), text=f"Detecting and tracking faces · {done} / {total}")

            thumbs = ThumbnailCollector()
            a = analyse_video(s.src, cfg, progress, on_frame=thumbs)
            bar.empty()
            s.tracks, s.events, s.stats, s.thumbs = a.tracks, a.events, a.stats, thumbs.thumbs
            s.analysis_warnings = a.stats.warnings
        s.analysed_with = analysis_key(cfg)
        s.draft = None
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


@st.cache_data(show_spinner=False)
def video_info(path: Path):
    with VideoReader(path) as r:
        return r.info


# --- file bar and stats ----------------------------------------------------


def file_bar(s, cfg: Config):
    """Name and properties of the loaded file, with the primary action on the right."""
    info = None
    if s.kind == "video":
        info = video_info(s.src)
        props = [f"{info.width} × {info.height}", f"{info.fps:.0f} fps", f"{info.duration:.1f} s",
                 f"{info.frame_count} frames"]
    else:
        img = load_image(s.src)
        props = [f"{img.shape[1]} × {img.shape[0]}", "Image"]
    left, right = st.columns([5, 2], vertical_alignment="center")
    left.markdown(
        f'<div class="fg-file"><b>{html.escape(s.name)}</b>'
        + "".join(f"<span>{html.escape(p)}</span>" for p in props) + "</div>",
        unsafe_allow_html=True,
    )
    with right:
        if s.tracks is None:
            if st.button("Detect faces", type="primary", icon=":material/center_focus_strong:", width="stretch"):
                run_analysis(s, cfg)
        elif s.analysed_with != analysis_key(cfg):
            if st.button("Re-run detection", type="primary", icon=":material/refresh:", width="stretch",
                         help="Detection settings changed. Re-running discards manual edits."):
                run_analysis(s, cfg)
    if s.tracks is not None and s.analysed_with != analysis_key(cfg):
        st.info("Detection settings changed. Re-run detection to apply them; manual edits will be discarded.",
                icon=":material/info:")
    if info is not None:
        for w in limit_warnings(info) + s.get("analysis_warnings", []):
            st.warning(w, icon=":material/warning:")
    return info


def stats_strip(s) -> None:
    tracks = s.tracks
    cells = [
        ("Faces detected", sum(t.source != "manual" for t in tracks), ""),
        ("Hidden", sum(t.redact for t in tracks), ""),
        ("Left visible", sum(not t.redact for t in tracks), "warn" if any(not t.redact for t in tracks) else ""),
    ]
    if s.kind == "video":
        n = len(flagged_frames(tracks))
        cells.append(("Frames to check", n, "warn" if n else ""))
    else:
        cells.append(("Added by hand", sum(t.source == "manual" for t in tracks), ""))
    body = "".join(f'<div class="{cls}"><span>{label}</span><b>{value}</b></div>' for label, value, cls in cells)
    perf = ""
    if s.stats is not None:
        st_ = s.stats
        perf = (f'<div class="fg-perf">{st_.frames} frames analysed at {st_.fps:.0f} fps · '
                f"{st_.detector_calls} detector runs ({st_.early_detections} triggered by uncertain tracks)</div>")
    st.markdown(f'<div class="fg-stats">{body}</div>{perf}', unsafe_allow_html=True)


# --- frame navigation ------------------------------------------------------


def set_frame(value: int) -> None:
    st.session_state.scrub = value


def frame_controls(s, n_frames: int, fps: float) -> int:
    if "pending_seek" in s:
        s.scrub = s.pop("pending_seek")
    s.setdefault("scrub", 0)
    s.scrub = min(max(s.scrub, 0), n_frames - 1)
    frame = s.scrub
    stops = review_stops(s.tracks)
    prev_stop, next_stop = step_to(stops, frame, -1), step_to(stops, frame, +1)
    c = st.columns([0.6, 0.6, 7, 0.6, 0.6, 1.6, 1.6], vertical_alignment="center", gap="small")
    c[0].button("", icon=":material/fast_rewind:", help=f"Back {FRAME_JUMP} frames", on_click=set_frame,
                args=(max(frame - FRAME_JUMP, 0),), width="stretch", key="nav-back10")
    c[1].button("", icon=":material/chevron_left:", help="Previous frame", on_click=set_frame,
                args=(max(frame - 1, 0),), width="stretch", key="nav-back1")
    c[2].slider("Frame", 0, n_frames - 1, key="scrub", label_visibility="collapsed")
    c[3].button("", icon=":material/chevron_right:", help="Next frame", on_click=set_frame,
                args=(min(frame + 1, n_frames - 1),), width="stretch", key="nav-fwd1")
    c[4].button("", icon=":material/fast_forward:", help=f"Forward {FRAME_JUMP} frames", on_click=set_frame,
                args=(min(frame + FRAME_JUMP, n_frames - 1),), width="stretch", key="nav-fwd10")
    c[5].button("Previous", icon=":material/flag:", disabled=prev_stop is None, on_click=set_frame,
                args=(prev_stop or 0,), width="stretch", key="nav-prevflag",
                help="Previous place where a box is a tracker estimate the detector did not confirm")
    c[6].button("Next", icon=":material/flag:", disabled=next_stop is None, on_click=set_frame,
                args=(next_stop or 0,), width="stretch", key="nav-nextflag",
                help="Next place where a box is a tracker estimate the detector did not confirm")
    st.markdown(
        f'<div class="fg-timecode"><b>{frame / fps:.2f} s</b> · frame {frame} / {n_frames - 1}'
        f'<span>{len(stops)} place{"s" if len(stops) != 1 else ""} to check</span></div>',
        unsafe_allow_html=True,
    )
    return frame


# --- review: canvas, people, new box ---------------------------------------


def editor(s, cfg: Config, base: np.ndarray, frame: int) -> None:
    view = st.segmented_control("View", ["Edit", "Preview", "Original"], default="Edit", key="view",
                                label_visibility="collapsed") or "Edit"
    if view == "Preview":
        shown = render_image(base, boxes_at(s.tracks, frame), cfg)
        hint = "Preview of the exported result."
    else:
        shown = base
        hint = ("Click a face to hide or show it. Drag on the image to cover a missed face."
                if view == "Edit" else "Original, unmodified.")
    res = face_canvas(shown, s.tracks, frame, key=f"canvas-{s.upload_id}", show_boxes=view == "Edit",
                      draft=s.draft if view == "Edit" else None, can_draw=view == "Edit", hint=hint)
    if res is not None and res.clicked is not None:
        t = by_id(s, int(res.clicked))
        if t is not None:
            t.redact = not t.redact
            tracks_changed(s)
            st.rerun()
    if res is not None and res.drawn:
        d = res.drawn
        h, w = base.shape[:2]
        set_draft(s, Box(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])).clip(w, h), frame)
        st.rerun()


def set_draft(s, box: Box, frame: int) -> None:
    s.draft = box
    s.draft_frame = frame
    s.mx, s.my, s.mw, s.mh = box.x, box.y, max(box.w, 1), max(box.h, 1)


def people(s, frame: int) -> None:
    tracks: list[Track] = s.tracks
    section("People", f"{len(tracks)}")
    if not tracks:
        st.markdown('<div class="fg-empty">No faces detected. If someone is visible, drag a box over '
                    "them on the image.</div>", unsafe_allow_html=True)
        return
    options = ["All", "In frame"] + (["To check"] if s.kind == "video" else [])
    which = st.segmented_control("Filter", options, default="All", key="pfilter",
                                 label_visibility="collapsed", width="stretch") or "All"
    if which == "In frame":
        shown = tracks_on_frame(tracks, frame)
    elif which == "To check":
        shown = [t for t in tracks if t.flags]
    else:
        shown = list(tracks)
    b = st.columns(2)
    if b[0].button("Hide all", icon=":material/visibility_off:", width="stretch"):
        for t in tracks:
            t.redact = True
        tracks_changed(s)
        st.rerun()
    if b[1].button("Show all", icon=":material/visibility:", width="stretch"):
        for t in tracks:
            t.redact = False
        tracks_changed(s)
        st.rerun()
    with st.container(height=min(PEOPLE_PANEL_PX, PERSON_ROW_PX * max(len(shown), 1) + 24), border=True):
        if not shown:
            st.caption("No one matches this filter.")
        for t in shown:
            person_row(s, t)
    merges = [e for e in s.events if e.kind == "duplicate"]
    cuts = sorted({e.frame for e in s.events if e.kind == "scene_cut"})
    if merges:
        with st.expander(f"Possible identity switches ({len(merges)})"):
            st.caption("Two tracks converged on one face and were merged. Check these frames.")
            for e in merges:
                st.button(f"Frame {e.frame}: #{e.dropped} merged into #{e.kept}", key=f"merge-{e.frame}-{e.dropped}",
                          on_click=set_frame, args=(e.frame,), type="tertiary")
    if cuts:
        st.caption(f"Scene cuts at frame {', '.join(map(str, cuts))}. Numbering restarts after a cut.")


def person_row(s, t: Track) -> None:
    v = s.editor_version
    c = st.columns([0.9, 2.6, 1.0, 0.55, 0.55], vertical_alignment="center", gap="small")
    thumb = s.thumbs.get(t.id)
    if thumb is not None:
        c[0].image(rgb(thumb), width=THUMB_SHOW_PX)
    meta = []
    if t.source == "manual":
        meta.append("added by hand")
    if s.kind == "video" and t.history:
        meta.append(f"frames {min(t.history)}–{max(t.history)}")
    if t.score is not None:
        meta.append(f"confidence {t.score:.2f}")
    flag = f'<span class="fg-flag">{len(t.flags)} to check</span>' if t.flags else ""
    state_cls = "on" if t.redact else "off"
    c[1].markdown(
        f'<div class="fg-person"><b>Person {t.id}</b><div><span class="fg-state {state_cls}">'
        f'{"Hidden" if t.redact else "Visible"}</span>{flag}</div><small>{" · ".join(meta)}</small></div>',
        unsafe_allow_html=True,
    )
    hide = c[2].toggle("Hide", value=t.redact, key=f"hide-{t.id}-{v}", label_visibility="collapsed",
                       help="Hide this person in the export")
    if hide != t.redact:
        t.redact = hide
        tracks_changed(s)
        st.rerun()
    if s.kind == "video" and t.history:
        target = min(t.flags) if t.flags else min(t.history)
        c[3].button("", icon=":material/my_location:", key=f"go-{t.id}-{v}", on_click=set_frame, args=(target,),
                    type="tertiary", help="Go to this person (first frame to check, if any)")
    if c[4].button("", icon=":material/delete:", key=f"del-{t.id}-{v}", type="tertiary",
                   help="Remove: not a face"):
        s.tracks = delete_track(s.tracks, t.id)
        tracks_changed(s)
        st.rerun()


def new_box_panel(s, cfg: Config, base: np.ndarray, frame: int, n_frames: int) -> None:
    h, w = base.shape[:2]
    if s.draft is None:
        st.button("Add box manually", icon=":material/add_box:", width="stretch", key="add-manual",
                  help="Or drag on the image.", on_click=set_draft,
                  args=(s, Box((w - max(24, w // 8)) // 2, (h - max(24, h // 6)) // 2,
                               max(24, w // 8), max(24, h // 6)), frame))
        return
    with st.container(border=True):
        section("New box", f"frame {s.get('draft_frame', frame)}" if s.kind == "video" else "")
        start = s.get("draft_frame", frame)
        c = st.columns(4)
        x = c[0].number_input("X", 0, w - 1, key="mx")
        y = c[1].number_input("Y", 0, h - 1, key="my")
        bw = c[2].number_input("Width", 1, w, key="mw")
        bh = c[3].number_input("Height", 1, h, key="mh")
        draft = Box(int(x), int(y), int(bw), int(bh)).clip(w, h)
        if draft != s.draft:
            s.draft = draft
            st.rerun()

        how, end, end_box = "single", None, None
        if s.kind == "video":
            how = st.selectbox(
                "Coverage",
                ("track", "hold", "interpolate"),
                format_func={
                    "track": "Follow the face with the tracker",
                    "hold": "Hold in place until an end frame",
                    "interpolate": "Move in a straight line to an end-frame box",
                }.get,
            )
            if how in ("hold", "interpolate"):
                end = st.number_input("End frame", start, n_frames - 1, min(n_frames - 1, start + 30))
            if how == "interpolate":
                e = st.columns(4)
                ex = e[0].number_input("End X", 0, w - 1, int(x))
                ey = e[1].number_input("End Y", 0, h - 1, int(y))
                ew = e[2].number_input("End width", 1, w, int(bw))
                eh = e[3].number_input("End height", 1, h, int(bh))
                end_box = Box(int(ex), int(ey), int(ew), int(eh)).clip(w, h)
                st.caption("Real motion is rarely linear. Review the interval before exporting.")

        a, b = st.columns(2)
        if a.button("Add box", type="primary", width="stretch"):
            try:
                if s.kind == "image":
                    history = {0: draft}
                elif how == "track":
                    with st.spinner("Tracking the box forward"):
                        tcfg = cfg if cfg.tracker in ("lk", "csrt", "kcf") else replace(cfg, tracker="lk")
                        history = propagate(s.src, start, draft, tcfg)
                elif how == "hold":
                    history = {f: draft for f in range(start, int(end) + 1)}
                else:
                    history = interpolate_boxes(start, draft, int(end), end_box)
                tid = next_track_id(s.tracks)
                s.tracks = s.tracks + [manual_track(tid, history)]
                s.thumbs[tid] = crop_thumbnail(base, draft, THUMB_PX, THUMB_PAD)
                s.draft = None
                tracks_changed(s)
                span = f", frames {min(history)}–{max(history)}" if s.kind == "video" else ""
                s.flash = f"Added person {tid}{span}"
                st.rerun()
            except FaceGuardError as exc:
                st.error(str(exc))
        if b.button("Cancel", width="stretch"):
            s.draft = None
            st.rerun()


# --- export ----------------------------------------------------------------


def export(s, cfg: Config) -> None:
    section("Export")
    to_check = len(flagged_frames(s.tracks))
    visible = [t.id for t in s.tracks if not t.redact]
    checks = []
    if visible:
        checks.append(("warn", f"{len(visible)} {'person' if len(visible) == 1 else 'people'} left visible "
                               f"on purpose (person {', '.join(map(str, visible))})."))
    if to_check:
        checks.append(("warn", f"{to_check} frames use tracker estimates the detector did not confirm. "
                               "Step through them with the flag buttons."))
    if cfg.mode != "fill":
        checks.append(("info", f"{MODE_LABELS[cfg.mode]} can be partly reversed. Use Solid when it matters."))
    if not checks:
        checks.append(("ok", "Every detected face is hidden with a solid fill."))
    st.markdown('<ul class="fg-checks">' + "".join(f'<li class="{k}">{html.escape(m)}</li>' for k, m in checks)
                + "</ul>", unsafe_allow_html=True)

    ext = s.src.suffix if s.kind == "image" else ".mp4"
    fmt = ext
    row = st.columns([2, 2, 3], vertical_alignment="bottom")
    if s.kind == "image":
        options = list(dict.fromkeys([ext, ".png", ".jpg"]))  # original first, no duplicates
        fmt = row[0].segmented_control("Format", options, default=ext,
                                       format_func=lambda e: "Original" if e == ext else e[1:].upper()) or ext
    out = s.job / f"{s.name}_redacted{fmt}"
    if row[1].button("Render and export", type="primary", icon=":material/ios_share:", width="stretch"):
        try:
            if s.kind == "image":
                img = render_image(load_image(s.src), boxes_at(s.tracks, 0), cfg)
                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt in (".jpg", ".jpeg") else []
                ok, buf = cv2.imencode(fmt, img, params)  # fresh encode: no EXIF copied
                if not ok:
                    raise FaceGuardError("Could not encode the image.")
                out.write_bytes(buf.tobytes())
                s.export = (out, "image/png" if fmt == ".png" else "image/jpeg")
            else:
                bar = st.progress(0.0, text="Rendering")

                def progress(done: int, total: int) -> None:
                    if total and (done % 5 == 0 or done == total):
                        bar.progress(min(done / total, 1.0), text=f"Rendering · {done} / {total}")

                stats = render_video(s.src, out, s.tracks, cfg, progress)
                bar.empty()
                s.export_warnings = stats.warnings
                s.export = (out, "video/mp4")
            st.rerun()
        except FaceGuardError as exc:
            st.error(str(exc))

    if s.export and Path(s.export[0]).exists():
        path, mime = s.export
        for w in s.get("export_warnings", []):
            st.warning(w, icon=":material/warning:")
        info = {"name": Path(path).name}
        if s.kind == "video":
            vi = video_info(s.src)
            info |= {"frames": vi.frame_count, "fps": vi.fps, "size": [vi.width, vi.height]}
        log = redaction_log(s.tracks, cfg, info, s.events)
        left, right = st.columns([3, 2], gap="large")
        with left:
            if s.kind == "video":
                st.video(str(path))
            else:
                st.image(rgb(cv2.imread(str(path))), width="stretch")
        with right:
            size_kb = Path(path).stat().st_size / 1024
            st.markdown(f'<div class="fg-file"><b>{html.escape(Path(path).name)}</b>'
                        f"<span>{size_kb:,.0f} KB</span></div>", unsafe_allow_html=True)
            st.download_button("Download", Path(path).read_bytes(), file_name=Path(path).name, mime=mime,
                               type="primary", width="stretch", icon=":material/download:")
            st.download_button("Redaction log (JSON)", json.dumps(log, indent=1), width="stretch",
                               file_name=f"{s.name}_redaction_log.json", mime="application/json",
                               icon=":material/description:",
                               help="Which regions were hidden, on which frames, and which were tracker "
                               "estimates. Contains no images.")
            st.caption("The original file is never modified. Location, capture time and other "
                       "metadata are not copied into the export.")


# --- pages -----------------------------------------------------------------


def empty_state() -> None:
    st.markdown(
        """
<div class="fg-intro">
  <div><b>1 · Detect</b><p>A neural face detector finds every face; in video, optical-flow
  tracking follows each person between detections.</p></div>
  <div><b>2 · Review</b><p>Click a face to keep it visible, drag to cover anything the detector
  missed, and step through every frame flagged as uncertain.</p></div>
  <div><b>3 · Export</b><p>Solid, pixelated or blurred masks, written to a new file without
  the original's metadata. Nothing leaves this computer.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )


def single(s, cfg: Config) -> None:
    handle_upload(s)
    if s.src is None:
        empty_state()
        return
    info = file_bar(s, cfg)
    if s.tracks is None:
        return
    if msg := s.pop("flash", None):
        st.toast(msg, icon=":material/check_circle:")
    stats_strip(s)

    if s.kind == "image":
        base, frame, n_frames = load_image(s.src), 0, 1
    else:
        n_frames = max(info.frame_count, 1)
        frame = frame_controls(s, n_frames, info.fps)
        try:
            base = load_frame(s.src, frame)
        except FaceGuardError as exc:
            st.error(str(exc))
            return

    left, right = st.columns([5, 3], gap="large")
    with left:
        editor(s, cfg, base, frame)
        if s.kind == "video":
            cuts = sorted({e.frame for e in s.events if e.kind == "scene_cut"})
            res = timeline(s.tracks, n_frames, info.fps, frame, cuts, key=f"timeline-{s.upload_id}")
            if res is not None and res.seek is not None and res.seek != frame:
                s.pending_seek = int(res.seek)
                st.rerun()
    with right:
        new_box_panel(s, cfg, base, frame, n_frames)
        people(s, frame)
    st.divider()
    export(s, cfg)


def batch(s, cfg: Config) -> None:
    st.markdown('<p class="fg-lead">Redact many photos at once with the current settings. Every detected '
                "face is hidden. To choose people or fix misses, open a photo in the Single file tab. "
                "For folders of video, use <code>python cli.py --input folder/ --output out/</code>.</p>",
                unsafe_allow_html=True)
    ups = st.file_uploader("Photos", type=[e.lstrip(".") for e in IMAGE_EXTS], accept_multiple_files=True,
                           key="batch_up", label_visibility="collapsed")
    if not ups:
        return
    if st.button(f"Redact {len(ups)} photo{'s' if len(ups) != 1 else ''}", type="primary",
                 icon=":material/shield:"):
        det = make_detector(cfg)
        bar = st.progress(0.0)
        results = []
        for k, up in enumerate(ups):
            img = cv2.imdecode(np.frombuffer(up.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({"name": up.name, "error": "Not a readable image"})
                continue
            faces = det.detect(img)
            out = render_image(img, [f.box for f in faces], cfg)
            ext = Path(up.name).suffix.lower()
            ok, buf = cv2.imencode(ext, out, [cv2.IMWRITE_JPEG_QUALITY, 95] if ext != ".png" else [])
            results.append({"name": up.name, "faces": len(faces), "bytes": buf.tobytes() if ok else b"",
                            "thumb": crop_thumbnail(out, Box(0, 0, out.shape[1], out.shape[0]), BATCH_THUMB_PX)})
            bar.progress((k + 1) / len(ups), text=f"{k + 1} / {len(ups)}")
        bar.empty()
        s.batch = results
    if s.batch:
        ok = [r for r in s.batch if "error" not in r]
        zero = [r["name"] for r in ok if r["faces"] == 0]
        row = st.columns([3, 1], vertical_alignment="center")
        row[0].markdown(f'<div class="fg-file"><b>{len(ok)} photos redacted</b>'
                        f'<span>{sum(r["faces"] for r in ok)} faces hidden</span>'
                        + (f'<span class="warn">{len(zero)} with no faces found</span>' if zero else "")
                        + "</div>", unsafe_allow_html=True)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for r in ok:
                z.writestr(r["name"], r["bytes"])
        row[1].download_button("Download all", zbuf.getvalue(), file_name="faceguard_redacted.zip",
                               mime="application/zip", type="primary", icon=":material/download:", width="stretch")
        cols = st.columns(BATCH_COLS)
        for k, r in enumerate(s.batch):
            with cols[k % BATCH_COLS]:
                if "error" in r:
                    st.error(f"{r['name']}: {r['error']}")
                else:
                    note = " · check by eye" if r["faces"] == 0 else ""
                    st.image(rgb(r["thumb"]), caption=f"{r['name']} · {r['faces']} faces{note}", width="stretch")


def main() -> None:
    s = state()
    cfg = sidebar()
    header(s)
    tab_single, tab_batch = st.tabs(["Single file", "Batch photos"])
    with tab_single:
        single(s, cfg)
    with tab_batch:
        batch(s, cfg)


main()
