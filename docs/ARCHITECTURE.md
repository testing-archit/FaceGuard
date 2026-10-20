# FaceGuard — Architecture

## 1. Repository layout

```
faceguard/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── cli.py                      # headless entry point
├── models/
│   └── face_detection_yunet_2026may.onnx   # committed; Haar XML ships with cv2.data
├── scripts/
│   └── download_model.py       # one-time, verifies SHA-256
├── core/
│   ├── __init__.py
│   ├── types.py                # Box, Detection, Track, Config dataclasses
│   ├── detectors.py            # YuNetDetector, HaarDetector
│   ├── tracking.py             # LKTracker, CSRTTracker, KCFTracker
│   ├── association.py          # IoU matching, track lifecycle
│   ├── redact.py               # blur / pixelate / fill, mask building
│   ├── video_io.py             # frame reader, writer, ffmpeg audio mux
│   └── pipeline.py             # orchestration
├── ui/
│   └── app.py                  # Streamlit; no CV logic lives here
├── eval/
│   ├── annotate.py             # manual ground-truth annotation helper
│   ├── metrics.py              # precision, recall, coverage, leak frames
│   ├── run_experiments.py      # ablation sweeps
│   └── experiments.yaml
├── data/
│   ├── samples/
│   └── ground_truth.json
├── tests/
└── docs/
    ├── SPEC.md
    ├── ARCHITECTURE.md
    ├── TASKS.md
    ├── EVALUATION.md
    ├── DECISIONS.md
    └── RESULTS.md
```

**Layering rule:** `ui/` and `eval/` may import `core/`. `core/` imports
neither. This keeps the pipeline testable headless.

## 2. Data model (`core/types.py`)

```python
@dataclass
class Box:
    x: int; y: int; w: int; h: int

    def iou(self, other: "Box") -> float: ...
    def pad(self, frac: float, frame_w: int, frame_h: int) -> "Box": ...
        # frac is PER SIDE: w grows to w * (1 + 2 * frac), then clipped to frame
    @property
    def center(self) -> tuple[float, float]: ...

@dataclass
class Detection:
    box: Box
    score: float
    landmarks: np.ndarray | None = None   # shape (5, 2): eyes, nose, mouth corners

@dataclass
class Track:
    id: int
    box: Box
    redact: bool = True              # user toggle
    source: str = "detector"         # "detector" | "manual"
    last_seen_frame: int = 0         # last frame a detection matched this track
    lost: bool = False               # tracker returned None; box is frozen
    history: dict[int, Box] = field(default_factory=dict)   # frame -> box

@dataclass
class Config:
    # Every tunable number lives here. See the field comments for units.
    model_path: str = "models/face_detection_yunet_2026may.onnx"
    # detection
    detector: str = "yunet"          # yunet | haar
    conf_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    min_face_px: int = 24            # Haar minSize; also the "small face" bound
    # tracking
    tracker: str = "lk"              # lk | csrt | kcf | none
    detect_interval: int = 10        # run detector every N frames
    lk_max_corners: int = 40
    lk_quality_level: float = 0.01
    lk_min_distance: int = 3
    lk_win_size: int = 21
    lk_pyr_levels: int = 3
    lk_fb_threshold_px: float = 1.0  # forward-backward round-trip tolerance
    lk_min_points: int = 6           # fewer survivors than this = lost
    # association
    iou_threshold: float = 0.3
    max_misses: int = 15             # grace period, in VIDEO FRAMES
    duplicate_iou: float = 0.7       # two tracks above this are one face
    # redaction
    mode: str = "pixelate"           # blur | pixelate | fill
    padding: float = 0.25            # per side; >= 0.207 needed, see section 7
    blur_strength: float = 0.35      # kernel size as fraction of box width
    blur_min_kernel: int = 11
    pixel_blocks: int = 8            # downscale target, N x N
    feather_frac: float = 0.08       # soft edge width as fraction of box width
    fill_color: tuple[int, int, int] = (0, 0, 0)   # BGR
    # io
    keep_audio: bool = False
```

## 3. Interfaces

```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

class Tracker(Protocol):
    def init(self, frame: np.ndarray, gray: np.ndarray, box: Box) -> None: ...
    def update(self, frame: np.ndarray, gray: np.ndarray) -> Box | None: ...  # None = lost
```

Both receive the grayscale copy alongside the BGR frame so the conversion
happens once per frame in the pipeline, not once per track (section 10).
`Detector.detect` also accepts an optional `gray=` for the same reason.

`tracker="none"` is a `NullTracker` that returns the box unchanged: boxes are
held between detections. With `detect_interval=1` this is the pure per-frame
detection baseline.

Everything downstream depends only on these two. That is what makes the
YuNet-vs-Haar and LK-vs-CSRT experiments one config line each.

## 4. Detection

### YuNetDetector

```python
self._model = cv2.FaceDetectorYN.create(
    model="models/face_detection_yunet_2026may.onnx",
    config="",                    # empty string for ONNX models
    input_size=(320, 320),        # overwritten per frame below
    score_threshold=cfg.conf_threshold,
    nms_threshold=cfg.nms_threshold,
    top_k=5000,
)
```

Per frame:

```python
h, w = frame.shape[:2]
self._model.setInputSize((w, h))     # MANDATORY before detect()
retval, faces = self._model.detect(frame)
if faces is None:
    return []
```

Each row of `faces` is 15 floats: `x, y, w, h`, then five landmark `(x, y)`
pairs (right eye, left eye, nose tip, right mouth corner, left mouth corner),
then the confidence score. YuNet handles its own preprocessing, multi-scale
decoding and non-max suppression internally, so do not reimplement any of it.

The model is 229,738 bytes (SHA-256 `ebafce4e…22f0f0`), which is why CPU
inference is viable and why it is committed to the repo rather than downloaded
at runtime. The `2026may` file is the `2023mar` model re-exported with dynamic
H/W input dims, which OpenCV 5's ONNX engine needs for arbitrary frame sizes.

### HaarDetector

`cv2.CascadeClassifier` on the grayscale frame, loaded from
`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`. Exists purely
as the classical
baseline for the report. No landmarks, no confidence score — synthesise
`score=1.0` so it satisfies the `Detection` type.

## 5. Tracking

### Why track at all

Running detection on every frame is slower and, more importantly, flickers. A
face detected in frames 1–40, missed in 41–43, and detected again in 44 produces
three frames where the person is fully visible in a supposedly redacted video.
Tracking fills those gaps. Detection re-anchors tracks so drift cannot
accumulate.

### LKTracker (primary — this is the Module III/IV content)

1. On `init`, convert the frame to grayscale and run
   `cv2.goodFeaturesToTrack` **inside the face box only** (Shi-Tomasi corners;
   `lk_max_corners`, `lk_quality_level`, `lk_min_distance` in `Config`).
2. On `update`, run `cv2.calcOpticalFlowPyrLK` from the previous grayscale
   frame to the current one.
3. Run the **forward-backward check**: track points forward, then track the
   results backward, and discard any point that does not return within
   `lk_fb_threshold_px` (1 px) of where it started. This is the single most effective way to kill bad
   correspondences.
4. Translation = median displacement of surviving points.
   Scale = median ratio of pairwise point distances, new over old.
5. If fewer than `lk_min_points` (6) survive, return `None` (track lost).

Document the assumptions in the docstring: brightness constancy, small motion
between frames, local spatial coherence. The image pyramid is what relaxes the
small-motion assumption. These are viva questions.

### CSRTTracker / KCFTracker

Thin wrappers over OpenCV's Tracking API for comparison. CSRT is more accurate
and slower; KCF is faster and loses targets sooner. The Python constructor names
differ between OpenCV 4 and 5, so:

```python
def _make_tracker(kind: str):
    for name in (f"Tracker{kind}_create", f"Tracker{kind}"):
        fn = getattr(cv2, name, None)
        if fn is not None:
            return fn.create() if hasattr(fn, "create") else fn()
    raise RuntimeError(
        f"{kind} tracker unavailable. Install opencv-contrib-python."
    )
```

## 6. Association (`core/association.py`)

Called every time the detector runs (every N frames). Trackers have **already
been updated on this frame**, so detections are compared with where each face
is now, not where it was on the previous detection frame.

```
detections D, existing tracks T, current frame i
1. build IoU matrix between D and T
2. greedy assignment: repeatedly take the highest-IoU pair, stop below
   iou_threshold (no scipy dependency; with < 20 faces greedy and Hungarian
   almost always agree)
3. for each matched pair:
       track.box = detection.box          # re-anchor, kills drift
       track.last_seen_frame = i
       track.lost = False
       re-init the tracker on the new box
4. for each unmatched detection:
       create a new Track with the next free id, redact = True
       init a tracker on it
5. for each unmatched track:
       if i - track.last_seen_frame > max_misses: delete the track
6. for each pair of tracks with IoU > duplicate_iou:
       keep the older id, drop the other, log a possible ID switch with frame i
```

The grace period is measured in frames (`i - last_seen_frame`), so its meaning
does not change with `detect_interval`. A track in its grace period is still
propagated and still redacted. If its tracker has lost it, the box stays
frozen at the last position: this is a known leak source when the person keeps
moving, and it is counted in the leak-frame metric.

Track IDs are what make selective redaction work: the user toggles person 3 once
and the decision follows that person for the rest of the clip.

## 7. Redaction (`core/redact.py`)

```python
def redact(frame, boxes, cfg) -> np.ndarray
```

For each box:

1. Pad by `cfg.padding` (detector boxes hug the face; hairline, ears and chin
   fall outside them), clipped to frame bounds.
2. Extract the ROI.
3. Apply the mode:
   - **blur** — `cv2.GaussianBlur`, kernel = `odd(box.w * cfg.blur_strength)`,
     minimum 11. Scaling with box size matters: a fixed kernel under-blurs large
     faces and obliterates small ones.
   - **pixelate** — resize down to `pixel_blocks × pixel_blocks` with
     `INTER_LINEAR`, resize back up with `INTER_NEAREST`.
   - **fill** — solid colour.
4. Build the mask: a filled ellipse inscribed in the padded box, Gaussian
   blurred by `feather_frac * box.w`, then `alpha = clip(2 * blurred, 0, 1)`.
   Blurring a hard edge puts alpha 0.5 exactly on the edge; doubling and
   clipping makes everything inside the ellipse fully opaque, so the soft ramp
   lies **outside** the ellipse. Then alpha-composite the redacted ROI over the
   original. The ROI is the padded box grown by the feather width.

Why `padding >= 0.207`: the face box has half-width `w/2`; an ellipse inscribed
in a box padded by `p` per side has semi-axis `(0.5 + p) w`. The face box
corner `(w/2, h/2)` is inside the ellipse when `2 * (0.5 / (0.5 + p))^2 <= 1`,
i.e. `p >= (√2 − 1) / 2 ≈ 0.207`. The default 0.25 leaves a margin.

Privacy ranking, which belongs in the report and in the UI as a hint:

| Mode | Reversibility |
|---|---|
| Gaussian blur | Weakest. Deblurring and face-restoration models can partially recover features |
| Pixelation | Stronger, but attackable given a small candidate set of faces |
| Solid fill | Destroys the information |

## 8. Pipeline (`core/pipeline.py`)

The pipeline is split into **analyse** (find boxes) and **render** (apply them).

```
analyse(video, cfg) -> TrackSet          # per-track {frame: Box}, boxes only
    for each frame i:
        gray = to_gray(frame)                        # once per frame
        for each track: box = tracker.update(frame, gray)
                        if box is None: track.lost = True  (box frozen)
                        else: track.box = box
        if i % detect_interval == 0:
            detections = detector.detect(frame, gray)
            associate(detections, tracks, i)         # section 6
        record track.box into track.history[i] for every live track
        yield progress

render(video, trackset, selected_ids, cfg) -> output file
    for each frame i:
        boxes = [t.history[i] for t in trackset if t.id in selected_ids and i in t.history]
        write redact(frame, boxes, cfg)
        yield progress
    finalise via ffmpeg (section 9)
```

Images use the same two functions on a single frame (no tracking).

Stream frames in both passes. Never accumulate decoded frames in memory
(NFR-4); a `TrackSet` is a few kilobytes of integers.

**UI:** analyse once, let the user toggle, delete and add tracks, then render.
Render **replays** the stored boxes and never re-runs detection, so the IDs the
user toggled are exactly the IDs that get rendered. A manual box drawn on frame
`f` becomes a new track whose boxes are produced by `propagate(video, f, box,
cfg)`: the configured tracker runs forward from `f` until it loses the face or
the clip ends.

**CLI:** analyse, then render with every track selected.

**Evaluation:** `analyse` output is written to `preds.json`. Coverage is
computed by rasterising the same `build_mask` used by `render`, so the metric
measures exactly what the exported video shows.

## 9. Video IO and audio (`core/video_io.py`)

`cv2.VideoWriter` writes video only, and its portable `mp4v` codec does not
play in Chrome or in Streamlit's `st.video`. So the writer produces a temporary
silent file and ffmpeg finalises it, re-encoding to H.264 and optionally adding
the original audio:

```bash
ffmpeg -y -i redacted.silent.mp4 -i original.mp4 \
       -map 0:v:0 -map 1:a:0? -c:v libx264 -pix_fmt yuv420p -crf 18 \
       -c:a aac -shortest output.mp4
```

Without `--keep-audio` the second input and audio map are omitted. The `?`
makes the audio stream optional so files with no audio track do not fail.
If ffmpeg is missing, warn and keep the `mp4v` file (it plays in VLC but may
not preview in the browser) rather than crashing.

## 10. Colour space handling

Frames arrive BGR. Grayscale is needed for optical flow and for the Haar
cascade. Convert **once per frame** and pass the grayscale copy to both; do not
convert twice. Gradient-based optical flow only needs intensity, so discarding
chroma costs nothing and saves two thirds of the data. Convert BGR to RGB only
at the Streamlit boundary.

## 11. Error handling

| Condition | Behaviour |
|---|---|
| Model file missing | Clear message pointing at `scripts/download_model.py` |
| `opencv-python` installed instead of contrib | Detected at import, explicit message |
| Unsupported file type | Rejected with the list of supported extensions |
| Video fails to open | Message naming likely codec cause |
| ffmpeg missing with `--keep-audio` | Warn, export silent video, exit 0 |
| Zero faces detected | Not an error. Export the unchanged copy and say so |
