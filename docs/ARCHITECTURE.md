# FaceGuard — Architecture

## 1. Repository layout

```
faceguard/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── cli.py                      # headless entry point
├── models/
│   ├── face_detection_yunet_2026may.onnx
│   └── haarcascade_frontalface_default.xml
├── scripts/
│   └── download_model.py       # one-time, verifies checksum
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
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Box:
    x: int; y: int; w: int; h: int

    def iou(self, other: "Box") -> float: ...
    def pad(self, frac: float, frame_w: int, frame_h: int) -> "Box": ...
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
    last_seen_frame: int = 0
    misses: int = 0                  # consecutive frames without a match
    history: list[tuple[int, Box]] = field(default_factory=list)

@dataclass
class Config:
    # detection
    detector: str = "yunet"          # yunet | haar
    conf_threshold: float = 0.6
    nms_threshold: float = 0.3
    # tracking
    tracker: str = "lk"              # lk | csrt | kcf | none
    detect_interval: int = 10        # run detector every N frames
    # association
    iou_threshold: float = 0.3
    max_misses: int = 15             # grace period before deleting a track
    # redaction
    mode: str = "pixelate"           # blur | pixelate | fill
    padding: float = 0.20
    blur_strength: float = 0.35      # kernel size as fraction of box width
    pixel_blocks: int = 8            # downscale target, N x N
    feather: int = 9                 # mask edge softening, odd number
    # io
    keep_audio: bool = False
```

## 3. Interfaces

```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

class Tracker(Protocol):
    def init(self, frame: np.ndarray, box: Box) -> None: ...
    def update(self, frame: np.ndarray) -> Box | None: ...   # None = lost
```

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

Model size is roughly 338 KB, which is why CPU inference is viable and why the
model is committed to the repo rather than downloaded at runtime.

### HaarDetector

`cv2.CascadeClassifier` on the grayscale frame. Exists purely as the classical
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
   `cv2.goodFeaturesToTrack` **inside the face box only** (Shi-Tomasi corners,
   `maxCorners=40`, `qualityLevel=0.01`, `minDistance=3`).
2. On `update`, run `cv2.calcOpticalFlowPyrLK` from the previous grayscale
   frame to the current one.
3. Run the **forward-backward check**: track points forward, then track the
   results backward, and discard any point that does not return within 1 px of
   where it started. This is the single most effective way to kill bad
   correspondences.
4. Translation = median displacement of surviving points.
   Scale = median ratio of pairwise point distances, new over old.
5. If fewer than 6 points survive, return `None` (track lost).

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

Called every time the detector runs (every N frames).

```
detections D, existing tracks T
1. build cost matrix C where C[i][j] = 1 - IoU(D[i].box, T[j].box)
2. greedy (or Hungarian) assignment, rejecting any pair with IoU < iou_threshold
3. for each matched pair:
       track.box = detection.box          # re-anchor, kills drift
       track.misses = 0
       re-init the tracker on the new box
4. for each unmatched detection:
       create a new Track with the next free id, redact = True
5. for each unmatched track:
       track.misses += 1
       if track.misses > max_misses: delete the track
6. if two tracks' boxes overlap by IoU > 0.7 after assignment:
       log a possible ID switch with the frame number
```

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
4. Build an elliptical mask inscribed in the padded box, blur the mask itself by
   `cfg.feather` to soften edges, then alpha-composite the redacted ROI over the
   original.

Privacy ranking, which belongs in the report and in the UI as a hint:

| Mode | Reversibility |
|---|---|
| Gaussian blur | Weakest. Deblurring and face-restoration models can partially recover features |
| Pixelation | Stronger, but attackable given a small candidate set of faces |
| Solid fill | Destroys the information |

## 8. Pipeline (`core/pipeline.py`)

```
open input
if image:
    detect -> (user selection) -> redact -> write
if video:
    for each frame i:
        if i % detect_interval == 0:
            detections = detector.detect(frame)
            tracks = associate(detections, tracks, i)
            re-init trackers on matched boxes
        else:
            for each track: box = tracker.update(frame)
                            if box is None: track.misses += 1
        redact(frame, [t.box for t in tracks if t.redact])
        write frame
        yield progress
close writer
if keep_audio: ffmpeg mux original audio into output
```

Stream frames. Never accumulate the whole decoded video in memory (NFR-4).

Processing happens in two passes in the UI: pass 1 collects tracks so the user
can review and toggle, pass 2 renders the output. The CLI runs a single pass
with everything redacted.

## 9. Video IO and audio (`core/video_io.py`)

`cv2.VideoWriter` writes video only. To preserve audio, write the silent
redacted file first, then:

```bash
ffmpeg -i redacted_silent.mp4 -i original.mp4 \
       -c:v copy -map 0:v:0 -map 1:a:0? -shortest output.mp4
```

The `?` makes the audio stream optional so files with no audio track do not
fail. If ffmpeg is missing, warn and return the silent file rather than
crashing.

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
