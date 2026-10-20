# CLAUDE.md — FaceGuard

## What this project is

FaceGuard is an **offline desktop tool** that detects faces in images and short
videos, lets the user choose which faces to hide, and exports a redacted copy.
It is a BTech CSE academic project. The deliverable is not just working code: it
is working code **plus measured results** comparing detection and tracking
strategies.

Read `docs/SPEC.md` for requirements, `docs/ARCHITECTURE.md` for design,
`docs/TASKS.md` for the current milestone, `docs/EVALUATION.md` for metrics.

## Hard constraints — do not violate these

1. **Fully offline at runtime.** No network calls during processing. No cloud
   APIs, no paid services, no telemetry. The ONNX model is downloaded once
   during setup and committed to `models/`.
2. **CPU only.** No CUDA, no GPU-specific code paths. Must run on a mid-range
   laptop.
3. **No model training.** Use the pretrained YuNet ONNX model as-is.
4. **No face recognition / identification.** Out of scope. Do not add embedding
   models (SFace etc.) unless `docs/TASKS.md` Week 5 stretch goals are reached.
5. **No camera or live capture.** Input is uploaded files only.

## Stack

- Python 3.10+
- `opencv-contrib-python>=4.10,<6` (contrib is required — `TrackerCSRT` and
  `TrackerKCF` are not in the base `opencv-python` wheel)
- `numpy`
- `streamlit` for the UI
- `ffmpeg` on PATH, used only to mux the original audio back into exported video
- Detector model: `models/face_detection_yunet_2026may.onnx` from OpenCV Zoo

## API gotchas — these have cost people days

- `cv2.FaceDetectorYN` requires OpenCV >= 4.5.4 and the `objdetect` module.
- **You must call `detector.setInputSize((w, h))` with the actual frame size
  before every `detect()` call on a new resolution**, or returned coordinates
  will be wrong.
- `FaceDetectorYN.detect()` returns `(retval, faces)` where `faces` is `None`
  when nothing is found. Always guard for `None` before indexing.
- Each row of `faces` is 15 values: `[x, y, w, h, 5 landmark xy pairs, score]`.
- Tracker constructor names differ between OpenCV 4 and 5 Python bindings.
  Wrap construction in a helper that tries both and raises a clear error.
- `cv2.VideoWriter` **drops audio**. Write a silent video, then mux audio with
  ffmpeg. Never claim audio is preserved without the mux step.
- OpenCV uses BGR, not RGB. Streamlit expects RGB. Convert at the UI boundary
  only, never inside `core/`.

## Code conventions

- `core/` must have **zero** Streamlit or UI imports. The pipeline has to be
  runnable headless from `cli.py` for the evaluation scripts.
- Use `@dataclass` for `Box`, `Detection`, `Track`, `Config`. No dicts of
  loose keys passed around.
- Type hints on every public function.
- Every tunable number (detection interval, thresholds, blur strength, box
  padding) lives in `Config` with a documented default. **No magic numbers
  inline.**
- Detectors implement one interface: `detect(frame: np.ndarray) -> list[Detection]`.
  Swapping YuNet for Haar must be a single config change.
- Trackers implement: `init(frame, box)`, `update(frame) -> Box | None`.
- Prefer small pure functions that take and return arrays. Easier to test.
- Docstrings should state the CV concept used (e.g. "pyramidal Lucas-Kanade,
  brightness-constancy assumption") because this feeds the report.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py        # one-time, writes models/*.onnx

# run UI
streamlit run ui/app.py

# headless
python cli.py --input data/samples/clip.mp4 --output out.mp4 --mode pixelate
python cli.py --input data/samples/photo.jpg --output out.jpg --mode blur

# evaluation
python eval/run_experiments.py --config eval/experiments.yaml
python eval/metrics.py --pred out/preds.json --truth data/ground_truth.json

# tests
pytest -q
```

## Working style

- Build strictly in the order of `docs/TASKS.md`. Each milestone must run
  end-to-end before starting the next. A half-finished tracker on top of a
  half-finished exporter is unrecoverable.
- When a milestone's acceptance criteria are met, tick the boxes in
  `docs/TASKS.md` in the same commit.
- Commit messages: `feat(core): add IoU track association`.
- If a design decision has a trade-off (accuracy vs speed), add one line to
  `docs/DECISIONS.md` with the choice and the reason. This becomes report text.
- Do not add dependencies without a reason written in the commit message.

## Known failure modes to handle, not ignore

- Profile faces and heads turned >60 degrees are frequently missed by YuNet.
  Tracking is what covers these frames.
- Faces smaller than ~24 px are unreliable.
- Flow-based boxes drift. Detection must re-anchor tracks at a fixed interval.
- When two people cross, IoU association can swap IDs. Log these events.
- A single frame with an uncovered face defeats the whole tool. Metrics must be
  **per-frame**, never per-video averages only.
