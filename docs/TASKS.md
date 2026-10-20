# FaceGuard — Build Plan

Build in order. Each milestone must run end-to-end before the next one starts.
Tick boxes in the same commit as the work.

---

## Week 1 — Images (M1)

- [ ] Repo skeleton, venv, `requirements.txt`
- [ ] `scripts/download_model.py` fetches the YuNet ONNX and verifies its size
- [ ] `core/types.py`: `Box` (with `iou`, `pad`), `Detection`, `Track`, `Config`
- [ ] `core/detectors.py`: `YuNetDetector` with correct `setInputSize` handling
      and `None` guard
- [ ] `core/redact.py`: blur, pixelate, fill with elliptical feathered masks
- [ ] `cli.py` handles single images
- [ ] Unit tests for `Box.iou` and `Box.pad` edge cases (box at frame border)

**Acceptance:** a photo containing 5 faces is correctly redacted in all three
modes, exported at original resolution, with no visible rectangular seams.

---

## Week 2 — Video baseline (M2)

- [ ] `core/video_io.py`: streaming frame reader, writer preserving fps and size
- [ ] `core/pipeline.py`: per-frame detection, redact, write
- [ ] ffmpeg audio mux behind `--keep-audio`, graceful when ffmpeg is absent
- [ ] Progress reporting (frames done / total)
- [ ] `HaarDetector` behind the same interface
- [ ] Record baseline FPS for per-frame detection in `docs/RESULTS.md`

**Acceptance:** a 20 s clip processes end to end, exports with audio intact,
plays correctly. Flicker is expected at this stage — that is the point.

---

## Week 3 — Tracking and association (M3)

- [ ] `core/tracking.py`: `LKTracker` with Shi-Tomasi seeding, pyramidal LK,
      forward-backward error check, median translation and scale
- [ ] `CSRTTracker`, `KCFTracker` wrappers with the version-safe constructor
- [ ] `core/association.py`: IoU matching, track creation, grace period,
      ID-switch logging
- [ ] Pipeline runs detection every N frames and tracks in between
- [ ] Detection re-anchors matched tracks (drift reset)
- [ ] Measure FPS and leak frames against Week 2 baseline

**Acceptance:** same clip at `--detect-interval 10 --tracker lk` has **higher
FPS and fewer leak frames** than the Week 2 baseline, with both numbers written
into `docs/RESULTS.md`.

---

## Week 4 — UI and correction (M4)

- [ ] `ui/app.py`: upload, process, preview
- [ ] Numbered overlay boxes per track
- [ ] Per-track redact toggle (selective redaction)
- [ ] Manual box drawing, stored as a `source="manual"` track and propagated by
      the tracker like a detection
- [ ] Delete a false-positive track
- [ ] Mode, interval, padding and strength controls
- [ ] Frame scrubber for video preview
- [ ] Privacy hint in the UI: fill is strongest, blur is weakest

**Acceptance:** a user can untick person 2 of 4, draw one manual box over a
missed face, preview, and export, without touching the CLI.

**Freeze features here.** Everything after this is evaluation and writing.

---

## Week 5 — Evaluation and report (M5)

- [ ] `eval/annotate.py` to label ground-truth boxes
- [ ] Annotate 30–50 images and 3–5 clips (**start this in Week 3**, it is the
      most commonly underestimated task in the whole project)
- [ ] `eval/metrics.py`: precision, recall, F1 at IoU 0.5; redaction coverage;
      leak-frame count; FPS; ID consistency
- [ ] `eval/run_experiments.py` ablation sweep
- [ ] Plots: FPS vs detect-interval; leak frames vs detect-interval; PR curve
- [ ] `docs/RESULTS.md` filled in
- [ ] Final report, demo clip, viva prep

**Acceptance:** the full results table and all plots regenerate from one command.

---

## Stretch goals — only if Weeks 1–4 finished early

- [ ] Allow-list via face recognition (SFace embeddings) — "blur everyone but me"
- [ ] Batch folder processing
- [ ] Temporal smoothing of box positions to reduce mask jitter
- [ ] Tiled detection pass for small background faces

---

## Cut order if time runs short

Cut in this order, not any other:

1. Selective redaction (redact everything instead)
2. CSRT / KCF comparison (keep LK only)
3. Video (images only)

A photo tool with real measured numbers scores better than a half-working video
tool with no evaluation.

---

## Demo day checklist

- [ ] A 15 s clip already tested on the demo machine. **Do not record live.**
- [ ] A crowd photo showing selective redaction clearly
- [ ] A deliberate failure case ready to show (profile face), with the
      explanation of why it fails and how tracking covers it
- [ ] Results table printed or on a slide
- [ ] Network disabled during the demo, to prove the offline claim
