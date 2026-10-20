# FaceGuard — Functional Specification

## 1. Problem statement

Content creators, event organisers, teachers and student journalists routinely
capture bystanders on camera. Publishing that footage exposes people who never
consented. Existing options are manual (frame-by-frame blurring in a video
editor, too slow to be used) or cloud-based (uploads faces to a third party,
which defeats the purpose and requires connectivity).

FaceGuard provides automated, locally-executed face redaction for images and
short videos, with human review before export.

## 2. Users

| User | Need |
|---|---|
| Student vlogger / campus content creator | Hide bystanders in vox-pop and campus footage |
| Event photographer | Redact crowd photos before sharing |
| Teacher | Share a classroom clip without exposing students |
| Student journalist | Protect sources and uninvolved passers-by |

## 3. Goals

- G1 — Detect faces in images and short videos (10–30 s) with high recall.
- G2 — Keep each face covered across **every** frame it appears in, including
  frames where the detector fails.
- G3 — Let the user choose per-person which faces to hide.
- G4 — Let the user manually fix misses before export.
- G5 — Run entirely offline on laptop CPU.
- G6 — Produce measured evidence comparing detection and tracking strategies.

## 4. Non-goals

- Face recognition or identification of specific individuals
- Live webcam or real-time stream processing
- Full-length video (> 2 minutes)
- Body, licence plate or text redaction
- Model training or fine-tuning
- Mobile or web-hosted deployment

## 5. Functional requirements

### Input
- **FR-1** Accept image files: `.jpg`, `.jpeg`, `.png`
- **FR-2** Accept video files: `.mp4`, `.avi`, `.mov`
- **FR-3** Reject unsupported files with a clear message, never a stack trace
- **FR-4** Warn, but proceed, if video exceeds 60 seconds or 1080p. Videos
  over 2 minutes are processed but are outside the tested envelope (NFR-2)

### Detection
- **FR-5** Detect faces using YuNet via `cv2.FaceDetectorYN`
- **FR-6** Expose a Haar cascade detector behind the same interface
- **FR-7** Confidence threshold configurable, default 0.6
- **FR-8** Return box, 5 facial landmarks and score per detection
- **FR-9** Operate with no network access after model download

### Tracking (video only)
- **FR-10** Run detection every N frames, N configurable, default 10
- **FR-11** Propagate boxes between detections with pyramidal Lucas-Kanade flow
- **FR-12** Offer CSRT and KCF as alternative trackers for comparison
- **FR-13** Assign a stable integer ID to each tracked person
- **FR-14** Associate new detections to existing tracks by IoU, threshold 0.3
- **FR-15** Keep an unmatched track alive so brief detector misses do not
  break identity or expose the face. While the tracker still follows the
  face, keep it for up to 60 frames without a detector match (drift cap); if
  the tracker loses it, keep the frozen box for a 15-frame grace period.
  Tracks in either state **continue to be redacted**
- **FR-16** Log every suspected ID switch (two tracks converging on one face)
  for the evaluation report

### Redaction
- **FR-17** Support three modes: Gaussian blur, pixelation, solid fill
- **FR-18** Scale blur kernel and pixelation block size with face box size
- **FR-19** Pad each box by a configurable fraction (default 0.2) before masking
- **FR-20** Use an elliptical mask with softened edges, not a hard rectangle.
  The mask is fully opaque over the whole face box; softening happens only
  outside it
- **FR-21** Never modify the source file; always write a new output file

### Review and correction
- **FR-22** Show detected faces as numbered, selectable overlays
- **FR-23** Toggle redaction per track (selective redaction)
- **FR-24** Allow the user to draw a manual box, which becomes a normal track
  and is propagated by the tracker like any detection
- **FR-25** Allow deletion of a false-positive track
- **FR-26** Preview the redacted result before export

### Export
- **FR-27** Export images preserving original resolution
- **FR-28** Export video preserving original resolution and frame rate
- **FR-29** Optionally mux the original audio stream back using ffmpeg
- **FR-30** Show progress during video processing

## 6. Non-functional requirements

- **NFR-1** Process a 720p image in under 1 second on a mid-range laptop CPU
- **NFR-2** Process 30 seconds of 720p/30fps video in under 3 minutes with
  tracking enabled
- **NFR-3** Zero outbound network requests during processing, verified by a
  test that blocks socket connections for a full pipeline run
- **NFR-4** Peak memory under 2 GB; do not load an entire video into RAM
- **NFR-5** Every error surfaces as a readable message in the UI

## 7. Acceptance criteria by milestone

| Milestone | Done when |
|---|---|
| M1 Images | A photo with 5 faces is redacted correctly in all 3 modes and exported at original resolution |
| M2 Video baseline | A 20 s clip is processed with per-frame detection and exported with audio intact |
| M3 Tracking | The same clip processed with `--detect-interval 10 --tracker lk` has **higher FPS** than M2, and leak frames are measured and recorded for both. Any increase in leak frames is explained in `docs/RESULTS.md` |
| M4 UI | A user can untick person 2 of 4, delete one false positive, draw one manual box, preview, and export, without touching the CLI |
| M5 Evaluation | `eval/run_experiments.py` produces the full results table in `docs/RESULTS.md` |

## 8. Out-of-scope ideas parked for future work

Face recognition for allow-listing known people; voice redaction; batch folder
processing; licence plate redaction; GPU acceleration; reversible encrypted
redaction.
