# Design Decisions

One entry per non-obvious choice. These become report paragraphs, so write them
in full sentences, not notes.

Format:

## D-00x — <decision>
**Date:** YYYY-MM-DD
**Context:** what problem forced a choice
**Options:** what was considered
**Decision:** what was chosen
**Trade-off:** what was given up
**Evidence:** the measurement, if any

---

## D-001 — YuNet as the primary detector
**Date:** 
**Context:** Need a face detector that runs on laptop CPU with no training and
no GPU.
**Options:** Haar cascade, OpenCV DNN SSD (Caffe), YuNet, MTCNN, RetinaFace.
**Decision:** YuNet through `cv2.FaceDetectorYN`.
**Trade-off:** Weaker than heavyweight detectors on extreme pose and tiny faces.
**Evidence:** 229,738-byte model; the OpenCV Zoo README reports 0.834 / 0.824 /
0.708 AP on WIDER Face validation easy / medium / hard. CPU inference viable. Haar retained as the classical baseline for E1.

## D-002 — Detect every N frames with tracking in between
**Date:** 
**Context:** Per-frame detection is slow and flickers; a dropped frame leaks a
face.
**Options:** detect every frame; detect + track; track only.
**Decision:** Detect every N frames (default 10), propagate with pyramidal LK
optical flow, re-anchor tracks at every detection.
**Trade-off:** Tracking drift between detections; a new face entering the scene
is not covered until the next detection frame.
**Evidence:** See E2 in docs/EVALUATION.md.

## D-003 — Pixelation as the default redaction mode
**Date:** 
**Context:** Gaussian blur is the most common choice but is partially
reversible by deblurring and face-restoration models.
**Decision:** Default to pixelation; offer solid fill as the strongest option
and surface the ranking in the UI.
**Trade-off:** Pixelation is visually harsher than blur.

## D-004 — `face_detection_yunet_2026may.onnx` over `2023mar`
**Date:** 2026-10-20
**Context:** OpenCV 5.0 runs ONNX models through its new engine, which needs
dynamic input dims to accept arbitrary frame sizes.
**Options:** `2023mar` (fixed input shape), `2023mar_int8` (quantised),
`2026may` (same weights, dynamic H/W).
**Decision:** `2026may`, pinned to opencv_zoo commit `26cc381e` and checked by
SHA-256.
**Trade-off:** None in accuracy (same weights); int8 would be slightly faster
but slightly less accurate, and speed is not the bottleneck.

## D-005 — Grace period counted in frames, not detector runs
**Date:** 2026-10-20
**Context:** Counting misses per detector run makes the grace period scale with
`detect_interval` (15 runs x 10 frames = 150 frames), which would silently bias
the interval sweep in E2.
**Decision:** A track is deleted when `frame - last_seen_frame > max_misses`.
**Trade-off:** Deletion is still only checked on detection frames, so the
effective grace period is rounded up to the next detection.

## D-006 — Padding 0.25 per side
**Date:** 2026-10-20
**Context:** The mask is an ellipse inscribed in the padded box. It only
contains the face box's corners when padding >= (√2 − 1)/2 ≈ 0.207 per side.
**Decision:** Default padding 0.25 per side.
**Trade-off:** Slightly more of the background is masked.
**Evidence:** E6 padding sweep.

## D-007 — Feather outward, not inward
**Date:** 2026-10-20
**Context:** Blurring a hard mask makes its edge semi-transparent on both sides,
leaking original pixels just inside the ellipse. In fill mode that defeats the
point of fill.
**Decision:** `alpha = clip(2 * blur(mask), 0, 1)`: fully opaque inside the
ellipse, soft ramp only outside it.
**Trade-off:** The masked region is a few pixels larger than the ellipse.

## D-008 — Track before associating
**Date:** 2026-10-20
**Context:** Matching detections against boxes from the previous detection
frame fails IoU on fast motion, breaking tracks and inflating ID switches.
**Decision:** Every tracker updates on every frame, including detection frames,
before association runs.
**Trade-off:** One extra tracker update per detection frame.

## D-009 — Greedy association instead of Hungarian
**Date:** 2026-10-20
**Context:** Hungarian assignment needs `scipy`.
**Decision:** Greedy highest-IoU-first matching.
**Trade-off:** Can be suboptimal when many boxes overlap; with the face counts
in this project the two almost always agree, and it avoids a dependency.

## D-010 — Separate analyse and render passes
**Date:** 2026-10-20
**Context:** The UI must render exactly the track IDs the user toggled.
Re-running detection in the render pass is not guaranteed to reproduce them.
**Decision:** `analyse` stores boxes per track per frame; `render` only replays
them.
**Trade-off:** The video is decoded twice.

## D-011 — Re-encode output to H.264 with ffmpeg
**Date:** 2026-10-20
**Context:** `cv2.VideoWriter`'s `mp4v` output does not play in browsers, so the
Streamlit preview would be blank.
**Decision:** Always finalise through ffmpeg to H.264 / yuv420p; fall back to
the `mp4v` file with a warning when ffmpeg is absent.
**Trade-off:** One extra encode per export.

## D-012 — Separate drift cap for tracks the tracker still follows
**Date:** 2026-10-20
**Context:** With one grace period (15 frames since the last detector match),
tracks were deleted while LK and CSRT were still following the face. In the
synthetic clip's rotation segment YuNet misses faces rolled ~40°; four tracks
were deleted at frames 279–289 and those faces were exposed until the next
detection at frame 300.
**Options:** longer single grace period; separate limits for followed vs lost
tracks; never delete followed tracks.
**Decision:** A followed track survives `max_unconfirmed` = 60 frames (2 s at
30 fps) without a detector match; a lost track keeps its frozen box for
`max_misses` = 15 frames from the moment it was lost. "none" has no motion
evidence, so it always uses the short limit.
**Trade-off:** A drifting track can mask the wrong region for up to 2 s. For a
redaction tool this costs precision, while early deletion costs recall.
**Evidence:** Synthetic clip, interval 10: LK leak frames 35 → 15 (rotation
20 → 0), CSRT 23 → 3. LK tracks 12 → 8 for 6 faces.

## D-013 — Four LK pyramid levels instead of three
**Date:** 2026-10-20
**Context:** After D-012, LK's remaining 15 leak frames were one face lost
during the whip pan once motion passed ~43 px/frame. The frozen box then
missed the face and the next detection created a new ID.
**Options:** `lk_pyr_levels` 3/4/5, `lk_win_size` 21/31.
**Decision:** 4 levels, window 21.
**Trade-off:** Negligible: each extra level is a quarter the size of the last.
**Evidence:** Synthetic clip, interval 10. Leak frames / tracks / FPS:
3 levels 15 / 8 / 81.8; 4 levels **0 / 6 / 79.7**; 5 levels 0 / 6 / 78.5;
window 31 changes nothing. The coarsest level of a 4-level pyramid sees a
60 px motion as under 4 px, inside LK's small-motion assumption. Tuned on the
synthetic clip, so it must be confirmed on the real clips (E2/E3).



## D-014 — LK runs on a search window around each face
**Date:** 2026-11-14
**Context:** The first M3 measurement had LK at interval 10 at 66.5 FPS,
below the 76.7 FPS per-frame baseline. Profiling: 6.6 s of 7.4 s was tracker
time. Every `calcOpticalFlowPyrLK` call builds pyramids of both full frames,
so 6 faces with a forward-backward check meant 24 full-frame pyramids per
frame, and corner re-seeding computed the corner response over the whole
frame under a mask.
**Options:** share one pyramid per frame (OpenCV 5's Python binding rejects
prebuilt pyramids); batch all tracks into one LK call (breaks the per-track
`Tracker` interface); crop both frames to the box plus a margin.
**Decision:** Crop to the box grown by `lk_search_frac` (1.0) x box size +
`lk_search_min_px` (128) on each side; seed corners on the box crop with an
8 px context border. Points that leave the crop are discarded.
**Trade-off:** The margin is a hard cap on per-frame motion. A small crop
also has fewer usable pyramid levels, which is why the margin needs a large
fixed floor, not just a fraction of a ~45 px face.
**Evidence:** Synthetic clip, interval 10 (analyse only). Margin floor /
FPS / leak frames / tracks: full frame 96.8 / 0 / 6; 48 px 216.2 / 15 / 9;
96 px 182.7 / 3 / 7; **128 px 162.4 / 0 / 6**. End to end with redaction and
encode: 121.5 FPS vs 76.7 per-frame (docs/RESULTS.md, M3).

## D-015 — Manual boxes by coordinate entry with live preview, not a drawing canvas
**Date:** 2026-11-14
**Context:** FR-24 needs the user to place a box over a missed face.
Streamlit has no built-in way to draw on an image.
**Options:** `streamlit-drawable-canvas` (third-party component, unmaintained
for recent Streamlit versions); Plotly box-select (adds plotly); number
inputs with the draft box drawn live on the preview.
**Decision:** Number inputs (x, y, width, height), drawn in magenta on the
preview as they change. In video the box is then followed by the configured
tracker from the current frame (as FR-24 requires), held still to an end
frame, or interpolated linearly to a second keyframe box.
**Trade-off:** Slower to place than dragging a rectangle. No new dependency,
and the result is pixel-exact and testable headless.

## D-016 — Track editing happens on the stored analysis, never by re-detecting
**Date:** 2026-11-14
**Context:** Changing the redaction mode, padding or strength must not undo
the user's ticks, deletions and manual boxes.
**Decision:** The UI splits settings in two. Redaction settings only affect
rendering and apply instantly. Detection settings (detector, threshold,
tracker, interval) mark the analysis stale and need an explicit re-run, which
warns that it discards manual edits.
**Trade-off:** One extra click after changing detection settings.

## D-017 — Run the detector early when tracker and detector disagree
**Date:** 2026-11-14
**Context:** On the crossing clip (`scripts/make_crossing_clip.py`), LK at
interval 10 leaked 13 frames: 5 when a face re-emerged from behind another
and 8 when a new face walked in. Tracing the boxes showed the occluded face's
track was never "lost": its corner points latched onto the shoulder of the
person passing in front and followed them. When the real face reappeared it
waited for the next scheduled detection. The course proposal also asks for
immediate detection when tracking is weak.
**Options:** (a) detect early when a tracker returns None; (b) detect early
while two face boxes overlap; (c) detect early while any track was not
matched by the last detection (tracker and detector disagree).
**Decision:** (a) and (c). (b) was implemented and dropped: the person in
front hides the face with their head and shoulders before the two face
boxes overlap, so it never fired. (c) runs the detector every
`unconfirmed_interval` = 3 frames until the disagreement resolves.
**Trade-off:** Extra detector calls exactly where tracking is uncertain
(rotated faces, occlusion). Entry of a brand-new face is not helped: no
track exists to disagree with, so it still waits for the next scheduled
detection.
**Evidence:** analyse-only FPS / detector calls / leak frames at interval 10.
Crossing: scheduled only 285 / 45 / 13; unconfirmed every 1: 201 / 91 / 8;
**every 3: 243 / 61 / 8** (crossing leaks 5 → 0, the 8 left are the
entering face). First clip: 172 / 60 / 0 → 168 / 72 / 0, no change in
leaks. Full table: E7 in docs/RESULTS.md.

## D-018 — Scene cuts need a change in both histogram and layout
**Date:** 2026-11-14
**Context:** Tracks must not continue across a cut: optical flow between two
different shots is meaningless and a track ID is a motion track, not an
identity. A grey-histogram correlation test alone fired 7 times on the first
synthetic clip, all in its low-light fade, splitting 6 people into 48 IDs.
**Decision:** A cut needs histogram correlation < 0.5 **and** normalised
cross-correlation of a 32 px wide zero-mean, unit-variance thumbnail < 0.3.
A fade changes the histogram but not the normalised layout; a whip pan
changes the layout but not the histogram; a cut changes both. On a cut every
live track is ended and logged, and the detector runs on that frame.
**Trade-off:** A cut between two shots with similar lighting and similar
composition is missed and handled as fast motion instead.
**Evidence:** 0 false cuts on both synthetic clips (was 7). Unit tests cover
fade, pan and a real cut. No real footage with cuts has been annotated yet.

## D-019 — Temporal smoothing is offered but off by default
**Date:** 2026-11-14
**Context:** Detector boxes wobble and re-anchoring makes tracks jump, so
masks shimmer (TASKS stretch goal). Analysis stores every box before
rendering, so a zero-phase centred moving average is possible.
**Decision:** `smooth_window` (centred moving average of box centre and size,
per contiguous run, window shrinking symmetrically at run ends) is available
in the CLI (`--smooth`) and UI, default 1 (off). Manual tracks are never
smoothed.
**Trade-off:** Smoothing low-passes real motion as well as noise: at a sharp
change of direction the smoothed box lags the face.
**Evidence:** LK, interval 10. Window / jitter (mean |second difference| of box
centre, px) / leak frames on synthetic_er: 1 / 1.02 / 0; 3 / 0.95 / 0;
5 / 0.97 / 5; 9 / 0.91 / 29; 15 / 0.80 / 57, all new leaks in the whip pan.
synthetic_cross: jitter 1.00 → 0.83 at window 3, leaks unchanged (8). Jitter
falls by at most ~20% while every window ≥ 5 costs coverage, so the default
stays off. Window 3 is the only setting that is free on both clips.

## D-020 — Rectangle masks as a "maximum coverage" option
**Date:** 2026-11-14
**Context:** The ellipse needs padding ≥ 0.207 to contain the face box
corners (D-006). Users who want everything hidden regardless of box
tightness need a shape with no such condition.
**Decision:** `mask_shape = "rect"` masks the whole padded box. It is still
feathered outward, and the hard rectangle is grown by ~0.6 blur sigma
because a blurred corner keeps only a quarter of its mass, so `clip(2·blur)`
alone left corners at alpha ≈ 0.98.
**Trade-off:** More background hidden; the boxy look is more conspicuous.
