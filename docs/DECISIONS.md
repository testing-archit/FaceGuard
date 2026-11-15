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

## D-021 — Review UI built on interactive components, styled as a tool
**Date:** 2026-11-14
**Context:** Coordinate entry (D-015) worked but was slow, and the first UI
looked like a demo (emoji labels, default title, cramped cards).
**Decision:** Streamlit components v2 (bundled, no build step, no network)
give a canvas where a click toggles a person and a drag draws a box, and a
timeline where a click seeks. Coordinate entry stays as fine-tuning of the
drawn box. Visual design is a theme in `.streamlit/config.toml` (separate
light and dark palettes, Streamlit's locally served Source Sans) plus
`ui/style.css`; icons are Material symbols, no emoji.
**Trade-off:** The canvas and timeline are JavaScript, which AppTest cannot
exercise; their Python side (what a click or drag does to state) is tested by
setting the same session state directly, and the JS was checked by hand in
Chrome.

## D-023 — Coasting lost tracks is offered but off by default
**Date:** 2026-11-14
**Context:** When a tracker loses a face, its box is frozen (FR-15). If the
person keeps moving, the frozen box falls behind and the face leaks.
**Decision:** `coast=True` keeps a lost track moving at its last velocity
(EMA of per-frame centre displacement, decaying 10% per frame) and grows the
recorded mask 4% per frame up to 1.8x, while the box used for IoU matching
keeps its size so the detector can still re-anchor it. Off by default.
**Trade-off:** Extrapolation assumes motion continues in a straight line.
**Evidence:** Interval 10, freeze / coast leak frames. synthetic_er: LK 0 / 0,
CSRT 3 / 3, KCF **10 / 23** (rotation 9 → 23). synthetic_cross: LK 8 / 8,
CSRT 8 / 8, KCF 13 / 13. LK never reports a loss on these clips, and where
losses happen (KCF in the rotation segment, both clips' whip pans) the face
changes direction, so the extrapolation overshoots. A negative result: it
stays available for footage with steady motion, such as people walking past
a static camera, where it has not been tested.

## D-022 — Tiled small-face detection is offered but off by default
**Date:** 2026-11-14
**Context:** YuNet's finest feature map has an 8 px stride, so faces of
roughly 10-15 px are below its reliable range (TASKS stretch goal).
**Decision:** `tile_grid = n` also runs the detector on n x n overlapping
tiles upscaled 2x and merges all detections with NMS (IoU 0.3). Exposed as
`--tiles N` and as "Search for small faces" in the UI. Off by default.
**Trade-off:** About 5x the detection time for a 2 x 2 grid.
**Evidence:** Seven annotated public photos (76 faces, see
`data/ground_truth.json` and its provenance note). YuNet alone: 76/76 faces,
0 false positives, 7.3 ms per image. With 2 x 2 tiles: 76/76, **1 false
positive** (a textured background patch in rehg-thanksgiving), 34 ms per
image. class57's 57 faces (about 25-40 px wide) are all found without
tiling. No annotated image yet has faces small enough for tiling to help,
so the option stays off until one does.

## D-024 — Confidence threshold 0.6 confirmed on real photos
**Date:** 2026-11-14
**Context:** The synthetic stills contain no face-like clutter, so they could
not show what a lower threshold costs (E4, first run).
**Evidence:** On the seven public photos: threshold 0.3 → recall 1.000,
2 false positives; 0.5 → 1.000, 1; **0.6 → 1.000, 0**; 0.7 → 1.000, 0;
0.9 → 0.803, 0. 0.6 is the lowest threshold with no false positives, so a
lower value buys nothing on this set. Caveat: the boxes were proposed by
YuNet and verified by an AI annotator, and the photos are easy (frontal,
well separated), so a harder set may move the optimum.

## D-025 — Allow-list by face verification (stretch goal), opt-in and local
**Date:** 2026-11-15
**Context:** Users often need "hide everyone except the people who agreed".
CLAUDE.md permits an embedding model only once Weeks 1–4 are complete and the
Week 5 stretch goals are reached; both now hold.
**Options:** none (manual unticking only); SFace embeddings from OpenCV Zoo.
**Decision:** SFace (`face_recognition_sface_2021dec.onnx`, Apache 2.0,
pinned commit and SHA-256, 38.7 MB, not committed; fetched by
`download_model.py --with-allowlist`). Faces are aligned with YuNet's five
landmarks, embedded, and compared by cosine similarity with reference faces
the user supplies (uploaded photos, or "keep visible everywhere" on a
detected person). Each video track is embedded from its three largest
detector-confirmed boxes, averaged. Matching tracks are left visible; the UI
shows every match score. It is verification against user-supplied
references, not identification: no names, no database, embeddings only in
memory, none written to the redaction log.
**Trade-off:** A false match exposes a stranger, so the default threshold
is 0.5, stricter than OpenCV's verification default of 0.363; this costs
some missed matches (a wanted person stays hidden), which is the safe error.
**Evidence:** addams-family, 7 people: same-person similarity after mirroring
and 0.8x rescaling 0.86–0.92, different people at most 0.31. Crossing clip:
the reference for person A matched both of A's tracks (before and after full
occlusion) at 0.93 and 0.92; all non-matching tracks at most 0.18.

## D-026 — Backfill new faces by tracking backwards, on by default
**Date:** 2026-11-15
**Context:** A face that enters between two scheduled detections stays
visible until the next detection runs: 8 leaked frames on the crossing clip
at N = 10 (REPORT section 8, "entry exposure"). Detecting more often only
shrinks the gap and costs FPS.
**Options:** detect every frame near the frame edges; detect more often;
keep the last few frames in memory and, when a face is first detected, run
the tracker backwards over them.
**Decision:** Backfill. The pipeline holds the last `backfill_frames`
(default 10, matching the default detection interval) frames; each new
track is initialised on its first detection and tracked newest-to-oldest
until the tracker loses it, less than half the box is inside the frame, a
scene cut is reached, or the track already has a box. The single-pass CLI
writer runs that many frames behind the analysis, so the output video
contains the backfilled masks without a second decode. Backfilled boxes are
flagged "backfill" for review (amber on the UI timeline).
**Trade-off:** Memory for N decoded frames (about 60 MB at 1080p for
N = 10), a little FPS, and a slightly lower box precision: backfilled boxes
follow a face slightly before the ground truth counts it as visible.
**Evidence:** E12. Crossing clip at N = 10: entry leak frames 8 → 0 with 18
frames filled; precision 0.944 → 0.935. The ER clip has no late entries and
is unchanged (0 frames filled).

## D-027 — Stronger blur and pixelation defaults, set by attacking the output
**Date:** 2026-11-15
**Context:** Coverage only says a mask was applied, not that it works. E13
attacks the exported photos: YuNet is re-run on the output, and SFace
compares each redacted face with its original, both as a same-person test
(cosine > 0.363) and as a closed-set re-identification against all 76
original faces (rank-1; chance is about 1 in 76).
**Options:** keep the old defaults (blur 0.35 of face width, pixelation 8
blocks) and only warn; strengthen the defaults.
**Decision:** Blur kernel = 1.0 x face width, pixelation = 5 blocks across
the padded face. Fill is unchanged and remains the recommended mode.
**Trade-off:** Coarser-looking output; blur at full strength halves render
throughput on the ER clip (redaction 1.75 s → 3.47 s, 95.8 → 78.0 FPS).
Detection of "a face is here" is not removed by blur at any strength
(69/76 still detectable) — that reveals presence, not identity.
**Evidence:** E13 on the 7 real photos (76 faces). Old blur: 43 faces
matchable and 52 re-identified at rank 1. Blur at 1.0: 0 and 0. Old
pixelation: 1 and 3, with 33 still detectable at 0.3. Five blocks: 0 and 0,
3 detectable. Fill: 0 and 1 (chance level), 2 detectable at score < 0.42
(neighbouring faces whose centre falls inside a padded box).

## D-028 — A self-check re-scans every export with YuNet at 0.3
**Date:** 2026-11-15
**Context:** Users need evidence for a specific file, not average results.
**Decision:** `core/verify.py` re-runs YuNet on the output at
`verify_threshold` = 0.3 (half the redaction threshold, so the checker is
stronger than the redactor) and reports any face whose centre falls inside a
region that should be hidden. The UI runs it after every export and records
the result in the redaction log; the CLI runs it with `--verify`. It always
uses YuNet, even when Haar did the redaction.
**Trade-off:** One extra decode and detector pass per export (about the cost
of per-frame detection). It is a detector, not a recogniser, so a pass means
"no detectable face", which E13 shows is stricter than "not identifiable".
**Evidence:** default exports of both synthetic clips: 0 residual detections
on the crossing clip; on the ER clip 160 of 3,600 hidden face-frames are
still detectable at 5 blocks (416 at 6, 1,355 at the old 8).

## D-029 — Hungarian association offered, greedy stays the default
**Date:** 2026-11-15
**Context:** D-009 chose greedy IoU matching and argued it almost always
agrees with optimal assignment for a handful of faces. That claim is now
tested rather than assumed.
**Decision:** `association = "hungarian"` runs Kuhn-Munkres with dual
potentials (pure NumPy, no SciPy dependency) on cost 1 − IoU; pairs below
the IoU threshold get a cost no real pair can reach and are dropped after
assignment. Greedy remains the default.
**Trade-off:** O(n³) instead of O(n² log n); irrelevant at these sizes.
**Evidence:** E14: identical leak frames, precision, track counts and ID
switches to greedy on both clips with LK and KCF. A unit test shows the
textbook case where greedy strands a track and Hungarian does not; it does
not occur in our data.

## D-030 — Kalman coasting offered, freeze stays the default
**Date:** 2026-11-15
**Context:** D-023's constant-velocity coasting overshot when faces turned.
A Kalman filter is the standard fix: it estimates velocity with measured
noise models instead of a hand-tuned decay, and its covariance says how far
the face may have gone.
**Decision:** `coast_model = "kalman"` (state centre + velocity,
white-noise acceleration). While lost, the box follows the prediction and
the mask grows by 2 position standard deviations per side, capped at
`coast_max_scale`. Off by default, like all coasting.
**Trade-off:** none in cost; it simply does not help here.
**Evidence:** E14. LK never loses a face on either clip, so every motion
model is identical. With KCF on the ER clip: freeze 0 leak frames,
velocity coasting 11, Kalman 12. Moving a lost box changes which detection
it is matched to next, and the frozen box was already where the face
reappeared. Negative result, kept for the report.

## D-031 — Re-ID linking by appearance, opt-in
**Date:** 2026-11-15
**Context:** A fully occluded face comes back with a new track ID (REPORT
section 8), so a per-person choice silently stops applying to it.
**Decision:** With `reid_link`, each track's SFace embedding (the same
collector as the allow-list) is compared with tracks that ended before it
began; at cosine ≥ 0.5 within 150 frames the later track is merged into
the earlier. A track ends at its last detector-confirmed frame: an occluded
face's LK track is dragged along by the occluder for a few frames, and
those flagged guesses overlap the new track. Merging keeps a track hidden
if either part was hidden. Links are logged as `reid_link` events.
**Trade-off:** One SFace embedding per confirmed face (a few ms each).
A wrong link joins two people under one ID, so making that ID visible
would expose both; this is why it is opt-in and the links are listed.
**Evidence:** Crossing clip: the one ID switch is removed with both LK and
KCF (similarity 0.96), coverage unchanged. ER clip with KCF: 74 links
(116 → 42 tracks), 1 of them joining two different people (similarity
0.61; all ER faces are cut from one group photo, an unusually hard case).

## D-032 — Haar profile-face fallback, opt-in
**Date:** 2026-11-15
**Context:** YuNet misses heads turned past about 60°. OpenCV ships a
profile cascade (trained on one direction).
**Decision:** `profile = true` adds profile-cascade hits on the frame and
its mirror, scored 0.5 so that YuNet wins every overlap in NMS. The cascade
is committed in `models/` (828 KB, pinned and hashed like the others).
**Trade-off:** About 4x detection time per image and extra false
positives; profile hits have no landmarks, so they never feed the
allow-list.
**Evidence:** E15. Our annotated data contains no profile faces, so the
fallback cannot show a gain here: real photos 76/76 either way; synthetic
stills gain 0 faces and 6 false positives (minNeighbors 6; 27 at 4, 3 at
8). Evaluating it needs a profile set (EVALUATION.md section 1 asks for
one), so it stays off.
