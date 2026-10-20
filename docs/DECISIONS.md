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

## D-014 — <next>
