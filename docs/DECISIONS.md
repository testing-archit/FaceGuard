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
**Evidence:** ~338 KB model, WIDER Face validation around 0.83 easy / 0.82
medium, CPU inference viable. Haar retained as the classical baseline for E1.

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

## D-004 — <next>
