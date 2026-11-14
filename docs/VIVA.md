# FaceGuard — Viva Preparation

Short answers to the questions an examiner is most likely to ask, each tied
to where the evidence lives. Numbers are quoted from `docs/RESULTS.md`; if
you re-run the experiments, re-check them here.

## The problem and the claim

**What does FaceGuard do, in one sentence?**
It finds faces in a photo or short video, lets the user choose who to hide
and fix misses, and exports a redacted copy, entirely offline on a laptop CPU.

**What is the contribution, given the detector is pretrained?**
The detector is OpenCV Zoo's YuNet, used as-is. The contribution is the
workflow around it: the detect-then-track policy and its refresh rules, the
association and track lifecycle, the mask geometry, review and correction,
and above all the evaluation that measures whether faces actually stay
covered (proposal section 11).

**Why is "leak frames" the headline metric and not accuracy?**
A redaction tool fails if a face is visible in even one frame. A mean
coverage of 99% over 900 frames can still hide 9 frames where someone is
fully identifiable. Leak frames counts frames with any face under 90%
coverage, as a raw number (EVALUATION 3.3).

**Why weight recall over precision?**
A false positive blurs a doorknob; a false negative exposes a person.

## Detection (Module I, Module IV classification)

**How does YuNet work?**
A small anchor-free CNN (about 230 KB) with a multi-scale feature pyramid.
For each location it regresses a box, five landmarks and a confidence score;
OpenCV's `FaceDetectorYN` handles preprocessing, decoding and non-maximum
suppression.

**Why must `setInputSize` be called?**
YuNet decodes boxes relative to its input size. A stale size returns
coordinates scaled for the wrong resolution, silently.

**How does the Haar cascade work and why keep it?**
Viola-Jones: Haar-like rectangle features computed on an integral image,
combined by a boosted cascade that rejects most windows early, scanned over
an image pyramid. It is the classical baseline for E1. On our data it was
both less accurate and slower than YuNet on this CPU (see E1), which
contradicts the usual "classical is faster" assumption.

**Is the confidence score a probability?**
No. It is a model output, not a calibrated probability. The threshold is
chosen from the precision-recall curve (E4), not interpreted as a chance.

## Tracking (Modules III and IV)

**Why track at all if detection is fast?**
Two reasons, both measured. Speed: detecting every 10 frames is faster
(E2). Coverage: per-frame detection has no memory, so every frame where the
detector misses a face (rolled ~40°) is a leaked frame. Tracking carries the
box through those frames. On the first synthetic clip leak frames went from
43 to 0 (E2).

**Explain Lucas-Kanade.**
For each point, solve for the displacement that best explains the intensity
change in a small window, assuming (1) brightness constancy, (2) small motion
between frames, (3) all pixels in the window move together. The 2x2 system
is the structure tensor; it is solvable where the tensor has two large
eigenvalues.

**Why Shi-Tomasi corners?**
They are exactly the points where the structure tensor's smaller eigenvalue
is large, so LK's system is well conditioned there (no aperture problem).

**What does the pyramid do?**
It relaxes the small-motion assumption: a 60 px motion is under 4 px at the
top of a 4-level pyramid. Three levels lost a face in the whip pan; four did
not (D-013).

**What is the forward-backward check?**
Track points forward, then back again; a correct correspondence returns to
where it started. Points that miss by more than 1 px are discarded. It
removes points on occlusion boundaries and specular changes.

**How is the box updated?**
Translation is the median point displacement; scale is the median ratio of
pairwise point distances. Medians tolerate the outliers that survive.

**Why did the first LK version run slower than per-frame detection?**
Each call to `calcOpticalFlowPyrLK` builds pyramids of both full frames, so
6 faces with a forward-backward check meant 24 full-frame pyramids per frame.
LK now runs on a crop of each face plus a search margin (D-014). The margin
needed a large fixed floor (128 px) because a small crop also loses pyramid
levels.

**CSRT and KCF?**
Correlation-filter trackers from OpenCV contrib. KCF learns a filter on HOG
features using circulant structure (fast, fixed scale). CSRT adds a spatial
reliability map and channel weights (handles scale, much slower). See E3.

## Association and refresh

**How are detections matched to tracks?**
Greedy highest-IoU-first matching at IoU 0.3, after every tracker has been
updated to the current frame (D-008, D-009). Matched tracks are re-anchored
to the detection, which resets drift.

**What happens to a track the detector misses?**
If the tracker still follows it, it is kept for up to 60 frames without a
detector match; if the tracker lost it, its frozen box is kept for 15
frames. Both are redacted throughout (D-012).

**When does the detector run off schedule?**
After a lost track, and every 3 frames while a tracked face went unmatched by
the last detection (D-017). The second rule was found by tracing a leak: on
the crossing clip, the hidden face's points latched onto the shoulder of the
person passing in front, so its track followed the wrong person without ever
being lost. Rule (b), detecting while face boxes overlap, was implemented and
dropped because it never fired: the occluder's head and shoulders hide the
face before the face boxes meet.

**What still leaks?**
A face entering the frame between scheduled detections has no track, so it
is exposed until the next detection (8 frames on the crossing clip at
interval 10). That is the honest cost of interval detection, and E2 shows it
shrinking as the interval falls.

**What is a track ID?**
A continuous motion track, not an identity. IDs reset at scene cuts and a
person who leaves and re-enters gets a new ID. There is no face recognition.

**How are scene cuts detected?**
Both a grey-histogram correlation drop and a drop in normalised
cross-correlation of a 32 px thumbnail (D-018). A fade changes only the
histogram; a whip pan changes only the layout; a cut changes both.

## Redaction (Module IV colour spaces, segmentation-related operations)

**Why an ellipse, and why padding 0.25?**
An ellipse inscribed in the padded box contains the face box's corners only
when padding is at least (√2 − 1)/2 ≈ 0.207 per side. At padding 0 the
coverage of a perfectly detected face is capped near π/4. E6 shows the step.

**Why feather outward?**
Blurring a hard mask puts alpha 0.5 on the edge, leaking original pixels
just inside the ellipse. `alpha = clip(2 · blur(mask), 0, 1)` keeps the
inside fully opaque and puts the ramp outside (D-007).

**Is this segmentation?**
No. It is region masking: an elliptical binary mask per box. No pixel is
classified as face or not-face.

**Which mode is safest?**
Solid fill destroys the information. Pixelation can be attacked given a
small set of candidate faces. Blur is weakest: deblurring and
face-restoration models can partially recover it. The UI says so.

**Colour spaces?**
OpenCV frames are BGR. Grayscale is computed once per frame and shared by
LK, Haar and scene-cut detection. Conversion to RGB happens only at the
Streamlit boundary.

## Evaluation honesty

**Where does the ground truth come from?**
Two synthetic clips generated from one public group photo, so every box in
every frame is known exactly, plus stills sampled from the first clip. No
hand-annotated real footage is in the results yet; `docs/RESULTS.md` says so
at the top. Real annotation uses `eval/annotate.py`, by hand, never seeded
from the detector under test.

**What are the limits of synthetic data?**
The faces are real photographs, but motion is 2D affine, there is no motion
blur, no out-of-plane head rotation, and only 6 distinct people. Results show
mechanism (why a leak happens), not field accuracy.

**Hardware?**
Recorded with every timing in `docs/RESULTS.md`. The development machine is
an Apple M5, faster than the "mid-range laptop" target, so absolute FPS
should be re-measured on the demo laptop; ratios between configurations are
what the report relies on.
