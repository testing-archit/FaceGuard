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

## Features added after the core build

**Why is face recognition in a tool that says "no identification"?**
The allow-list is *verification* against faces the user supplies on the
same computer ("is this the person in my reference photo?"), opt-in, with no
names, no database, and embeddings kept only in memory (D-025). It answers a
real need: hide everyone except the people who consented.

**How does SFace decide two faces match?**
Both faces are aligned with YuNet's five landmarks (an affine warp of eyes,
nose and mouth to canonical positions), embedded as 128-d vectors, and
compared by cosine similarity. Same person scored 0.86–0.93 in our tests,
different people at most 0.31.

**Why 0.5 and not OpenCV's 0.363?**
The errors are not symmetric. A false match leaves a stranger visible; a
missed match only hides someone who wanted to be shown. So the threshold
leans towards hiding, and the UI shows each score.

**What do tiling, smoothing and coasting do, and why are they off?**
All three were built, measured, and left off by default because the data
did not justify them: tiling found no extra faces on the annotated photos
and added a false positive at ~5x the cost (D-022); smoothing reduced jitter
by at most ~20% while windows of 5+ frames exposed faces in the whip pan
(D-019); coasting lost tracks at constant velocity overshot when faces
changed direction and made KCF worse (D-023). Negative results are reported
as results.

**Why does the rectangle mask need its corners grown?**
Feathering blurs the mask and doubles it. Along an edge half the blur mass
is inside, so 2x reaches full opacity; at a corner only a quarter is, so the
corner stayed at ~0.98. Growing the rectangle by ~0.6 sigma fixes it (D-020).

**Who labelled the real photos?**
An AI assistant, from detector proposals that were each inspected visually,
with every image checked for missed faces. That is stated in the data file
and in RESULTS.md. Box geometry follows YuNet, which favours YuNet in the
Haar comparison; a human re-check is the first thing to do before quoting
the numbers.

## Features added in the further-development round

**A face walks in between two detections. Is it exposed?**
Not any more (D-026). The pipeline keeps the last 10 frames in memory. When
a face is detected for the first time, its tracker is started on that
detection and run *backwards* through the buffered frames. Optical flow
doesn't care which way time runs: brightness constancy holds from t to t−1
as well as from t−1 to t. On the crossing clip this removed all 8
entry-leak frames. The cost is memory for 10 frames and a writer that runs 10
frames behind.

**How do you know the redaction actually works, not just that a mask was
drawn?**
We attack our own output (E13). YuNet is re-run on the redacted image at
half the normal threshold. SFace then compares each redacted face with the
original, and we check whether its nearest neighbour among all 76 original
faces is the right person. At the old defaults, blur was re-identified 52
times out of 76, which is why the defaults changed. The same check runs on
every export in the UI, and in the CLI with `--verify`.

**Isn't a face that YuNet can still detect a leak?**
Detection means "a face is here", not "this is Alice". Blur at full strength
still lets YuNet find 69 of 76 faces, but SFace matched none of them. That is
why the self-check is described as stricter than identifiability.

**Did the Kalman filter help?**
No (D-030, E14). LK never lost a face on our clips, so there was nothing to
coast. With KCF, moving a lost box with any motion model made leaks *worse*
(0 → 11–12), because the frozen box was already where the face reappeared.
We report it as a negative result. Hungarian association likewise changed
nothing (D-029), which confirms D-009 with data.

**What does re-ID linking do, and why is it opt-in?**
It gives a face that reappears after being fully hidden its old ID back, by
appearance (SFace cosine ≥ 0.5). It removed the one ID switch on the
crossing clip. On the ER clip with KCF it made 74 links, and 1 of them
joined two different people. If the user then made that ID visible, a
stranger would be exposed, so it is off by default and every link is listed.

**Do you read number plates?**
No (D-034). A text *detector* finds where writing is. There is no
recognition step, so no text is ever extracted. The mask is a padded
rectangle, so the corners of a plate stay covered.

**Is the pitch-shifted voice anonymous?**
No (D-033). Shifting the pitch back undoes it. It is offered as a disguise,
and the UI says mute is the only safe option.

**How is reversible redaction secure?**
It uses standard primitives only (D-035): scrypt to turn the passphrase into
a key, and AES-256-GCM so a wrong passphrase or a tampered file fails
loudly. The hash of the exported image is bound into the encryption, so a
vault only opens against the exact file it was made for.

**How did the parallel mode go wrong at first?**
With 8 workers it leaked 43 frames, all inside one chunk. Seeking to frame
2,595 of that file returned a different frame, so every box in the chunk
was shifted in time. Workers now reach their start by decoding forward.
With that fix the results are identical to sequential, and 4 workers are
1.75x faster (D-036).
