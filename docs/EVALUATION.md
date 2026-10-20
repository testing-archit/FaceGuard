# FaceGuard — Evaluation Protocol

This is what turns the project from a tutorial follow-along into something
defensible. Code that works is the baseline expectation; measured comparisons
are the contribution.

## 1. Test dataset

Build it yourself. It need not be large, it must be **labelled**.

| Set | Size | Content |
|---|---|---|
| Images | 30–50 | 1 face, 2–5 faces, crowd (10+), low light, profile views, partial occlusion, small faces |
| Videos | 3–5 clips, 10–30 s each | Static camera / moving subject, moving camera, two people crossing (ID-switch case), low light, fast motion |

Annotate every face with a ground-truth box. For video, annotate every 5th frame
and interpolate linearly between keyframes. Budget 3–5 hours. Start in Week 3.

Record consent for anyone identifiable in your test data, and say so in the
report. Using a privacy tool to build an unconsented face dataset is an easy
thing for an examiner to point out.

## 2. Ground-truth format (`data/ground_truth.json`)

```json
{
  "images": [
    {
      "file": "samples/crowd_01.jpg",
      "faces": [
        {"id": 0, "box": [120, 80, 64, 72]},
        {"id": 1, "box": [310, 95, 58, 66]}
      ]
    }
  ],
  "videos": [
    {
      "file": "samples/campus_walk.mp4",
      "fps": 30,
      "frames": {
        "0":  [{"id": 0, "box": [200, 150, 80, 90]}],
        "5":  [{"id": 0, "box": [206, 152, 80, 90]}]
      },
      "interpolate": true
    }
  ]
}
```

Boxes are `[x, y, w, h]` in pixels. `id` is a person identity, stable across
frames, used for the ID-consistency metric.

## 3. Metrics

### 3.1 Detection precision / recall / F1

A prediction matches a ground-truth box if IoU >= 0.5.

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

**Recall matters more than precision here.** A false positive blurs a doorknob,
which is ugly. A false negative exposes a real person. Say this explicitly in
the report and in the viva — it shows the metric choice was reasoned about
rather than copied.

### 3.2 Redaction coverage (headline metric)

For each ground-truth face box G and the union M of all applied masks:

```
coverage(G) = area(G ∩ M) / area(G)
```

A pixel counts as masked when the mask alpha is >= 0.5. Report the mean
across all faces and all frames. This is more honest than
detection recall because it accounts for box padding and mask shape, which is
what actually determines whether a person is hidden.

### 3.3 Leak frames (the one that really matters)

```
leak_frames = count of frames containing at least one ground-truth face
              with coverage < 0.90
```

**Report this as a raw count, never only as a percentage.** 99% coverage over a
900-frame clip still means 9 frames where someone is fully identifiable, and one
such frame defeats the entire tool. This framing is the strongest single point
in the report.

### 3.4 Throughput

Frames per second, end to end including IO and redaction, averaged over 3 runs
on the same machine. Record CPU model, RAM and OpenCV version alongside the
numbers.

### 3.5 ID consistency

```
id_consistency = frames where a track keeps the correct person identity
                 / total frames that track exists
```

Each predicted track is assigned to the ground-truth identity it overlaps
(IoU >= 0.5) in the most frames. A frame where the track overlaps a *different*
ground-truth identity counts against it.

Also report the raw count of detected ID switches.

## 4. Experiments

### E1 — Classical vs neural detection
YuNet vs Haar cascade, across the whole image set and all clips.
Report: precision, recall, F1, FPS, and a per-condition breakdown.
Expected finding: Haar is competitive only on frontal, well-lit, large faces and
degrades sharply elsewhere. Prove it with the condition breakdown, do not assert
it.

### E2 — Detection interval sweep (the core experiment)
`detect_interval ∈ {1, 5, 10, 15, 30}` with LK tracking.
Report: FPS and leak frames at each setting.
This produces your headline sentence:

> "Detecting every 10 frames with Lucas-Kanade interpolation achieved **X×** the
> throughput of per-frame detection while changing leak frames from **A** to
> **B**."

Note that leak frames can go **down** with tracking even though the detector
runs less often, because tracking covers frames the detector would have missed.
If your data shows that, it is the single most interesting result in the project.

### E3 — Tracker comparison
LK vs CSRT vs KCF vs none, at a fixed interval of 10.
Report: FPS, coverage, leak frames, ID consistency, and frames until the track
is lost.

### E4 — Confidence threshold sweep
`conf_threshold ∈ {0.3, 0.5, 0.6, 0.7, 0.9}`.
Plot a precision-recall curve. Justify the shipped default from this curve,
weighted toward recall.

### E5 — Condition breakdown
Group results by: lighting (normal / low), face size (large >100 px / medium /
small <40 px), motion (static / moving), occlusion (clear / partial), pose
(frontal / profile). Report recall and coverage per group. This table tells the
reader exactly when the tool should not be trusted.

### E6 — Padding sweep
`padding ∈ {0.0, 0.1, 0.2, 0.25, 0.3}` against coverage. Geometry predicts a
step: an ellipse inscribed in the padded box only contains the face box's
corners once padding >= (√2 − 1)/2 ≈ 0.207 per side. Below that, coverage of
a perfectly detected face is capped (π/4 ≈ 78.5% at padding 0). Justifies the default and
shows you understood that detector boxes are tighter than the region that
actually needs hiding.

## 5. Outputs

`eval/run_experiments.py` must regenerate everything from one command:

- `docs/RESULTS.md` with all tables
- `eval/out/fps_vs_interval.png`
- `eval/out/leakframes_vs_interval.png`
- `eval/out/pr_curve.png`
- `eval/out/condition_breakdown.png`
- `eval/out/raw_results.json`

## 6. Reporting rules

- State the hardware, OpenCV version and model filename with every timing number
- Average timings over 3 runs; report the spread
- Report failures. A limitations section with honest numbers reads as stronger
  work than a results section with suspiciously clean ones
- No claim in the report without a number behind it
- Include at least one screenshot of a failure case with an explanation
