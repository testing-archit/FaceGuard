# FaceGuard — Demo Script

About 6 minutes. Everything runs offline: **turn Wi-Fi off before you start**
to prove it (TASKS demo checklist).

## Before the day

1. On the demo laptop: double-click `FaceGuard.command` / `FaceGuard.bat` once
   with internet, so components install. Then test again with Wi-Fi off.
2. `python scripts/fetch_samples.py`, `python scripts/make_synthetic_clip.py`,
   `python scripts/make_crossing_clip.py`, `python scripts/make_demo.py`.
3. Re-run `python eval/run_experiments.py` on the demo laptop so the numbers
   you quote are from that machine (`docs/RESULTS.md` records the CPU).
4. Take two UI screenshots for the report (Review screen with the timeline;
   Export screen) and save them in `docs/figures/`.
5. Optional: `python scripts/download_model.py --with-allowlist` for step 5.

Do not record anything live. Every input is a file prepared in advance.

## Running order

| # | Show | Say |
|---|---|---|
| 1 | Turn Wi-Fi off. Double-click the launcher. | "No account, no upload, no network. Detection, tracking and redaction all run on this CPU." |
| 2 | Upload `addams-family.png`, **Detect faces**. Click the boy's face so he turns amber (visible). Drag a box over the bear rug. **Export**. | "A pretrained CNN, YuNet, finds every face with a confidence score. One click leaves someone visible; a drag covers anything the detector missed." Point at `docs/figures/selective.png` if time is short. |
| 3 | Upload `synthetic_er.mp4`, **Detect faces**. Scrub to ~9.5 s (frame 281). Toggle **Preview**. | "Faces here are rolled 40°. The detector misses them, but Lucas-Kanade optical flow keeps following the corner points." Show `docs/figures/failure_rotation.png`: per-frame detection exposes 5 faces on this frame, tracking exposes 0. |
| 4 | Click the **timeline** near the red marks; press **Next** (flag). | "Red marks are frames where a box is a tracker estimate the detector did not confirm. The tool tells you where to look instead of claiming it is perfect." |
| 5 | (Optional) Upload `synthetic_cross.mp4`, add a reference photo of person A under **Keep visible**. | "Opt-in verification against a reference the user provides, on this computer. A is recognised again after being fully hidden behind B." |
| 6 | Show `docs/figures/failure_entry.png`. | "The honest limit: someone who walks in between two detections is exposed until the next one, 8 frames here. Detecting more often shrinks it; the interval sweep measures it." |
| 7 | Open `docs/RESULTS.md`, Headline and E2. | "Detecting every 10 frames with tracking was 1.5x faster than per-frame detection and cut exposed frames from 43 to 0 on this clip. All of it regenerates from one command." |

## Likely questions

See `docs/VIVA.md`. The three that come up most: why leak frames rather than
accuracy; why tracking can *reduce* leaks; and what the synthetic data can
and cannot show.

## If something goes wrong

- App does not start: run `python launcher.py` in a terminal to see the error.
- Video preview blank: ffmpeg missing; the exported file still plays in VLC.
- Fall back to `out/demo/demo_side_by_side.mp4` and the figures in `docs/figures/`.
