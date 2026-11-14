# FaceGuard

Offline face detection, tracking and redaction for images and videos.

FaceGuard finds faces in a photo or short video, lets you pick which ones to
hide, and exports a redacted copy. All processing happens locally on your CPU.
No upload, no API key, no internet connection needed once set up.

## Why

Anyone filming in a public space records bystanders who never consented.
Manually keyframing a blur over every face in a video editor is slow enough
that most people skip it. Facial images are personal data under India's DPDP
Act 2023 and under GDPR, and sending faces to a cloud service in order to
protect them is self-defeating. Hence: local only.

## Features

- Face detection using YuNet (a small CNN) via OpenCV's `FaceDetectorYN`
- Optional Haar cascade detector as a classical baseline for comparison
- Face tracking between detections using pyramidal Lucas-Kanade optical flow,
  with CSRT and KCF as alternatives
- Three redaction modes: Gaussian blur, pixelation, solid fill
- Per-person selection — tick a face once, it stays hidden through the clip
- Manual box drawing to cover anything the detector missed
- Preview before export; optional audio preservation via ffmpeg mux
  (`--keep-audio`)

## Install

Requires Python 3.10+ and `ffmpeg` on your PATH.

```bash
git clone https://github.com/testing-archit/FaceGuard.git faceguard
cd faceguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py   # only if models/ is missing; the model is committed (~224 KB)
```

`opencv-contrib-python` is required, not `opencv-python`. The CSRT and KCF
trackers live in the contrib build. If both are installed, uninstall both and
reinstall only the contrib wheel.

## Usage

### UI

```bash
streamlit run ui/app.py
```

Opens on `http://localhost:8501`, bound to this machine with Streamlit usage
statistics disabled (`.streamlit/config.toml`).

1. Upload a photo or video and press **Detect faces**.
2. **Review.** Click a face on the image to hide or show that person; drag on
   the image to cover a face the detector missed (fine-tune the box in the
   panel, and for video choose whether the tracker follows it, it holds
   still, or it moves to an end-frame box). The People list has thumbnails,
   toggles and delete for false positives.
3. **Video.** Scrub with the slider or click the timeline. The flag buttons
   jump between frames where a box is a tracker estimate the detector did not
   confirm: those are the frames to check.
4. **Export.** Switch the view to Preview, then render. Download the file and,
   optionally, a JSON redaction log of what was hidden where. Audio is dropped
   unless you turn on **Keep original audio**; metadata is never copied.

The **Batch photos** tab redacts many photos at once and returns a ZIP.
Presets (Maximum privacy, Balanced, Fast) set several options together.

**Reset session** deletes the upload and every preview and export.

### Command line

```bash
python cli.py --input photo.jpg --output out.jpg --mode blur
python cli.py --input clip.mp4 --output out.mp4 --mode pixelate \
              --detect-interval 10 --tracker lk --keep-audio
```

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `pixelate` | `blur`, `pixelate`, or `fill` |
| `--detector` | `yunet` | `yunet` or `haar` |
| `--tracker` | `lk` | `lk`, `csrt`, `kcf`, or `none` |
| `--detect-interval` | `10` | Run the detector every N frames |
| `--conf` | `0.6` | Detection confidence threshold |
| `--padding` | `0.25` | Expand each box by this fraction of its size, per side |
| `--shape` | `ellipse` | `ellipse` or `rect` (hides the whole padded box) |
| `--smooth` | `1` | Video: average boxes over N frames (1 = off) |
| `--keep-audio` | off | Mux the original audio back into the output |

`--input` may be a folder: every supported file in it is redacted into the
`--output` folder.

## Evaluation

```bash
python scripts/fetch_samples.py         # public OpenCV test photos (one-time)
python scripts/make_synthetic_clip.py   # 20 s clip, exact ground truth
python scripts/make_crossing_clip.py    # 15 s clip: occlusion, crossing, entry
python eval/annotate.py images data/samples/my_photo.jpg   # label real media
python eval/annotate.py video data/samples/my_clip.mp4 --every 5
python eval/run_experiments.py --config eval/experiments.yaml
```

The last command regenerates every table in `docs/RESULTS.md` and every plot
in `eval/out/` (about 10 minutes on an Apple M5; `--runs 1` for a quick check).
Leak frames, the count of frames in which any face is less than 90% covered,
is the headline metric: one such frame defeats the tool.

## A note on how private this actually is

Gaussian blur is the weakest option. Deblurring and face-restoration models can
partially recover features from blurred faces. Pixelation is better but can be
attacked if the adversary has a small candidate set of faces. **Solid fill is
the only mode that destroys the information outright.** Use `fill` when it
genuinely matters.

FaceGuard also will not catch every face. Profile views, very small faces, heavy
motion blur and partial occlusion all cause misses. Always review the preview
before publishing.

## Project layout

See `docs/ARCHITECTURE.md`.

## License

MIT (code). The YuNet model is also MIT-licensed (Copyright (c) 2020 Shiqi Yu),
see the OpenCV Zoo `models/face_detection_yunet/LICENSE`.
