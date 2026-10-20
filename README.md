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
- Preview before export; original audio preserved via ffmpeg mux

## Install

Requires Python 3.10+ and `ffmpeg` on your PATH.

```bash
git clone <your-repo-url> faceguard
cd faceguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py   # downloads the YuNet ONNX model (~338 KB)
```

`opencv-contrib-python` is required, not `opencv-python`. The CSRT and KCF
trackers live in the contrib build. If both are installed, uninstall both and
reinstall only the contrib wheel.

## Usage

### UI

```bash
streamlit run ui/app.py
```

Upload a file, wait for detection, untick any face you want left visible,
choose a redaction mode, export.

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
| `--padding` | `0.2` | Expand each box by this fraction before masking |
| `--keep-audio` | off | Mux the original audio back into the output |

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

MIT (code). The YuNet model is distributed under its own OpenCV Zoo license.
