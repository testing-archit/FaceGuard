"""FaceGuard command line: redact faces in an image or video, headless.

    python cli.py --input photo.jpg --output out.jpg --mode blur
    python cli.py --input photos/ --output redacted/          # batch a folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.errors import IMAGE_EXTS, VIDEO_EXTS, FaceGuardError, media_kind
from core.pipeline import (
    analyse_image,
    check_paths,
    process_video,
    read_image,
    render_image,
    save_tracks,
    write_image,
)
from core.types import DETECTORS, MODES, SHAPES, TRACKERS, Config


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(description="Offline face redaction for images and short videos.")
    p.add_argument("--input", required=True,
                   help="image (.jpg/.jpeg/.png), video (.mp4/.avi/.mov), or a folder of them")
    p.add_argument("--output", required=True,
                   help="where to write the redacted copy (a folder when --input is a folder)")
    p.add_argument("--mode", choices=MODES, default=d.mode)
    p.add_argument("--detector", choices=DETECTORS, default=d.detector)
    p.add_argument("--tracker", choices=TRACKERS, default=d.tracker)
    p.add_argument("--detect-interval", type=int, default=d.detect_interval, help="run the detector every N frames")
    p.add_argument("--conf", type=float, default=d.conf_threshold, help="detection confidence threshold")
    p.add_argument("--padding", type=float, default=d.padding, help="box growth per side, as a fraction of box size")
    p.add_argument("--tiles", type=int, default=d.tile_grid, metavar="N",
                   help="also detect on N x N upscaled tiles to find small faces (0 = off; slower)")
    p.add_argument("--shape", choices=SHAPES, default=d.mask_shape,
                   help="mask shape: soft ellipse, or rect to hide the whole padded box")
    p.add_argument("--smooth", type=int, default=d.smooth_window, metavar="FRAMES",
                   help="video: centred moving average of boxes over this many frames (1 = off)")
    p.add_argument("--keep-audio", action="store_true", help="mux the original audio back into the output")
    p.add_argument("--save-tracks", metavar="JSON", help="video only: write per-frame track boxes for evaluation")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        mode=args.mode,
        detector=args.detector,
        tracker=args.tracker,
        detect_interval=args.detect_interval,
        conf_threshold=args.conf,
        padding=args.padding,
        mask_shape=args.shape,
        tile_grid=args.tiles,
        smooth_window=args.smooth,
        keep_audio=args.keep_audio,
    )


def run_image(src: str, dst: str, cfg: Config) -> None:
    img = read_image(src)
    detections = analyse_image(img, cfg)
    write_image(dst, render_image(img, [d.box for d in detections], cfg))
    if detections:
        print(f"Redacted {len(detections)} face(s) with {cfg.mode} -> {dst}")
    else:
        print(f"No faces detected. Wrote an unchanged copy -> {dst}")


def print_progress(done: int, total: int) -> None:
    if not sys.stderr.isatty() or total <= 0 or done % 10 and done != total:
        return
    width = 30
    filled = min(width, width * done // total)
    print(f"\r[{'#' * filled}{'.' * (width - filled)}] {done}/{total} frames", end="", file=sys.stderr)
    if done >= total:
        print(file=sys.stderr)


def run_video(src: str, dst: str, cfg: Config, tracks_path: str | None = None) -> None:
    analysis = process_video(src, dst, cfg, print_progress)
    stats = analysis.stats
    for w in stats.warnings:
        print(f"warning: {w}", file=sys.stderr)
    merges = [e for e in analysis.events if e.kind == "duplicate"]
    print(
        f"Processed {stats.frames} frames at {stats.fps:.1f} FPS "
        f"({stats.total_seconds:.1f} s total), {len(analysis.tracks)} track(s), "
        f"{stats.detector_calls} detector call(s) ({stats.early_detections} early), "
        f"{stats.scene_cuts} scene cut(s), {len(merges)} suspected ID switch(es) -> {dst}"
    )
    for e in merges:
        print(f"  frame {e.frame}: track {e.dropped} merged into {e.kept}", file=sys.stderr)
    if tracks_path:
        save_tracks(tracks_path, analysis.tracks, cfg)


def batch_jobs(src_dir: Path, dst_dir: Path) -> list[tuple[Path, Path]]:
    """Every supported file directly inside `src_dir`, mapped to the same name in `dst_dir`.

    The output folder must differ from the input folder, so a batch can
    never overwrite an original (FR-21).
    """
    if not src_dir.is_dir():
        raise FaceGuardError(f"Input folder not found: {src_dir}")
    if dst_dir.resolve() == src_dir.resolve():
        raise FaceGuardError("Output folder must differ from the input folder.")
    exts = IMAGE_EXTS + VIDEO_EXTS
    files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FaceGuardError(f"No {', '.join(exts)} files in {src_dir}")
    return [(p, dst_dir / p.name) for p in files]


def run_batch(src_dir: str, dst_dir: str, cfg: Config) -> int:
    """Redact every file in a folder. One bad file is reported and skipped."""
    jobs = batch_jobs(Path(src_dir), Path(dst_dir))
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    failed = 0
    for k, (src, dst) in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {src.name}")
        try:
            if media_kind(src) == "image":
                run_image(str(src), str(dst), cfg)
            else:
                run_video(str(src), str(dst), cfg)
        except FaceGuardError as exc:
            failed += 1
            print(f"  error: {exc}", file=sys.stderr)
    print(f"Batch done: {len(jobs) - failed} redacted, {failed} failed -> {dst_dir}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        if Path(args.input).is_dir():
            return run_batch(args.input, args.output, cfg)
        kind = check_paths(args.input, args.output)
        if kind == "image":
            run_image(args.input, args.output, cfg)
        else:
            run_video(args.input, args.output, cfg, args.save_tracks)
    except (FaceGuardError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
