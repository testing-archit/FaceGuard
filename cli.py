"""FaceGuard command line: redact faces in an image or video, headless.

    python cli.py --input photo.jpg --output out.jpg --mode blur
"""

from __future__ import annotations

import argparse
import sys

from core.errors import FaceGuardError
from core.pipeline import (
    analyse_image,
    check_paths,
    process_video,
    read_image,
    render_image,
    save_tracks,
    write_image,
)
from core.types import DETECTORS, MODES, TRACKERS, Config


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(description="Offline face redaction for images and short videos.")
    p.add_argument("--input", required=True, help="image (.jpg/.jpeg/.png) or video (.mp4/.avi/.mov)")
    p.add_argument("--output", required=True, help="where to write the redacted copy")
    p.add_argument("--mode", choices=MODES, default=d.mode)
    p.add_argument("--detector", choices=DETECTORS, default=d.detector)
    p.add_argument("--tracker", choices=TRACKERS, default=d.tracker)
    p.add_argument("--detect-interval", type=int, default=d.detect_interval, help="run the detector every N frames")
    p.add_argument("--conf", type=float, default=d.conf_threshold, help="detection confidence threshold")
    p.add_argument("--padding", type=float, default=d.padding, help="box growth per side, as a fraction of box size")
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
    print(
        f"Processed {stats.frames} frames at {stats.fps:.1f} FPS "
        f"({stats.total_seconds:.1f} s total), {len(analysis.tracks)} track(s), "
        f"{len(analysis.events)} suspected ID switch(es) -> {dst}"
    )
    for e in analysis.events:
        print(f"  frame {e.frame}: track {e.dropped} merged into {e.kept}", file=sys.stderr)
    if tracks_path:
        save_tracks(tracks_path, analysis.tracks)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
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
