from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert extracted iLSU-T .avi videos to mp4 or mkv using ffmpeg when available, otherwise OpenCV."
    )
    p.add_argument("--root", required=True, help="Root folder with extracted iLSU-T sources")
    p.add_argument("--sources", nargs="+", default=["source2", "source3"])
    p.add_argument("--input-ext", default=".avi")
    p.add_argument("--output-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--tool", choices=["auto", "ffmpeg", "opencv"], default="auto")
    p.add_argument("--quality", choices=["copy", "fast", "balanced"], default="balanced")
    p.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source video after a successful conversion (saves disk space).",
    )
    return p.parse_args()


def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _iter_input_videos(source_root: Path, input_ext: str):
    for p in source_root.rglob(f"*{input_ext}"):
        if p.is_file():
            yield p


def _output_path(root: Path, src: Path, output_ext: str) -> Path:
    rel = src.relative_to(root)
    return root / rel.with_suffix(output_ext)


def _convert_ffmpeg(ffmpeg: str, src: Path, dst: Path, quality: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if quality == "copy":
        cmd = [ffmpeg, "-y", "-i", str(src), "-c", "copy", str(dst)]
    else:
        crf = "23" if quality == "balanced" else "28"
        preset = "medium" if quality == "balanced" else "veryfast"
        # Some iLSU-T AVIs can have odd frame sizes (e.g., height=391).
        # H.264 requires even width/height for yuv420p, so we force an even scale.
        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-vf",
            vf,
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"ffmpeg conversion failed: {src}")


def _convert_opencv(src: Path, dst: Path) -> None:
    import cv2

    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {src}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise SystemExit(f"Could not determine video size: {src}")

        if dst.suffix.lower() == ".mkv":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(dst), fourcc, float(fps), (width, height))
        if not out.isOpened():
            raise SystemExit(f"Could not open output writer: {dst}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                out.write(frame)
        finally:
            out.release()
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    ffmpeg = _find_ffmpeg()
    tool = args.tool
    if tool == "auto":
        tool = "ffmpeg" if ffmpeg else "opencv"
    if tool == "ffmpeg" and not ffmpeg:
        raise SystemExit("Requested ffmpeg but it is not installed.")

    converted = 0
    for source in args.sources:
        source_root = root / source
        if not source_root.exists():
            print(f"skip missing source: {source_root}")
            continue
        for src in _iter_input_videos(source_root, args.input_ext):
            dst = _output_path(root, src, args.output_ext)
            if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
                print(f"skip: {dst}")
                continue
            print(f"convert: {src} -> {dst}")
            if tool == "ffmpeg":
                _convert_ffmpeg(ffmpeg or "ffmpeg", src, dst, args.quality)
            else:
                _convert_opencv(src, dst)
            if args.delete_source and src != dst:
                try:
                    if dst.exists() and dst.stat().st_size > 0:
                        src.unlink()
                except Exception as e:
                    print(f"warn: could not delete source video: {src} ({e})")
            converted += 1

    print(f"Done. converted_videos={converted} tool={tool}")


if __name__ == "__main__":
    main()
