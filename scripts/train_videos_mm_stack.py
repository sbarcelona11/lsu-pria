from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare labeled videos and run the multimodal temporal training stack.")
    p.add_argument("--videos-root", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    p.add_argument("--subject-default", default="video")
    p.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="group_id")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"

    work_dir = Path(args.work_dir)
    prepared_dir = work_dir / "prepared_multimodal_from_videos"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    extract_cmd: list[object] = [
        py,
        repo / "scripts" / "extract_labeled_video_multimodal_sequences.py",
        "--videos-root",
        args.videos_root,
        "--out-dir",
        prepared_dir,
        "--layout",
        args.layout,
        "--subject-default",
        args.subject_default,
        "--fps",
        args.fps,
        "--min-frames",
        args.min_frames,
        "--exts",
        *args.exts,
    ]
    if args.preprocess:
        extract_cmd.append("--preprocess")
    _run([str(x) for x in extract_cmd])

    train_cmd: list[object] = [
        py,
        repo / "scripts" / "train_multimodal_stack.py",
        "--seq-dir",
        prepared_dir,
        "--work-dir",
        work_dir,
        "--window",
        args.window,
        "--min-frames",
        args.min_frames,
        "--group-col",
        args.group_col,
    ]
    _run([str(x) for x in train_cmd])


if __name__ == "__main__":
    main()
