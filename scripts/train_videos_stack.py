from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare labeled videos and run the frame-based training stack.")
    p.add_argument("--videos-root", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    p.add_argument("--subject-default", default="video")
    p.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--max-per-video", type=int, default=120)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--skin-mask", action="store_true")
    p.add_argument("--camera-like", action="store_true")
    p.add_argument("--cnn-image-col", default="img_raw_path")
    p.add_argument("--group-col", default="group_id")
    p.add_argument("--cnn-epochs", type=int, default=10)
    p.add_argument("--cnn-device", default="cpu")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--ablation-seconds", type=float, default=10.0)
    p.add_argument("--camera", type=int, default=0)
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
    prepared_dir = work_dir / "prepared_from_videos"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    extract_cmd: list[object] = [
        py,
        repo / "scripts" / "extract_labeled_video_dataset.py",
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
        "--max-per-video",
        args.max_per_video,
        "--exts",
        *args.exts,
        "--save-landmarks",
    ]
    if args.preprocess:
        extract_cmd.append("--preprocess")
    if args.skin_mask:
        extract_cmd += ["--skin-mask", "--save-masked"]
    if args.camera_like:
        extract_cmd.append("--camera-like")
    _run([str(x) for x in extract_cmd])

    train_cmd: list[object] = [
        py,
        repo / "scripts" / "train_stack.py",
        "--csv",
        prepared_dir / "landmarks.csv",
        "--work-dir",
        work_dir,
        "--cnn-image-col",
        args.cnn_image_col,
        "--group-col",
        args.group_col,
        "--cnn-epochs",
        args.cnn_epochs,
        "--cnn-device",
        args.cnn_device,
        "--ablation-seconds",
        args.ablation_seconds,
        "--camera",
        args.camera,
    ]
    if args.skip_ablation:
        train_cmd.append("--skip-ablation")
    _run([str(x) for x in train_cmd])


if __name__ == "__main__":
    main()
