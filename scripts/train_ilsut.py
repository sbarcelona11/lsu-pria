from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare iLSU-T weak labels and train one or both project pipelines from the extracted samples."
    )
    p.add_argument("--episodes-csv", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True, help="Output workspace for manifest, extracted samples, and models")
    p.add_argument("--pipelines", nargs="+", choices=["landmarks", "cnn"], default=["cnn"])
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--max-per-seg", type=int, default=40)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--skin-mask", action="store_true")
    p.add_argument("--camera-like", action="store_true")
    p.add_argument("--group-col", default="group_id")
    p.add_argument("--cnn-epochs", type=int, default=10)
    p.add_argument("--cnn-device", default="cpu")
    p.add_argument("--cnn-use-masked", action="store_true")
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
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
    manifest_path = work_dir / "manifest.csv"
    extract_dir = work_dir / "prepared"
    models_dir = work_dir / "models"
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            py,
            repo / "scripts" / "ilsut_make_manifest.py",
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--out",
            manifest_path,
            "--path-mode",
            args.path_mode,
        ]
    )

    extract_cmd = [
        py,
        repo / "scripts" / "ilsut_extract_rois.py",
        "--manifest",
        manifest_path,
        "--out-dir",
        extract_dir,
        "--fps",
        args.fps,
        "--max-per-seg",
        args.max_per_seg,
        "--save-landmarks",
    ]
    if args.preprocess:
        extract_cmd.append("--preprocess")
    if args.skin_mask:
        extract_cmd += ["--skin-mask", "--save-masked"]
    if args.camera_like:
        extract_cmd.append("--camera-like")
    _run(extract_cmd)

    if "landmarks" in args.pipelines:
        _run(
            [
                py,
                repo / "scripts" / "train_landmarks.py",
                "--csv",
                extract_dir / "landmarks.csv",
                "--out",
                models_dir / "landmarks_ilsut.joblib",
                "--group-col",
                args.group_col,
            ]
        )

    if "cnn" in args.pipelines:
        image_col = "img_masked_path" if args.cnn_use_masked else "img_raw_path"
        _run(
            [
                py,
                repo / "scripts" / "train_cnn.py",
                "--csv",
                extract_dir / "samples.csv",
                "--image-col",
                image_col,
                "--out",
                models_dir / "cnn_ilsut.pt",
                "--epochs",
                args.cnn_epochs,
                "--device",
                args.cnn_device,
                "--group-col",
                args.group_col,
            ]
        )

    print(f"Prepared iLSU-T workspace: {work_dir}")


if __name__ == "__main__":
    main()

