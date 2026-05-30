from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _manifest_utils import filter_manifest_by_label_count
from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare iLSU-T weak labels and train one or both project pipelines from the extracted samples."
    )
    p.add_argument(
        "--episodes-csv",
        default="auto",
        help="Path to the iLSU-T episodes CSV, or 'auto' to build one from the extracted files",
    )
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True, help="Output workspace for manifest, extracted samples, and models")
    p.add_argument("--sources", nargs="*", default=None, help="Optional source folders used when auto-building episodes.csv")
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
    p.add_argument("--manifest-limit", type=int, default=0, help="Optional max number of episodes to read into the weak-label manifest")
    p.add_argument(
        "--min-label-count",
        type=int,
        default=0,
        help="Optional minimum number of weak-labeled segments required to keep a class in the manifest",
    )
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
    manifest_path = work_dir / "manifest.csv"
    extract_dir = work_dir / "prepared"
    models_dir = work_dir / "models"
    results_dir = work_dir / "results"
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv = args.episodes_csv

    if not episodes_csv or str(episodes_csv).strip().lower() == "auto":
        episodes_csv = work_dir / "episodes_generated.csv"
        build_cmd = [
            py,
            repo / "scripts" / "build_ilsut_episodes_csv.py",
            "--root",
            args.root,
            "--out",
            episodes_csv,
        ]
        if args.sources:
            build_cmd += ["--sources", *args.sources]
        _run(build_cmd)

    _run(
        [
            py,
            repo / "scripts" / "ilsut_make_manifest.py",
            "--episodes-csv",
            episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--out",
            manifest_path,
            "--path-mode",
            args.path_mode,
            "--limit",
            args.manifest_limit,
        ]
    )
    kept_counts = filter_manifest_by_label_count(manifest_path, int(args.min_label_count))
    if kept_counts:
        print(f"Filtered manifest labels: kept={kept_counts}")
    else:
        print("Filtered manifest labels: no rows left after filtering" if int(args.min_label_count) > 1 else "Manifest labels kept as-is")

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

    _run(
        [
            py,
            repo / "scripts" / "dataset_stats.py",
            "--csv",
            extract_dir / "landmarks.csv",
            "--by",
            "label_subject",
            "--out-md",
            results_dir / "dataset_stats.md",
        ]
    )

    eval_cmd = [
        py,
        repo / "scripts" / "eval_split.py",
        "--csv",
        extract_dir / "landmarks.csv",
        "--group-col",
        args.group_col,
        "--json-out",
        results_dir / "eval_split.json",
    ]
    if "landmarks" in args.pipelines:
        eval_cmd += ["--landmarks-model", models_dir / "landmarks_ilsut.joblib"]
    if "cnn" in args.pipelines:
        image_col = "img_masked_path" if args.cnn_use_masked else "img_raw_path"
        eval_cmd += ["--cnn-model", models_dir / "cnn_ilsut.pt", "--cnn-image-col", image_col]
    _run(eval_cmd)

    if not args.skip_ablation:
        ablation_cmd = [
            py,
            repo / "scripts" / "run_ablation.py",
            "--csv",
            extract_dir / "landmarks.csv",
            "--group-col",
            args.group_col,
            "--seconds",
            args.ablation_seconds,
            "--camera",
            args.camera,
            "--out-md",
            results_dir / "ablation_table.md",
            "--out-json",
            results_dir / "ablation_results.json",
        ]
        if "landmarks" in args.pipelines:
            ablation_cmd += ["--landmarks-model", models_dir / "landmarks_ilsut.joblib"]
        if "cnn" in args.pipelines:
            image_col = "img_masked_path" if args.cnn_use_masked else "img_raw_path"
            ablation_cmd += ["--cnn-model", models_dir / "cnn_ilsut.pt", "--cnn-image-col", image_col]
        _run(ablation_cmd)

        _run(
            [
                py,
                repo / "scripts" / "make_report.py",
                "--ablation-json",
                results_dir / "ablation_results.json",
                "--ablation-table",
                results_dir / "ablation_table.md",
                "--dataset-stats-md",
                results_dir / "dataset_stats.md",
                "--out-md",
                results_dir / "report.md",
                "--out-fig",
                results_dir / "precision_vs_fps.png",
            ]
        )

    print(f"Prepared iLSU-T workspace: {work_dir}")


if __name__ == "__main__":
    main()
