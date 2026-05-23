from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the recommended end-to-end training stack on a local collected dataset."
    )
    p.add_argument("--csv", default="", help="Single landmarks CSV to use directly")
    p.add_argument("--csvs", nargs="*", default=None, help="Optional multiple landmarks CSVs to merge first")
    p.add_argument("--work-dir", required=True, help="Workspace for merged csv, models and reports")
    p.add_argument("--cnn-image-col", default="img_raw_path", help="img_raw_path | img_masked_path | auto")
    p.add_argument("--group-col", default="subject_id")
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
    models_dir = work_dir / "models"
    results_dir = work_dir / "results"
    data_dir = work_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.csvs:
        merged_csv = data_dir / "merged_landmarks.csv"
        _run(
            [
                py,
                repo / "scripts" / "merge_subject_csvs.py",
                "--out",
                merged_csv,
                "--inputs",
                *args.csvs,
            ]
        )
        csv_path = merged_csv
    elif args.csv:
        csv_path = Path(args.csv)
    else:
        raise SystemExit("Provide either --csv or --csvs.")

    landmarks_model = models_dir / "landmarks.joblib"
    cnn_model = models_dir / "cnn.pt"

    _run(
        [
            py,
            repo / "scripts" / "train_landmarks.py",
            "--csv",
            csv_path,
            "--out",
            landmarks_model,
            "--group-col",
            args.group_col,
        ]
    )

    _run(
        [
            py,
            repo / "scripts" / "train_cnn.py",
            "--csv",
            csv_path,
            "--image-col",
            args.cnn_image_col,
            "--out",
            cnn_model,
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
            csv_path,
            "--by",
            "label_subject",
            "--out-md",
            results_dir / "dataset_stats.md",
        ]
    )

    _run(
        [
            py,
            repo / "scripts" / "eval_split.py",
            "--csv",
            csv_path,
            "--landmarks-model",
            landmarks_model,
            "--cnn-model",
            cnn_model,
            "--cnn-image-col",
            args.cnn_image_col,
            "--group-col",
            args.group_col,
            "--json-out",
            results_dir / "eval_split.json",
        ]
    )

    if not args.skip_ablation:
        _run(
            [
                py,
                repo / "scripts" / "run_ablation.py",
                "--csv",
                csv_path,
                "--landmarks-model",
                landmarks_model,
                "--cnn-model",
                cnn_model,
                "--cnn-image-col",
                args.cnn_image_col,
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
        )

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

    print(f"Prepared stack workspace: {work_dir}")


if __name__ == "__main__":
    main()

