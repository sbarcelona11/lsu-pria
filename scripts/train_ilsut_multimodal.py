from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _manifest_utils import filter_manifest_by_label_count
from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare iLSU-T weak labels and train a multimodal temporal baseline.")
    p.add_argument("--episodes-csv", default="auto")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--max-per-seg", type=int, default=40)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    p.add_argument("--manifest-limit", type=int, default=0, help="Optional max number of episodes to read into the weak-label manifest")
    p.add_argument(
        "--min-label-count",
        type=int,
        default=0,
        help="Optional minimum number of weak-labeled segments required to keep a class in the manifest",
    )
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
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir = work_dir / "models"
    results_dir = work_dir / "results"
    prepared_dir = work_dir / "prepared_multimodal"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = args.episodes_csv
    if not episodes_csv or str(episodes_csv).strip().lower() == "auto":
        episodes_csv = work_dir / "episodes_generated.csv"
        build_cmd = [py, repo / "scripts" / "build_ilsut_episodes_csv.py", "--root", args.root, "--out", episodes_csv]
        if args.sources:
            build_cmd += ["--sources", *args.sources]
        _run(build_cmd)

    manifest_path = work_dir / "manifest.csv"
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
        repo / "scripts" / "extract_ilsut_multimodal_sequences.py",
        "--manifest",
        manifest_path,
        "--out-dir",
        prepared_dir,
        "--fps",
        args.fps,
        "--max-per-seg",
        args.max_per_seg,
    ]
    if args.preprocess:
        extract_cmd.append("--preprocess")
    _run(extract_cmd)

    model_path = models_dir / "multimodal_sequence_ilsut.joblib"
    _run(
        [
            py,
            repo / "scripts" / "train_multimodal_sequence.py",
            "--seq-dir",
            prepared_dir,
            "--out",
            model_path,
            "--window",
            args.window,
            "--min-frames",
            args.min_frames,
            "--group-col",
            args.group_col,
        ]
    )
    print(f"Prepared multimodal iLSU-T workspace: {work_dir}")


if __name__ == "__main__":
    main()
