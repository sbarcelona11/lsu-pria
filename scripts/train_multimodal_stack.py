from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the recommended local training stack for multimodal sequences.")
    p.add_argument("--seq-dir", default="", help="Single multimodal sequence root")
    p.add_argument("--seq-dirs", nargs="*", default=None, help="Optional multiple multimodal sequence roots")
    p.add_argument("--work-dir", required=True, help="Workspace for merged sequences, models and reports")
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="subject_id")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _copy_npz_tree(src_root: Path, dst_root: Path) -> dict[str, int]:
    copied = 0
    label_counts: Counter[str] = Counter()
    for p in sorted(src_root.rglob("*.npz")):
        label = p.parent.name
        rel = p.relative_to(src_root)
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.read_bytes())
        copied += 1
        label_counts[label] += 1
    return {"copied": copied, "labels": dict(label_counts)}


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

    seq_dirs: list[Path] = []
    if args.seq_dirs:
        seq_dirs.extend(Path(x) for x in args.seq_dirs)
    elif args.seq_dir:
        seq_dirs.append(Path(args.seq_dir))
    else:
        raise SystemExit("Provide either --seq-dir or --seq-dirs.")

    merged_root = data_dir / "multimodal_sequences"
    merged_root.mkdir(parents=True, exist_ok=True)
    merge_meta: dict[str, dict] = {}
    total = 0
    label_totals: Counter[str] = Counter()
    for src in seq_dirs:
        real_src = src / "multimodal_sequences" if (src / "multimodal_sequences").exists() else src
        summary = _copy_npz_tree(real_src, merged_root)
        merge_meta[str(src)] = summary
        total += int(summary["copied"])
        for label, count in summary["labels"].items():
            label_totals[label] += int(count)

    (results_dir / "multimodal_dataset_stats.json").write_text(
        json.dumps(
            {
                "sources": merge_meta,
                "total_sequences": total,
                "label_counts": dict(label_totals),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _run(
        [
            py,
            repo / "scripts" / "train_multimodal_sequence.py",
            "--seq-dir",
            merged_root,
            "--out",
            models_dir / "multimodal_sequence.joblib",
            "--window",
            args.window,
            "--min-frames",
            args.min_frames,
            "--group-col",
            args.group_col,
        ]
    )
    print(f"Prepared multimodal stack workspace: {work_dir}")


if __name__ == "__main__":
    main()
