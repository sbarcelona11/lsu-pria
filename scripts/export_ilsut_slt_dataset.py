from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from _slt_dataset_utils import ensure_clip_path, load_subset_rows, package_signjoey_features, validate_slt_dataset_dir, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an iLSU-T subset as a reusable SLT dataset package.")
    p.add_argument("--subset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["features", "raw"], default="features")
    p.add_argument("--sample-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _write_backend_bundle(out_dir: Path, split_rows: dict[str, list[dict]], args: argparse.Namespace) -> None:
    backend_dir = out_dir / "backend" / args.backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    data_dir = backend_dir / "data" / "ilsut"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        package_signjoey_features(data_dir / f"ilsut.{split}", rows)
    cfg_text = "\n".join(
        [
            "name: ilsut_slt_wrapper",
            "data:",
            "  data_path: ./data/ilsut",
            "  train: ilsut.train",
            "  dev: ilsut.val",
            "  test: ilsut.test",
            "training:",
            "  epochs: 10",
            "  batch_size: 16",
            "model:",
            "  note: Generated wrapper config. Adjust to the backend schema before real training.",
            "",
        ]
    )
    (backend_dir / "config_template.yaml").write_text(cfg_text, encoding="utf-8")
    run_text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'BACKEND_REPO=\"${1:-/abs/path/to/neccam-slt}\"',
            "cd \"$BACKEND_REPO\"",
            "python -m signjoey train /abs/path/to/config_template.yaml",
            "",
        ]
    )
    run_path = backend_dir / "run_train.sh"
    run_path.write_text(run_text, encoding="utf-8")
    run_path.chmod(0o755)
    notes = "\n".join(
        [
            "# neccam/slt backend bundle",
            "",
            "- The packaged split files are gzip+pickle payloads inspired by the `slt` repo expectations.",
            "- `config_template.yaml` is a starter config and likely needs adaptation to the backend version.",
            "- `run_train.sh` is a helper stub; this project's `train-ilsut-slt` wrapper can also attempt external execution.",
            "",
        ]
    )
    (backend_dir / "README_backend.md").write_text(notes, encoding="utf-8")


def _apply_limit(df, limit: int):
    if int(limit) <= 0 or len(df) <= int(limit):
        return df
    if "split" not in df.columns:
        return df.head(int(limit)).copy()
    parts = []
    splits = [name for name, part in df.groupby("split", sort=False) if len(part) > 0]
    if not splits:
        return df.head(int(limit)).copy()
    base = max(1, int(limit) // len(splits))
    used = 0
    for split_name in splits:
        part = df[df["split"] == split_name]
        take = min(len(part), base)
        parts.append(part.head(take))
        used += take
    remaining = int(limit) - used
    if remaining > 0:
        remainder = df.drop(index=pd.concat(parts).index if parts else [])
        if not remainder.empty:
            parts.append(remainder.head(remaining))
    return pd.concat(parts, ignore_index=False).sort_index().reset_index(drop=True)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_subset_rows(subset_dir)
    if args.limit:
        df = _apply_limit(df, int(args.limit)).copy()
    df = ensure_clip_path(df, out_dir / "clips", clip_ext=args.clip_ext, max_clips=int(args.max_clips))

    subset_export = out_dir / "subset_export.csv"
    df.to_csv(subset_export, index=False)

    split_rows: dict[str, list[dict]] = {}
    for row in df.to_dict(orient="records"):
        split = str(row.get("split", "train"))
        payload = {
            "sample_id": str(row.get("sample_id") or f"{split}_{len(split_rows.get(split, [])):06d}"),
            "split": split,
            "clip_path": str(row.get("clip_path", "")),
            "target_text": str(row.get("target_text", "")),
            "label": str(row.get("label", "")),
            "group_id": str(row.get("group_id", "")),
            "episode_id": str(row.get("episode_id", "")),
            "source": str(row.get("source", "")),
            "start_ms": int(row.get("start_ms", 0)),
            "end_ms": int(row.get("end_ms", 0)),
            "video_path": str(row.get("video_path", "")),
        }
        split_rows.setdefault(split, []).append(payload)

    manifests_dir = out_dir / "manifests"
    for split, rows in split_rows.items():
        write_jsonl(manifests_dir / f"{split}.jsonl", rows)
        write_csv(manifests_dir / f"{split}.csv", rows)

    feature_dir = out_dir / "features_package"
    if args.mode == "features":
        _run(
            [
                py,
                repo / "scripts" / "extract_ilsut_slt_features.py",
                "--subset-dir",
                out_dir,
                "--out-dir",
                feature_dir,
                "--sample-fps",
                args.sample_fps,
                "--max-frames",
                args.max_frames,
                "--splits",
                "train",
                "val",
                "test",
            ]
            + (["--preprocess"] if args.preprocess else [])
        )
        features_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for split in ("train", "val", "test"):
            p = feature_dir / f"{split}.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            features_rows[split] = rows
        _write_backend_bundle(out_dir, features_rows, args)

    metadata = {
        "mode": args.mode,
        "sample_fps": float(args.sample_fps),
        "max_frames": int(args.max_frames),
        "preprocess": bool(args.preprocess),
        "rows": int(len(df)),
        "splits": {k: len(v) for k, v in split_rows.items()},
        "backend": args.backend,
        "subset_dir": str(subset_dir),
    }
    (out_dir / "dataset_export_info.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = validate_slt_dataset_dir(out_dir, require_features=args.mode == "features")
    (out_dir / "dataset_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {subset_export}")
    print(f"Wrote: {out_dir / 'dataset_export_info.json'}")
    print(f"Wrote: {out_dir / 'dataset_validation.json'}")
    if not validation["valid"]:
        raise SystemExit("Exported SLT dataset failed validation")


if __name__ == "__main__":
    main()
