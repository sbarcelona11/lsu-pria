from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from _bootstrap import ensure_repo_root_on_path
from _manifest_utils import filter_manifest_by_label_count

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare an iLSU-T clip-level subset with grouped train/val/test splits, closer to the official SLT dataset workflow."
    )
    p.add_argument("--episodes-csv", default="auto")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--min-label-count", type=int, default=20)
    p.add_argument("--labels-json", default="", help="Optional recommended_labels.json to force a curated label set")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1, help="Fraction of the full subset used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-clips", action="store_true")
    p.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--max-clips", type=int, default=0, help="Optional cap on exported clips for quick iterations")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _write_clip(src: Path, dst: Path, start_ms: int, end_ms: int) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return False
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if dst.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, int(start_ms)))
    ok = False
    while True:
        pos_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_ms > int(end_ms):
            break
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        ok = True
    writer.release()
    cap.release()
    if not ok and dst.exists():
        dst.unlink()
    return ok


def _assign_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.Series:
    groups = df["group_id"].astype(str).to_numpy()
    idx = df.index.to_numpy()
    y = df["label"].astype(str).to_numpy()
    unique_groups = pd.unique(groups)

    if len(unique_groups) <= 1:
        return pd.Series(index=df.index, data="train", dtype="object")

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_val_idx, test_idx = next(splitter_test.split(idx, y, groups=groups))

    train_val_abs = idx[train_val_idx]
    test_abs = idx[test_idx]
    train_val_df = df.loc[train_val_abs]
    train_val_groups = train_val_df["group_id"].astype(str).to_numpy()
    train_val_y = train_val_df["label"].astype(str).to_numpy()
    unique_train_val_groups = pd.unique(train_val_groups)

    val_share = float(val_size) / max(1e-9, 1.0 - float(test_size))
    val_share = min(max(val_share, 0.05), 0.5)
    if len(unique_train_val_groups) <= 1:
        train_abs = train_val_abs
        val_abs = train_val_abs[:0]
    else:
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=int(seed))
        train_rel, val_rel = next(splitter_val.split(train_val_abs, train_val_y, groups=train_val_groups))
        train_abs = train_val_abs[train_rel]
        val_abs = train_val_abs[val_rel]

    split = pd.Series(index=df.index, data="train", dtype="object")
    split.loc[test_abs] = "test"
    split.loc[val_abs] = "val"
    split.loc[train_abs] = "train"
    return split


def _export_clips(df: pd.DataFrame, clips_dir: Path, clip_ext: str, max_clips: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = 0
    for row in df.itertuples(index=False):
        if int(max_clips) > 0 and total >= int(max_clips):
            break
        label_dir = clips_dir / str(row.split) / str(row.label)
        label_dir.mkdir(parents=True, exist_ok=True)
        clip_name = f"{row.source}_{row.episode_id.split(':')[-1]}_{int(row.start_ms):09d}_{int(row.end_ms):09d}{clip_ext}"
        clip_path = label_dir / clip_name
        ok = _write_clip(Path(row.video_path), clip_path, int(row.start_ms), int(row.end_ms))
        if not ok:
            continue
        rec = row._asdict()
        rec["clip_path"] = str(clip_path)
        rows.append(rec)
        total += 1
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

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

    filter_manifest_by_label_count(manifest_path, int(args.min_label_count))
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise SystemExit("No rows left in manifest after filtering.")

    if args.labels_json:
        payload = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        keep_labels = set(payload.get("labels", []))
        if keep_labels:
            df = df[df["label"].isin(keep_labels)].copy()
        if df.empty:
            raise SystemExit("No rows left after applying labels-json selection.")

    df["split"] = _assign_splits(df, float(args.test_size), float(args.val_size), int(args.seed))
    subset_manifest = work_dir / "subset_manifest.csv"
    df.to_csv(subset_manifest, index=False)
    for split_name in ("train", "val", "test"):
        df[df["split"] == split_name].to_csv(work_dir / f"{split_name}.csv", index=False)

    split_group_counts = {}
    if "group_id" in df.columns:
        split_group_counts = {
            str(split_name): int(part["group_id"].astype(str).nunique())
            for split_name, part in df.groupby("split")
        }
    summary = {
        "sources": list(args.sources or []),
        "rows": int(len(df)),
        "labels": sorted(df["label"].astype(str).unique().tolist()),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "split_episode_counts": split_group_counts,
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "episodes": int(df["group_id"].astype(str).nunique()) if "group_id" in df.columns else None,
        "config": {
            "episodes_csv": str(episodes_csv),
            "root": str(args.root),
            "keywords": str(args.keywords),
            "path_mode": str(args.path_mode),
            "manifest_limit": int(args.manifest_limit),
            "min_label_count": int(args.min_label_count),
            "labels_json": str(args.labels_json or ""),
            "test_size": float(args.test_size),
            "val_size": float(args.val_size),
            "seed": int(args.seed),
            "export_clips": bool(args.export_clips),
            "clip_ext": str(args.clip_ext),
            "max_clips": int(args.max_clips),
        },
    }
    (work_dir / "subset_info.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.export_clips:
        clips_dir = work_dir / "clips"
        clips_df = _export_clips(df, clips_dir, args.clip_ext, int(args.max_clips))
        clips_df.to_csv(work_dir / "clips_index.csv", index=False)
        print(f"Wrote: {work_dir / 'clips_index.csv'}")

    print(f"Wrote: {subset_manifest}")
    print(f"Wrote: {work_dir / 'train.csv'}")
    print(f"Wrote: {work_dir / 'val.csv'}")
    print(f"Wrote: {work_dir / 'test.csv'}")
    print(f"Wrote: {work_dir / 'subset_info.json'}")


if __name__ == "__main__":
    main()
