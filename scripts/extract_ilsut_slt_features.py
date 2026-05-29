from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from _slt_dataset_utils import load_subset_rows, write_jsonl
from lsu_pria.slt_features import extract_slt_features_from_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract clip-level SLT features from an iLSU-T subset.")
    p.add_argument("--subset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    df = load_subset_rows(subset_dir)
    wanted = set(args.splits or [])
    if wanted:
        df = df[df["split"].astype(str).isin(wanted)].copy()
    if args.limit:
        df = df.head(int(args.limit)).copy()

    rows_out: list[dict] = []
    split_rows: dict[str, list[dict]] = {}
    for idx, row in enumerate(df.to_dict(orient="records")):
        clip_path = Path(str(row.get("clip_path") or row.get("video_path") or ""))
        if not clip_path.exists():
            continue
        if "clip_path" in row and row.get("clip_path"):
            start_ms = None
            end_ms = None
        else:
            start_ms = int(row.get("start_ms", 0))
            end_ms = int(row.get("end_ms", 0))
        extraction = extract_slt_features_from_video(
            clip_path,
            start_ms=start_ms,
            end_ms=end_ms if end_ms and end_ms > 0 else None,
            sample_fps=float(args.sample_fps),
            max_frames=int(args.max_frames),
            preprocess=bool(args.preprocess),
            include_debug=False,
        )
        if extraction.features.size == 0 or extraction.frames_used <= 0:
            continue
        split = str(row.get("split", "train"))
        sample_id = f"{split}_{idx:06d}"
        split_dir = features_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / f"{sample_id}.npz"
        np.savez_compressed(
            str(out_path),
            features=extraction.features.astype(np.float32),
            ts_ms=extraction.ts_ms.astype(np.int64),
            sample_id=sample_id,
            split=split,
            target_text=str(row.get("target_text", "")),
            label=str(row.get("label", "")),
            group_id=str(row.get("group_id", "")),
            episode_id=str(row.get("episode_id", "")),
            source=str(row.get("source", "")),
            clip_path=str(clip_path),
        )
        payload = {
            "sample_id": sample_id,
            "split": split,
            "feature_path": str(out_path),
            "clip_path": str(clip_path),
            "target_text": str(row.get("target_text", "")),
            "label": str(row.get("label", "")),
            "group_id": str(row.get("group_id", "")),
            "episode_id": str(row.get("episode_id", "")),
            "source": str(row.get("source", "")),
            "frames_used": int(extraction.frames_used),
            "frames_total": int(extraction.frames_total),
        }
        rows_out.append(payload)
        split_rows.setdefault(split, []).append(payload)

    write_jsonl(out_dir / "features_index.jsonl", rows_out)
    for split, rows in split_rows.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    (out_dir / "features_info.json").write_text(
        json.dumps(
            {
                "rows": len(rows_out),
                "sample_fps": float(args.sample_fps),
                "max_frames": int(args.max_frames),
                "preprocess": bool(args.preprocess),
                "splits": {k: len(v) for k, v in split_rows.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {out_dir / 'features_index.jsonl'}")
    print(f"Wrote: {out_dir / 'features_info.json'}")


if __name__ == "__main__":
    main()
