from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd


def load_subset_rows(subset_dir: str | Path) -> pd.DataFrame:
    subset_dir = Path(subset_dir)
    candidates = [subset_dir / "subset_manifest.csv", subset_dir / "clips_index.csv", subset_dir / "subset_export.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise SystemExit(f"Missing subset manifest in: {subset_dir}")
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        split_parts = []
        for split_name in ("train", "val", "test"):
            p = subset_dir / f"{split_name}.csv"
            if p.exists():
                part = pd.read_csv(p)
                part["split"] = split_name
                split_parts.append(part)
        if split_parts:
            df = pd.concat(split_parts, ignore_index=True)
    if "target_text" not in df.columns:
        if "text" in df.columns:
            df["target_text"] = df["text"].astype(str)
        elif "label" in df.columns:
            df["target_text"] = df["label"].astype(str).str.replace("_", " ", regex=False)
        else:
            raise SystemExit("Subset manifest must provide either target_text, text, or label")
    return df


def ensure_clip_path(df: pd.DataFrame, clips_dir: Path, clip_ext: str = ".mp4", max_clips: int = 0) -> pd.DataFrame:
    if "clip_path" in df.columns and df["clip_path"].fillna("").astype(str).map(lambda p: Path(p).exists()).all():
        return df

    clips_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    exported = 0
    for row in df.to_dict(orient="records"):
        if int(max_clips) > 0 and exported >= int(max_clips):
            rows.append(row)
            continue
        clip_path = row.get("clip_path")
        if clip_path and Path(str(clip_path)).exists():
            rows.append(row)
            continue
        src = Path(str(row["video_path"]))
        split = str(row.get("split", "train"))
        label = str(row.get("label", "sample"))
        label_dir = clips_dir / split / label
        label_dir.mkdir(parents=True, exist_ok=True)
        episode_stub = str(row.get("episode_id", "episode")).split(":")[-1]
        out_path = label_dir / f"{episode_stub}_{int(row['start_ms']):09d}_{int(row['end_ms']):09d}{clip_ext}"
        if _write_clip(src, out_path, int(row["start_ms"]), int(row["end_ms"])):
            row["clip_path"] = str(out_path)
            exported += 1
        rows.append(row)
    return pd.DataFrame(rows)


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
    wrote = False
    while True:
        pos_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        if pos_ms > int(end_ms):
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        wrote = True
    writer.release()
    cap.release()
    if not wrote and dst.exists():
        dst.unlink()
    return wrote


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def package_signjoey_features(path: Path, rows: list[dict]) -> None:
    """
    Package features for neccam/slt (SignJoey) expected format.

    Input rows are expected to come from `features_package/{split}.jsonl`
    and must include `feature_path` and `target_text`. We generate a list of
    dicts containing torch tensors under the `sign` key:

      {name, signer, gloss, text, sign}

    where `sign` has shape [feature_size, T] and is float32.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Packaging SignJoey data requires PyTorch. Install requirements.txt (includes torch) "
            "or run this step inside the project venv."
        ) from e

    import numpy as np

    payload: list[dict] = []
    for row in rows:
        feature_path = Path(str(row.get("feature_path", "")))
        if not feature_path.exists():
            continue
        data = np.load(str(feature_path), allow_pickle=True)
        seq = np.asarray(data["features"], dtype=np.float32)  # [T, F]
        if seq.ndim != 2 or seq.shape[0] <= 0 or seq.shape[1] <= 0:
            continue
        signer = str(row.get("group_id") or row.get("signer") or "unknown")
        text = str(row.get("target_text") or row.get("text") or "").strip()
        name = str(row.get("sample_id") or row.get("name") or feature_path.stem)
        payload.append(
            {
                "name": name,
                "signer": signer,
                "gloss": "",
                "text": text,
                "sign": torch.from_numpy(seq.T).contiguous(),  # [F, T]
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_slt_dataset_dir(dataset_dir: str | Path, require_features: bool = False) -> dict:
    dataset_dir = Path(dataset_dir)
    rows = load_subset_rows(dataset_dir).to_dict(orient="records")
    split_counts: dict[str, int] = {}
    split_groups: dict[str, set[str]] = {}
    empty_target_text: list[str] = []
    missing_clip_paths: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    for idx, row in enumerate(rows):
        split = str(row.get("split", "train"))
        split_counts[split] = split_counts.get(split, 0) + 1
        split_groups.setdefault(split, set()).add(str(row.get("group_id", "")))
        sample_id = str(row.get("sample_id") or f"{split}_{idx:06d}")
        if not str(row.get("target_text", "")).strip():
            empty_target_text.append(sample_id)
        clip_path = str(row.get("clip_path", "")).strip()
        if clip_path and not Path(clip_path).exists():
            missing_clip_paths.append(clip_path)

    group_overlap: list[str] = []
    split_names = sorted(split_groups.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(g for g in (split_groups[left] & split_groups[right]) if g)
            for group_id in overlap:
                group_overlap.append(f"{left}<->{right}:{group_id}")

    feature_rows: list[dict] = []
    if (dataset_dir / "features_package").exists():
        for split in ("train", "val", "test"):
            feature_rows.extend(_load_jsonl(dataset_dir / "features_package" / f"{split}.jsonl"))
    missing_feature_paths: list[str] = []
    for row in feature_rows:
        feature_path = str(row.get("feature_path", "")).strip()
        if not feature_path or not Path(feature_path).exists():
            missing_feature_paths.append(feature_path or str(row.get("sample_id", "")))

    if group_overlap:
        errors.append("group_id overlap detected across splits")
    if empty_target_text:
        errors.append("some exported rows have empty target_text")
    if missing_clip_paths:
        errors.append("some exported rows reference missing clip_path")
    if require_features and not feature_rows:
        errors.append("missing feature rows under features_package")
    if missing_feature_paths:
        errors.append("some feature rows reference missing feature_path")
    if not split_counts:
        errors.append("dataset export has no rows")
    if "train" not in split_counts:
        warnings.append("dataset export has no train split rows")
    if "test" not in split_counts:
        warnings.append("dataset export has no test split rows")

    return {
        "dataset_dir": str(dataset_dir),
        "valid": not errors,
        "rows": len(rows),
        "split_counts": split_counts,
        "group_overlap": group_overlap,
        "empty_target_text": empty_target_text,
        "missing_clip_paths": missing_clip_paths,
        "feature_rows": len(feature_rows),
        "missing_feature_paths": missing_feature_paths,
        "warnings": warnings,
        "errors": errors,
    }
