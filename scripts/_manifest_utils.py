from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_manifest_by_label_count(manifest_path: str | Path, min_label_count: int) -> dict[str, int]:
    manifest_path = Path(manifest_path)
    if int(min_label_count) <= 1:
        df = pd.read_csv(manifest_path)
        counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        return {str(k): int(v) for k, v in counts.items()}

    df = pd.read_csv(manifest_path)
    if df.empty or "label" not in df.columns:
        df.to_csv(manifest_path, index=False)
        return {}

    counts = df["label"].value_counts()
    keep = counts[counts >= int(min_label_count)].index
    filtered = df[df["label"].isin(keep)].copy()
    filtered.to_csv(manifest_path, index=False)
    return {str(k): int(v) for k, v in filtered["label"].value_counts().to_dict().items()}
