from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.features import extract_landmark_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", type=str, default="", help="Optional grouped split column, e.g. group_id or subject_id")
    return p.parse_args()


def _load_landmarks_tensor(df: pd.DataFrame) -> np.ndarray:
    feat_cols = [c for c in df.columns if c.startswith("lm_")]
    if feat_cols:
        return df[feat_cols].to_numpy(dtype=np.float32).reshape(-1, 21, 3)

    if "landmarks" in df.columns:
        rows = []
        for raw in df["landmarks"].tolist():
            vals = json.loads(str(raw))
            arr = np.array(vals, dtype=np.float32).reshape(21, 3)
            rows.append(arr)
        return np.stack(rows, axis=0)

    raise SystemExit("Could not find landmark columns. Expected `lm_*` columns or a `landmarks` JSON column.")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["label"])

    X_raw = _load_landmarks_tensor(df)
    handed = df.get("handedness", pd.Series([None] * len(df))).tolist()
    X = np.stack([extract_landmark_features(X_raw[i], handed[i]) for i in range(len(df))], axis=0)

    labels = sorted(df["label"].unique().tolist())
    y = df["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
    if args.group_col and args.group_col in df.columns:
        groups = df[args.group_col].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))
    print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": labels,
            "smoothing_alpha": 0.35,
            "group_col": args.group_col,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
