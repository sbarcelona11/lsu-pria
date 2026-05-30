from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.features import extract_landmark_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple temporal baseline from collected landmark sequences.")
    p.add_argument("--seq-dir", required=True, help="Directory produced by scripts/collect_sequence.py (contains sequences/<label>/*.npz)")
    p.add_argument("--out", required=True, help="Output joblib model file")
    p.add_argument("--window", type=int, default=16, help="Window size used at inference")
    p.add_argument("--min-frames", type=int, default=8, help="Min frames required to predict")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--group-col", choices=["subject_id", "none"], default="subject_id", help="Split by subject_id if available")
    return p.parse_args()


def _aggregate_window(window_feats: np.ndarray) -> np.ndarray:
    mean = window_feats.mean(axis=0)
    std = window_feats.std(axis=0)
    delta = window_feats[-1] - window_feats[0]
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)


def _load_sequences(seq_dir: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    seqs: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for p in sorted(seq_dir.rglob("*.npz")):
        try:
            d = np.load(str(p), allow_pickle=True)
        except Exception:
            continue
        lm = d.get("landmarks")
        lab = d.get("label")
        subj = d.get("subject_id")
        if lm is None or lab is None:
            continue
        try:
            lm = np.array(lm, dtype=np.float32)
        except Exception:
            continue
        if lm.ndim != 3 or lm.shape[1:] != (21, 3):
            continue
        seqs.append(lm)
        labels.append(str(lab))
        groups.append(str(subj) if subj is not None else "S?")
    if not seqs:
        raise SystemExit(f"No sequences found under: {seq_dir}")
    return seqs, labels, groups


def _windowify(seq_feats: np.ndarray, window: int) -> np.ndarray:
    # seq_feats: (T,F). Return (window,F) by padding/striding.
    T = seq_feats.shape[0]
    if T >= window:
        return seq_feats[-window:]
    pad = np.repeat(seq_feats[:1], window - T, axis=0)
    return np.concatenate([pad, seq_feats], axis=0)


def main() -> None:
    args = parse_args()
    seq_root = Path(args.seq_dir)
    # Accept either the root out dir or the sequences/ dir.
    if (seq_root / "sequences").exists():
        seq_root = seq_root / "sequences"

    seqs, y_str, groups = _load_sequences(seq_root)

    # Build per-sequence aggregated features from per-frame landmark features.
    X: list[np.ndarray] = []
    for lm in seqs:
        # Extract per-frame features.
        per_frame = []
        for t in range(lm.shape[0]):
            per_frame.append(extract_landmark_features(lm[t], handedness=None).astype(np.float32))
        per_frame = np.stack(per_frame, axis=0)
        win = _windowify(per_frame, int(args.window))
        X.append(_aggregate_window(win))
    X_arr = np.stack(X, axis=0)

    le = LabelEncoder()
    y = le.fit_transform(np.array(y_str))

    if args.group_col == "subject_id":
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
        train_idx, test_idx = next(splitter.split(X_arr, y, groups=np.array(groups)))
    else:
        # Fallback: random split
        n = X_arr.shape[0]
        rng = np.random.default_rng(int(args.seed))
        idx = rng.permutation(n)
        k = int(round(n * (1.0 - float(args.test_size))))
        train_idx, test_idx = idx[:k], idx[k:]

    clf = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_arr[train_idx], y[train_idx])

    y_pred = clf.predict(X_arr[test_idx])
    print("== Sequence model eval ==")
    print(classification_report(y[test_idx], y_pred, target_names=list(le.classes_), digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y[test_idx], y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": list(le.classes_),
            "window_size": int(args.window),
            "min_frames": int(args.min_frames),
            "smoothing_alpha": 0.30,
        },
        out_path,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

