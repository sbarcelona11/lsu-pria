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

from lsu_pria.features import extract_multimodal_frame_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multimodal temporal baseline (hands + pose + face).")
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="group_id")
    return p.parse_args()


def _aggregate_window(window_feats: np.ndarray) -> np.ndarray:
    mean = window_feats.mean(axis=0)
    std = window_feats.std(axis=0)
    delta = window_feats[-1] - window_feats[0]
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)


def _windowify(seq_feats: np.ndarray, window: int) -> np.ndarray:
    if seq_feats.shape[0] >= window:
        return seq_feats[-window:]
    pad = np.repeat(seq_feats[:1], window - seq_feats.shape[0], axis=0)
    return np.concatenate([pad, seq_feats], axis=0)


def _load_sequences(seq_root: Path):
    seqs: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    subjects: list[str] = []
    for p in sorted(seq_root.rglob("*.npz")):
        try:
            d = np.load(str(p), allow_pickle=True)
        except Exception:
            continue
        left = np.array(d.get("left_hand"), dtype=np.float32)
        right = np.array(d.get("right_hand"), dtype=np.float32)
        pose = np.array(d.get("pose"), dtype=np.float32)
        face = np.array(d.get("face"), dtype=np.float32)
        lab = d.get("label")
        if lab is None or left.ndim != 3 or right.ndim != 3 or pose.ndim != 3 or face.ndim != 3:
            continue
        T = min(left.shape[0], right.shape[0], pose.shape[0], face.shape[0])
        if T <= 0:
            continue
        feats = []
        for i in range(T):
            lf = left[i] if np.any(left[i]) else None
            rf = right[i] if np.any(right[i]) else None
            pf = pose[i] if np.any(pose[i]) else None
            ff = face[i] if np.any(face[i]) else None
            feats.append(extract_multimodal_frame_features(lf, rf, pf, ff))
        seqs.append(np.stack(feats, axis=0))
        labels.append(str(lab))
        groups.append(str(d.get("group_id", d.get("subject_id", "G?"))))
        subjects.append(str(d.get("subject_id", "S?")))
    if not seqs:
        raise SystemExit(f"No multimodal sequences found under: {seq_root}")
    return seqs, labels, groups, subjects


def main() -> None:
    args = parse_args()
    seq_root = Path(args.seq_dir)
    if (seq_root / "multimodal_sequences").exists():
        seq_root = seq_root / "multimodal_sequences"

    seqs, y_str, groups, subjects = _load_sequences(seq_root)
    X = np.stack([_aggregate_window(_windowify(seq, int(args.window))) for seq in seqs], axis=0)

    le = LabelEncoder()
    y = le.fit_transform(np.array(y_str))
    if args.group_col == "group_id":
        groups_arr = np.array(groups)
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
        train_idx, test_idx = next(splitter.split(X, y, groups=groups_arr))
    elif args.group_col == "subject_id":
        groups_arr = np.array(subjects)
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
        train_idx, test_idx = next(splitter.split(X, y, groups=groups_arr))
    else:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.permutation(X.shape[0])
        k = int(round(X.shape[0] * (1.0 - float(args.test_size))))
        train_idx, test_idx = idx[:k], idx[k:]

    clf = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    print("== Multimodal sequence model eval ==")
    labels_all = list(range(len(le.classes_)))
    print(
        classification_report(
            y[test_idx],
            y_pred,
            labels=labels_all,
            target_names=list(le.classes_),
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y[test_idx], y_pred, labels=labels_all))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": list(le.classes_),
            "window_size": int(args.window),
            "min_frames": int(args.min_frames),
            "smoothing_alpha": 0.30,
            "feature_type": "multimodal_sequence",
        },
        out_path,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
