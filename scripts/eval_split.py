from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.features import extract_landmark_features
from lsu_pria.train_cnn_model import build_model


@dataclass
class SplitConfig:
    test_size: float
    seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="data/collected/landmarks.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", type=str, default="", help="If set, do a group split by this column (e.g. subject_id)")
    p.add_argument("--landmarks-model", type=str, default="", help="models/landmarks.joblib")
    p.add_argument("--cnn-model", type=str, default="", help="models/cnn.pt")
    p.add_argument(
        "--cnn-image-col",
        type=str,
        default="auto",
        help="auto | img_masked_path | img_raw_path | img_path (legacy)",
    )
    p.add_argument("--json-out", type=str, default="", help="Write a JSON summary to this path")
    return p.parse_args()


def _split_indices(y: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    return train_idx, test_idx


def _split_indices_grouped(y: np.ndarray, groups: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    idx = np.arange(len(y))
    train_idx, test_idx = next(splitter.split(idx, y, groups=groups))
    return train_idx, test_idx


def _load_landmarks_tensor(df: pd.DataFrame) -> np.ndarray:
    feat_cols = [c for c in df.columns if c.startswith("lm_")]
    if feat_cols:
        return df[feat_cols].to_numpy(dtype=np.float32).reshape(-1, 21, 3)

    if "landmarks" in df.columns:
        rows = []
        for raw in df["landmarks"].tolist():
            vals = json.loads(str(raw))
            rows.append(np.array(vals, dtype=np.float32).reshape(21, 3))
        return np.stack(rows, axis=0)

    raise SystemExit("Could not find landmarks in CSV. Expected `lm_*` columns or a `landmarks` JSON column.")


def eval_landmarks(df: pd.DataFrame, model_path: Path, cfg: SplitConfig, group_col: str = "") -> dict:
    payload = joblib.load(model_path)
    model = payload["model"]
    labels = payload["labels"]

    X_raw = _load_landmarks_tensor(df)
    handed = df.get("handedness", pd.Series([None] * len(df))).tolist()
    X = np.stack([extract_landmark_features(X_raw[i], handed[i]) for i in range(len(df))], axis=0)

    y_str = df["label"].astype(str).to_numpy()
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_to_idx.get(v, -1) for v in y_str], dtype=np.int64)
    keep = y >= 0
    X = X[keep]
    y = y[keep]

    if group_col and group_col in df.columns:
        groups = df.loc[keep, group_col].astype(str).to_numpy()
        train_idx, test_idx = _split_indices_grouped(y, groups, cfg)
        split_name = f"group({group_col})"
    else:
        train_idx, test_idx = _split_indices(y, cfg)
        split_name = "random"
    y_pred = model.predict(X[test_idx])
    y_true = y[test_idx]

    y_pred_str = np.array([labels[i] for i in y_pred])
    y_true_str = np.array([labels[i] for i in y_true])
    macro_f1 = float(f1_score(y_true_str, y_pred_str, average="macro"))

    print("\n=== Split eval: Landmarks ===")
    print(f"split: {split_name}")
    print(classification_report(y_true_str, y_pred_str))
    print("confusion_matrix:\n", confusion_matrix(y_true_str, y_pred_str, labels=labels))
    return {"macro_f1": macro_f1, "labels": labels, "n_test": int(len(test_idx)), "split": split_name}


def _pick_image_col(df: pd.DataFrame, arg: str) -> str:
    if arg != "auto":
        return arg
    for c in ("img_masked_path", "img_raw_path", "img_path"):
        if c in df.columns:
            return c
    return ""


def eval_cnn(df: pd.DataFrame, model_path: Path, cfg: SplitConfig, image_col_arg: str, group_col: str = "") -> dict:
    ckpt = torch.load(model_path, map_location="cpu")
    labels = ckpt["labels"]
    model = build_model(num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image_col = _pick_image_col(df, image_col_arg)
    if not image_col or image_col not in df.columns:
        raise SystemExit(f"Could not find image column (requested={image_col_arg})")

    df2 = df.dropna(subset=["label", image_col]).copy()
    y_str = df2["label"].astype(str).to_numpy()
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_to_idx.get(v, -1) for v in y_str], dtype=np.int64)
    keep = y >= 0
    df2 = df2.iloc[np.where(keep)[0]]
    y = y[keep]

    if group_col and group_col in df2.columns:
        groups = df2[group_col].astype(str).to_numpy()
        train_idx, test_idx = _split_indices_grouped(y, groups, cfg)
        split_name = f"group({group_col})"
    else:
        train_idx, test_idx = _split_indices(y, cfg)
        split_name = "random"

    # Lazy import to keep dependency footprint simple.
    from PIL import Image

    y_true = []
    y_pred = []
    for i in test_idx:
        p = Path(str(df2.iloc[int(i)][image_col]))
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").resize((224, 224))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)
        with torch.no_grad():
            proba = torch.softmax(model(x), dim=1)[0]
            idx = int(torch.argmax(proba).item())
        y_true.append(int(y[int(i)]))
        y_pred.append(idx)

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_true_str = np.array([labels[i] for i in y_true])
    y_pred_str = np.array([labels[i] for i in y_pred])
    macro_f1 = float(f1_score(y_true_str, y_pred_str, average="macro"))

    print(f"\n=== Split eval: CNN ({image_col}) ===")
    print(f"split: {split_name}")
    print(classification_report(y_true_str, y_pred_str))
    print("confusion_matrix:\n", confusion_matrix(y_true_str, y_pred_str, labels=labels))
    return {"macro_f1": macro_f1, "labels": labels, "n_test": int(len(y_true)), "image_col": image_col, "split": split_name}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv).dropna(subset=["label"])
    cfg = SplitConfig(test_size=float(args.test_size), seed=int(args.seed))

    summary: dict = {
        "csv": str(args.csv),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "results": {},
    }

    if args.landmarks_model:
        res = eval_landmarks(df, Path(args.landmarks_model), cfg, group_col=args.group_col)
        summary["results"]["landmarks"] = {**res, "model": str(args.landmarks_model)}
    if args.cnn_model:
        res = eval_cnn(df, Path(args.cnn_model), cfg, args.cnn_image_col, group_col=args.group_col)
        summary["results"]["cnn"] = {**res, "model": str(args.cnn_model)}

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
