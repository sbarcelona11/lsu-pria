from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.features import extract_landmark_features
from lsu_pria.train_cnn_model import build_model


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="landmarks.csv produced by collect_data.py")
    p.add_argument("--landmarks-model", type=str, default="", help="models/landmarks.joblib")
    p.add_argument("--cnn-model", type=str, default="", help="models/cnn.pt")
    return p.parse_args()


def eval_landmarks(csv_path: Path, model_path: Path) -> None:
    payload = joblib.load(model_path)
    model = payload["model"]
    labels = payload["labels"]

    df = pd.read_csv(csv_path).dropna(subset=["label"])
    X_raw = _load_landmarks_tensor(df)
    handed = df.get("handedness", pd.Series([None] * len(df))).tolist()
    X = np.stack([extract_landmark_features(X_raw[i], handed[i]) for i in range(len(df))], axis=0)
    y_true = df["label"].to_numpy()
    y_pred = np.array([labels[i] for i in model.predict(X)])
    print("\n=== Landmarks model ===")
    print(classification_report(y_true, y_pred))
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred, labels=labels))


def eval_cnn(csv_path: Path, model_path: Path) -> None:
    ckpt = torch.load(model_path, map_location="cpu")
    labels = ckpt["labels"]
    model = build_model(num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = pd.read_csv(csv_path)
    if "img_masked_path" in df.columns:
        img_col = "img_masked_path"
    elif "img_raw_path" in df.columns:
        img_col = "img_raw_path"
    else:
        img_col = "img_path" if "img_path" in df.columns else ""
    if not img_col:
        raise SystemExit("No image path column found (expected img_masked_path, img_raw_path, or img_path)")
    df = df.dropna(subset=["label", img_col])
    y_true = df["label"].to_numpy()
    y_pred = []
    for p in df[img_col].to_list():
        img = Path(p)
        if not img.exists():
            continue
        from PIL import Image

        im = Image.open(img).convert("RGB").resize((224, 224))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)
        with torch.no_grad():
            proba = torch.softmax(model(x), dim=1)[0]
            idx = int(torch.argmax(proba).item())
            y_pred.append(labels[idx])

    y_pred = np.array(y_pred)
    y_true = y_true[: len(y_pred)]
    print("\n=== CNN model ===")
    print(classification_report(y_true, y_pred))
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred, labels=labels))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if args.landmarks_model:
        eval_landmarks(csv_path, Path(args.landmarks_model))
    if args.cnn_model:
        eval_cnn(csv_path, Path(args.cnn_model))


if __name__ == "__main__":
    main()
