from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.train_cnn_model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", type=str, default="", help="Folder with subfolders per class (e.g. data/collected/images_masked)")
    p.add_argument("--csv", type=str, default="", help="Optional metadata CSV with image paths and labels")
    p.add_argument("--image-col", type=str, default="auto", help="For --csv: auto | img_masked_path | img_raw_path | img_path")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune backbone too (needs more data/compute)")
    p.add_argument("--group-col", type=str, default="", help="Optional grouped split column when using --csv")
    return p.parse_args()


class CsvImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str, class_to_idx: dict[str, int], transform) -> None:
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.df.iloc[int(idx)]
        path = Path(str(row[self.image_col]))
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        y = self.class_to_idx[str(row["label"])]
        return x, y


def _pick_image_col(df: pd.DataFrame, arg: str) -> str:
    if arg != "auto":
        return arg
    for c in ("img_masked_path", "img_raw_path", "img_path"):
        if c in df.columns:
            return c
    raise SystemExit("Could not detect an image column in CSV. Expected img_masked_path, img_raw_path, or img_path.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not args.img_dir and not args.csv:
        raise SystemExit("Provide either --img-dir or --csv.")

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    if args.csv:
        df = pd.read_csv(args.csv).dropna(subset=["label"]).copy()
        image_col = _pick_image_col(df, args.image_col)
        df = df.dropna(subset=[image_col]).copy()
        df = df[df[image_col].astype(str).map(lambda p: Path(p).exists())].copy()
        labels = sorted(df["label"].astype(str).unique().tolist())
        class_to_idx = {lab: i for i, lab in enumerate(labels)}
        idx = np.arange(len(df))
        y = df["label"].map(class_to_idx).to_numpy()
        if args.group_col and args.group_col in df.columns:
            groups = df[args.group_col].astype(str).to_numpy()
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
            train_idx, val_idx = next(splitter.split(idx, y, groups=groups))
        else:
            train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y)

        ds_train = CsvImageDataset(df.iloc[train_idx].copy(), image_col, class_to_idx, train_tf)
        ds_val = CsvImageDataset(df.iloc[val_idx].copy(), image_col, class_to_idx, val_tf)
    else:
        ds_full = datasets.ImageFolder(args.img_dir)
        labels = ds_full.classes
        n = len(ds_full)
        n_val = max(1, int(0.2 * n))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(args.seed)
        ds_train, ds_val = torch.utils.data.random_split(ds_full, [n_train, n_val], generator=gen)
        ds_train.dataset.transform = train_tf
        ds_val.dataset.transform = val_tf

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = build_model(num_classes=len(labels)).to(device)

    # Fine-tune only classifier by default for small datasets.
    for name, p in model.named_parameters():
        if args.unfreeze_backbone:
            p.requires_grad = True
        else:
            p.requires_grad = name.startswith("classifier")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(dl_train, desc=f"train {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(1, total)
        print(f"val_acc: {acc:.4f}")
        best_val = max(best_val, acc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "best_val_acc": best_val,
            "smoothing_alpha": 0.25,
            "group_col": args.group_col,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
