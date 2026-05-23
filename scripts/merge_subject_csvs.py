from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="One or more landmarks.csv paths")
    p.add_argument("--out", type=str, required=True, help="Output merged CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"Missing: {path}")
        df = pd.read_csv(path)
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if "subject_id" not in merged.columns:
        merged["subject_id"] = "S?"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote: {out} (rows={len(merged)})")


if __name__ == "__main__":
    main()
