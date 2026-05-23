from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="landmarks.csv")
    p.add_argument("--out-md", type=str, default="", help="Optional markdown output path")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    p.add_argument("--by", choices=["label", "label_subject"], default="label_subject")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["label"])
    if "subject_id" not in df.columns:
        df["subject_id"] = "S?"

    if args.by == "label":
        table = df.groupby(["label"]).size().reset_index(name="n").sort_values("n", ascending=False)
    else:
        table = (
            df.groupby(["label", "subject_id"])
            .size()
            .reset_index(name="n")
            .sort_values(["label", "subject_id"])
        )

    print(table.to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"Wrote: {out}")

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

