from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


@dataclass
class Row:
    name: str
    macro_f1: float
    fps: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results/table.md")
    p.add_argument("--rows", nargs="+", required=True, help="Triples: name,macro_f1,fps (repeat)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.rows) % 3 != 0:
        raise SystemExit("rows must be multiples of 3: name macro_f1 fps")
    rows = []
    for i in range(0, len(args.rows), 3):
        rows.append(Row(name=args.rows[i], macro_f1=float(args.rows[i + 1]), fps=float(args.rows[i + 2])))

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["macro_f1", "fps"], ascending=False)
    md = df.to_markdown(index=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
