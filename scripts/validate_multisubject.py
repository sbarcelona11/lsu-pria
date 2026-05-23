from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="merged landmarks.csv")
    p.add_argument("--min-per-label-per-subject", type=int, default=30)
    p.add_argument("--out-md", type=str, default="", help="Optional markdown output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv).dropna(subset=["label"])
    if "subject_id" not in df.columns:
        raise SystemExit("CSV missing subject_id. Collect with scripts/collect_data.py --subject-id ... and merge.")

    counts = df.groupby(["subject_id", "label"]).size().reset_index(name="n")
    bad = counts[counts["n"] < int(args.min_per_label_per_subject)].sort_values(["subject_id", "n"])

    summary = {
        "subjects": int(df["subject_id"].nunique()),
        "labels": int(df["label"].nunique()),
        "rows": int(len(df)),
        "min_per_label_per_subject": int(args.min_per_label_per_subject),
        "violations": int(len(bad)),
    }

    print("summary:", summary)
    if len(bad):
        print("\nViolations (need more samples):")
        print(bad.to_string(index=False))
    else:
        print("\nOK: all subject/label pairs meet the minimum.")

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Multi-subject validation\n")
        lines.append("```json\n")
        import json

        lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
        lines.append("\n```\n")
        if len(bad):
            lines.append("## Violations\n")
            lines.append(bad.to_markdown(index=False))
            lines.append("")
        else:
            lines.append("OK: no violations.\n")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

