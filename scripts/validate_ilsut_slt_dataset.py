from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from _slt_dataset_utils import validate_slt_dataset_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an exported iLSU-T SLT dataset package.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--json-out", default="")
    p.add_argument("--md-out", default="")
    p.add_argument("--require-features", action="store_true")
    return p.parse_args()


def _markdown(report: dict) -> str:
    lines = [
        "# SLT Dataset Validation",
        "",
        f"- Dataset dir: `{report['dataset_dir']}`",
        f"- Valid: `{report['valid']}`",
        f"- Rows: `{report['rows']}`",
        f"- Splits: `{report['split_counts']}`",
        f"- Split group overlap count: `{len(report['group_overlap'])}`",
        f"- Empty target_text rows: `{len(report['empty_target_text'])}`",
        f"- Missing clip paths: `{len(report['missing_clip_paths'])}`",
        f"- Missing feature paths: `{len(report['missing_feature_paths'])}`",
        "",
    ]
    if report["group_overlap"]:
        lines += ["## Group overlaps", ""] + [f"- `{item}`" for item in report["group_overlap"]] + [""]
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- {item}" for item in report["warnings"]] + [""]
    if report["errors"]:
        lines += ["## Errors", ""] + [f"- {item}" for item in report["errors"]] + [""]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = validate_slt_dataset_dir(Path(args.dataset_dir), require_features=bool(args.require_features))
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote: {json_path}")
    if args.md_out:
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown(report), encoding="utf-8")
        print(f"Wrote: {md_path}")
    if not report["valid"]:
        raise SystemExit(1)
    print("SLT dataset export is valid.")


if __name__ == "__main__":
    main()
