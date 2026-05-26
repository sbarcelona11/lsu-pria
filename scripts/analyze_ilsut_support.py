from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build an iLSU-T weak-label manifest and summarize label support to decide which classes are trainable."
    )
    p.add_argument("--episodes-csv", default="auto")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--min-label-count", type=int, default=20)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _write_markdown(
    out_path: Path,
    counts: pd.DataFrame,
    recommended: pd.DataFrame,
    min_label_count: int,
    sources: list[str],
    manifest_rows: int,
) -> None:
    lines = [
        "# iLSU-T Label Support",
        "",
        f"- Sources: `{', '.join(sources) if sources else 'all'}`",
        f"- Manifest rows: `{manifest_rows}`",
        f"- Recommended minimum label count: `{min_label_count}`",
        "",
        "## Label counts",
        "",
        "| label | segments | episodes | sources | recommended |",
        "|---|---:|---:|---:|---|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.segments} | {row.episodes} | {row.sources} | {'yes' if row.segments >= min_label_count else 'no'} |"
        )
    lines += [
        "",
        "## Recommended classes",
        "",
    ]
    if recommended.empty:
        lines.append("_No classes reach the requested support threshold._")
    else:
        for row in recommended.itertuples(index=False):
            lines.append(f"- `{row.label}`: {row.segments} segments, {row.episodes} episodes, {row.sources} sources")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv = args.episodes_csv
    if not episodes_csv or str(episodes_csv).strip().lower() == "auto":
        episodes_csv = work_dir / "episodes_generated.csv"
        build_cmd = [py, repo / "scripts" / "build_ilsut_episodes_csv.py", "--root", args.root, "--out", episodes_csv]
        if args.sources:
            build_cmd += ["--sources", *args.sources]
        _run(build_cmd)

    manifest_path = work_dir / "manifest.csv"
    _run(
        [
            py,
            repo / "scripts" / "ilsut_make_manifest.py",
            "--episodes-csv",
            episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--out",
            manifest_path,
            "--path-mode",
            args.path_mode,
            "--limit",
            args.manifest_limit,
        ]
    )

    df = pd.read_csv(manifest_path)
    if df.empty:
        raise SystemExit("Manifest is empty; no weak-labeled segments were found.")

    if "label" not in df.columns:
        raise SystemExit("Manifest does not contain a 'label' column.")

    group_col = "group_id" if "group_id" in df.columns else ("episode_id" if "episode_id" in df.columns else None)
    source_col = "source" if "source" in df.columns else None

    rows: list[dict[str, object]] = []
    for label, part in df.groupby("label"):
        rows.append(
            {
                "label": str(label),
                "segments": int(len(part)),
                "episodes": int(part[group_col].nunique()) if group_col else int(len(part)),
                "sources": int(part[source_col].nunique()) if source_col else 1,
            }
        )
    counts = pd.DataFrame(rows).sort_values(["segments", "episodes", "label"], ascending=[False, False, True]).reset_index(drop=True)
    recommended = counts[counts["segments"] >= int(args.min_label_count)].copy().reset_index(drop=True)

    counts_path = work_dir / "label_support.csv"
    recommended_path = work_dir / "recommended_labels.json"
    markdown_path = work_dir / "label_support.md"
    counts.to_csv(counts_path, index=False)
    recommended_payload = {
        "min_label_count": int(args.min_label_count),
        "sources": list(args.sources or []),
        "manifest_rows": int(len(df)),
        "labels": recommended["label"].tolist(),
        "counts": recommended.to_dict(orient="records"),
    }
    recommended_path.write_text(json.dumps(recommended_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, counts, recommended, int(args.min_label_count), list(args.sources or []), int(len(df)))
    print(f"Wrote: {counts_path}")
    print(f"Wrote: {recommended_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()
