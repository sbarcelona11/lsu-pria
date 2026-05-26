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
        description="Audit iLSU-T keyword rules by showing per-label support and the matched transcript variants they actually trigger."
    )
    p.add_argument("--episodes-csv", default="auto")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _build_or_load_manifest(args: argparse.Namespace, repo: Path, py: str) -> Path:
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
    return manifest_path


def _make_payload(df: pd.DataFrame, top_k: int) -> dict:
    payload: dict[str, object] = {
        "rows": int(len(df)),
        "labels": {},
    }
    for label, part in df.groupby("label"):
        top_norm = (
            part["matched_word_norm"].fillna("").astype(str).value_counts().head(int(top_k)).rename_axis("matched_word_norm").reset_index(name="count")
        )
        top_raw = (
            part["matched_word"].fillna("").astype(str).value_counts().head(int(top_k)).rename_axis("matched_word").reset_index(name="count")
        )
        payload["labels"][str(label)] = {
            "segments": int(len(part)),
            "episodes": int(part["episode_id"].astype(str).nunique()) if "episode_id" in part.columns else None,
            "sources": int(part["source"].astype(str).nunique()) if "source" in part.columns else None,
            "top_matched_word_norm": top_norm.to_dict(orient="records"),
            "top_matched_word": top_raw.to_dict(orient="records"),
            "examples": part[["matched_word", "matched_word_norm", "episode_id", "source"]]
            .drop_duplicates()
            .head(int(top_k))
            .to_dict(orient="records"),
        }
    return payload


def _write_markdown(out_path: Path, payload: dict, top_k: int) -> None:
    lines = [
        "# iLSU-T Keywords Audit",
        "",
        f"- Manifest rows: `{payload['rows']}`",
        f"- Top variants per label: `{top_k}`",
        "",
    ]
    labels = payload.get("labels", {})
    for label, info in labels.items():
        lines += [
            f"## {label}",
            "",
            f"- Segments: `{info['segments']}`",
            f"- Episodes: `{info['episodes']}`",
            f"- Sources: `{info['sources']}`",
            "",
            "### Top normalized matches",
            "",
            "| matched_word_norm | count |",
            "|---|---:|",
        ]
        for row in info["top_matched_word_norm"]:
            lines.append(f"| {row['matched_word_norm']} | {row['count']} |")
        lines += ["", "### Top raw matches", "", "| matched_word | count |", "|---|---:|"]
        for row in info["top_matched_word"]:
            lines.append(f"| {row['matched_word']} | {row['count']} |")
        lines += ["", "### Example matches", ""]
        for row in info["examples"]:
            lines.append(
                f"- `{row.get('matched_word_norm','')}` <- `{row.get('matched_word','')}` (`{row.get('source','')}`, `{row.get('episode_id','')}`)"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    manifest_path = _build_or_load_manifest(args, repo, py)
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise SystemExit("Manifest is empty; nothing to audit.")
    if "label" not in df.columns or "matched_word" not in df.columns:
        raise SystemExit("Manifest is missing required columns: label, matched_word")
    payload = _make_payload(df, int(args.top_k))
    work_dir = Path(args.work_dir)
    json_path = work_dir / "keywords_audit.json"
    md_path = work_dir / "keywords_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload, int(args.top_k))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
