from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-json", type=str, default="results/ablation_results.json")
    p.add_argument("--ablation-table", type=str, default="results/ablation_table.md")
    p.add_argument("--dataset-stats-md", type=str, default="results/dataset_stats.md")
    p.add_argument("--out-md", type=str, default="results/report.md")
    p.add_argument("--out-fig", type=str, default="results/precision_vs_fps.png")
    return p.parse_args()


def _read_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> None:
    args = parse_args()
    ablation_json = Path(args.ablation_json)
    ablation_table = Path(args.ablation_table)
    dataset_stats = Path(args.dataset_stats_md)
    out_md = Path(args.out_md)
    out_fig = Path(args.out_fig)

    # Plot if we have ablation json.
    fig_rel = out_fig.as_posix()
    if ablation_json.exists():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/plot_ablation.py", "--ablation-json", str(ablation_json), "--out", str(out_fig)],
            check=True,
        )

    ablation_table_md = _read_if_exists(ablation_table).strip()
    dataset_stats_md = _read_if_exists(dataset_stats).strip()

    meta = {}
    if ablation_json.exists():
        meta = json.loads(ablation_json.read_text(encoding="utf-8"))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("# lsu-pria — Reporte rápido\n")
    out.append("## Dataset\n")
    if dataset_stats_md:
        out.append(dataset_stats_md + "\n")
    else:
        out.append("_No dataset stats found. Run `scripts/dataset_stats.py` first._\n")

    out.append("## Ablation (precisión vs FPS)\n")
    if ablation_table_md:
        out.append(ablation_table_md + "\n")
    else:
        out.append("_No ablation table found. Run `scripts/run_ablation.py` first._\n")

    if out_fig.exists():
        out.append(f"![precision_vs_fps]({fig_rel})\n")

    if meta.get("fps_config"):
        out.append("### FPS config\n")
        out.append("```json\n" + json.dumps(meta["fps_config"], indent=2, ensure_ascii=False) + "\n```\n")

    out_md.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
