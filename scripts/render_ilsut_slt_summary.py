from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render report/pitch-friendly artifacts from an iLSU-T SLT summary.json.")
    p.add_argument("--summary-json", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(summary.get("support", {}).get("recommended_labels", []))
    split_counts = summary.get("subset", {}).get("split_counts", {})
    split_episode_counts = summary.get("subset", {}).get("split_episode_counts", {})
    eval_slt = summary.get("eval_slt", {})
    validation = summary.get("dataset_validation", {})
    train_eval = summary.get("train_eval", {})

    overview = {
        "sources": ", ".join(summary.get("sources", [])),
        "preset": summary.get("preset", ""),
        "recommended_labels": ", ".join(labels),
        "subset_rows": summary.get("subset", {}).get("rows", 0),
        "train_rows": split_counts.get("train", 0),
        "val_rows": split_counts.get("val", 0),
        "test_rows": split_counts.get("test", 0),
        "train_episodes": split_episode_counts.get("train", 0),
        "val_episodes": split_episode_counts.get("val", 0),
        "test_episodes": split_episode_counts.get("test", 0),
        "dataset_valid": validation.get("valid", False),
        "exact_match_rate": eval_slt.get("exact_match_rate", 0.0),
        "token_overlap": eval_slt.get("token_overlap", 0.0),
        "bleu_like": eval_slt.get("bleu_like", 0.0),
        "avg_confidence": eval_slt.get("avg_confidence", 0.0),
        "backend_status": train_eval.get("backend_report", {}).get("status", "unknown"),
    }

    metrics_rows = [
        {"metric": "exact_match_rate", "value": overview["exact_match_rate"]},
        {"metric": "token_overlap", "value": overview["token_overlap"]},
        {"metric": "bleu_like", "value": overview["bleu_like"]},
        {"metric": "avg_confidence", "value": overview["avg_confidence"]},
    ]
    split_rows = [
        {
            "split": split_name,
            "rows": split_counts.get(split_name, 0),
            "episodes": split_episode_counts.get(split_name, 0),
        }
        for split_name in ("train", "val", "test")
    ]

    md_lines = [
        "# Resumen SLT iLSU-T",
        "",
        f"- Fuentes: `{overview['sources']}`",
        f"- Preset: `{overview['preset']}`",
        f"- Labels recomendadas: `{overview['recommended_labels'] or '(ninguna)'}`",
        f"- Filas subset: `{overview['subset_rows']}`",
        f"- Dataset válido: `{overview['dataset_valid']}`",
        f"- Estado backend: `{overview['backend_status']}`",
        "",
        "## Splits",
        "",
        "| split | rows | episodes |",
        "|---|---:|---:|",
    ]
    for row in split_rows:
        md_lines.append(f"| {row['split']} | {row['rows']} | {row['episodes']} |")
    md_lines += [
        "",
        "## Métricas SLT",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in metrics_rows:
        md_lines.append(f"| {row['metric']} | {float(row['value']):.3f} |")
    if validation.get("warnings"):
        md_lines += ["", "## Warnings", ""] + [f"- {item}" for item in validation["warnings"]]
    if validation.get("errors"):
        md_lines += ["", "## Errors", ""] + [f"- {item}" for item in validation["errors"]]
    md_lines.append("")

    (out_dir / "overview.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(out_dir / "metrics.csv", metrics_rows)
    _write_csv(out_dir / "splits.csv", split_rows)
    (out_dir / "summary_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote: {out_dir / 'overview.json'}")
    print(f"Wrote: {out_dir / 'metrics.csv'}")
    print(f"Wrote: {out_dir / 'splits.csv'}")
    print(f"Wrote: {out_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
