from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the end-to-end iLSU-T SLT pipeline: support -> subset -> export -> validate -> train -> eval.")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", default="deliverables/ilsut_keywords.focused.json")
    p.add_argument("--sources", nargs="+", default=["source2", "source3"])
    p.add_argument("--work-root", default="runs/ilsut_slt_pipeline")
    p.add_argument("--preset", choices=["custom", "quick", "standard"], default="custom")
    p.add_argument("--episodes-csv", default="auto")
    p.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--min-label-count", type=int, default=20)
    p.add_argument("--labels-json", default="")
    p.add_argument("--sample-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    p.add_argument("--backend-repo", default="")
    p.add_argument("--config-base", default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-backend", action="store_true")
    return p.parse_args()


def _run(cmd: list[str | Path]) -> None:
    cmd_str = [str(x) for x in cmd]
    print("+", " ".join(cmd_str))
    rc = subprocess.call(cmd_str)
    if rc != 0:
        raise SystemExit(rc)


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "quick":
        if int(args.manifest_limit) == 0:
            args.manifest_limit = 12
        if int(args.min_label_count) == 20:
            args.min_label_count = 20
        if float(args.sample_fps) == 6.0:
            args.sample_fps = 4.0
        if int(args.max_frames) == 48:
            args.max_frames = 8
        if int(args.max_clips) == 0:
            args.max_clips = 24
        if int(args.epochs) == 10:
            args.epochs = 3
        if int(args.batch_size) == 16:
            args.batch_size = 8
        return
    if args.preset == "standard":
        if int(args.min_label_count) == 20:
            args.min_label_count = 20
        if float(args.sample_fps) == 6.0:
            args.sample_fps = 6.0
        if int(args.max_frames) == 48:
            args.max_frames = 48
        if int(args.epochs) == 10:
            args.epochs = 10
        if int(args.batch_size) == 16:
            args.batch_size = 16


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(work_root: Path, args: argparse.Namespace, labels_json: str, support_dir: Path, subset_dir: Path, export_dir: Path, train_dir: Path) -> None:
    results_dir = train_dir / "results"
    support = _load_json(support_dir / "recommended_labels.json")
    subset = _load_json(subset_dir / "subset_info.json")
    validation = _load_json(export_dir / "dataset_validation.json")
    train_eval = _load_json(results_dir / "train_eval_summary.json")
    eval_slt = _load_json(results_dir / "eval_slt.json")
    summary = {
        "root": str(args.root),
        "sources": list(args.sources),
        "preset": str(args.preset),
        "keywords": str(args.keywords),
        "labels_json": str(labels_json),
        "backend": str(args.backend),
        "backend_repo": str(args.backend_repo or ""),
        "support": {
            "recommended_labels": support.get("labels", []),
            "manifest_rows": support.get("manifest_rows"),
        },
        "subset": {
            "rows": subset.get("rows"),
            "labels": subset.get("labels", []),
            "split_counts": subset.get("split_counts", {}),
            "split_episode_counts": subset.get("split_episode_counts", {}),
        },
        "dataset_validation": {
            "valid": validation.get("valid"),
            "rows": validation.get("rows"),
            "split_counts": validation.get("split_counts", {}),
            "group_overlap": validation.get("group_overlap", []),
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
        },
        "train_eval": train_eval,
        "eval_slt": eval_slt.get("results", {}).get("slt", {}),
        "paths": {
            "support_dir": str(support_dir),
            "subset_dir": str(subset_dir),
            "export_dir": str(export_dir),
            "train_dir": str(train_dir),
            "proxy_model": str(train_dir / "models" / "slt_proxy.joblib"),
            "eval_json": str(results_dir / "eval_slt.json"),
        },
    }
    summary_json = work_root / "summary.json"
    summary_md = work_root / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_lines = [
        "# iLSU-T SLT Pipeline Summary",
        "",
        f"- Sources: `{', '.join(args.sources)}`",
        f"- Preset: `{args.preset}`",
        f"- Keywords: `{args.keywords}`",
        f"- Recommended labels: `{', '.join(summary['support']['recommended_labels']) if summary['support']['recommended_labels'] else '(none)'}`",
        f"- Subset rows: `{summary['subset']['rows']}`",
        f"- Split counts: `{summary['subset']['split_counts']}`",
        f"- Dataset valid: `{summary['dataset_validation']['valid']}`",
        f"- Eval exact/token/BLEU-like: `{summary['eval_slt'].get('exact_match_rate', 0.0):.3f}` / `{summary['eval_slt'].get('token_overlap', 0.0):.3f}` / `{summary['eval_slt'].get('bleu_like', 0.0):.3f}`",
        f"- Avg confidence: `{summary['eval_slt'].get('avg_confidence', 0.0):.3f}`",
        f"- Backend status: `{train_eval.get('backend_report', {}).get('status', 'unknown')}`",
        "",
        "## Key paths",
        "",
        f"- Support: `{support_dir}`",
        f"- Subset: `{subset_dir}`",
        f"- Export: `{export_dir}`",
        f"- Train: `{train_dir}`",
        f"- Proxy model: `{train_dir / 'models' / 'slt_proxy.joblib'}`",
        f"- Eval JSON: `{results_dir / 'eval_slt.json'}`",
        "",
    ]
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _apply_preset(args)
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    support_dir = work_root / "support"
    subset_dir = work_root / "subset"
    export_dir = work_root / "export"
    train_dir = work_root / "train"
    results_dir = train_dir / "results"

    _run(
        [
            py,
            repo / "scripts" / "analyze_ilsut_support.py",
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            support_dir,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
            "--sources",
            *args.sources,
        ]
    )

    labels_json = args.labels_json or str(support_dir / "recommended_labels.json")
    subset_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "prepare_ilsut_slt_subset.py",
        "--episodes-csv",
        args.episodes_csv,
        "--root",
        args.root,
        "--keywords",
        args.keywords,
        "--work-dir",
        subset_dir,
        "--path-mode",
        args.path_mode,
        "--manifest-limit",
        args.manifest_limit,
        "--min-label-count",
        args.min_label_count,
        "--labels-json",
        labels_json,
        "--export-clips",
        "--clip-ext",
        args.clip_ext,
        "--sources",
        *args.sources,
    ]
    if int(args.max_clips) > 0:
        subset_cmd += ["--max-clips", args.max_clips]
    _run(subset_cmd)

    export_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "export_ilsut_slt_dataset.py",
        "--subset-dir",
        subset_dir,
        "--out-dir",
        export_dir,
        "--mode",
        "features",
        "--sample-fps",
        args.sample_fps,
        "--max-frames",
        args.max_frames,
        "--clip-ext",
        args.clip_ext,
        "--backend",
        args.backend,
    ]
    if args.preprocess:
        export_cmd.append("--preprocess")
    _run(export_cmd)

    validate_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "validate_ilsut_slt_dataset.py",
        "--dataset-dir",
        export_dir,
        "--json-out",
        export_dir / "dataset_validation.json",
        "--md-out",
        export_dir / "dataset_validation.md",
        "--require-features",
    ]
    _run(validate_cmd)

    train_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "train_ilsut_slt.py",
        "--subset-dir",
        subset_dir,
        "--out-dir",
        train_dir,
        "--dataset-dir",
        export_dir,
        "--backend",
        args.backend,
        "--backend-repo",
        args.backend_repo,
        "--config-base",
        args.config_base,
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--device",
        args.device,
        "--seed",
        args.seed,
        "--sample-fps",
        args.sample_fps,
        "--max-frames",
        args.max_frames,
    ]
    if args.preprocess:
        train_cmd.append("--preprocess")
    if args.run_backend:
        train_cmd.append("--run-backend")
    _run(train_cmd)

    eval_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "eval_ilsut_slt.py",
        "--dataset-dir",
        export_dir,
        "--model",
        train_dir / "models" / "slt_proxy.joblib",
        "--json-out",
        results_dir / "eval_slt.json",
        "--md-out",
        results_dir / "eval_slt.md",
    ]
    _run(eval_cmd)

    _write_summary(work_root, args, labels_json, support_dir, subset_dir, export_dir, train_dir)

    print(f"Prepared support: {support_dir}")
    print(f"Prepared subset: {subset_dir}")
    print(f"Prepared export: {export_dir}")
    print(f"Prepared training: {train_dir}")
    print(f"Prepared evaluation: {results_dir / 'eval_slt.json'}")
    print(f"Prepared summary: {work_root / 'summary.json'}")


if __name__ == "__main__":
    main()
