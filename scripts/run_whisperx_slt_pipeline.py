from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an end-to-end SLT pipeline directly from WhisperX segments (clip -> target_text): "
            "subset -> export(features) -> validate -> train(proxy + optional backend) -> eval(proxy)."
        )
    )
    p.add_argument("--root", required=True)
    p.add_argument("--sources", nargs="+", default=["source2", "source3"])
    p.add_argument("--work-root", default="runs/whisperx_slt_pipeline")

    # Segment filtering
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--max-chars", type=int, default=240)
    p.add_argument("--min-duration-ms", type=int, default=700)
    p.add_argument("--max-duration-ms", type=int, default=25000)
    p.add_argument("--keep-punctuation", action="store_true")
    p.add_argument("--max-segments-per-episode", type=int, default=0)
    p.add_argument(
        "--dedup-eval-text",
        choices=["off", "train_exact", "train_val_exact"],
        default="train_exact",
        help="Drop duplicated target_text from val/test to reduce evaluation leakage (default: train_exact).",
    )

    # Export/features
    p.add_argument("--sample-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Optional cap on exported dataset rows")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing subset/export/train artifacts under --work-root when present (useful for resume).",
    )

    # Training
    p.add_argument("--epochs", type=int, default=20, help="Backend epochs (when --run-backend)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)

    # Backend (generative)
    p.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    p.add_argument("--backend-repo", default="", help="Path to neccam/slt repo. If set, we can run backend training.")
    p.add_argument("--config-base", default="", help="Optional backend config base to copy over the template")
    p.add_argument(
        "--backend-loader",
        choices=["auto", "native", "torchtext"],
        default="auto",
        help="Backend data loader selection (default: auto). Use native for modern PyTorch/MPS.",
    )
    p.add_argument("--run-backend", action="store_true")
    return p.parse_args()


def _run(cmd: list[str | Path], *, cwd: Path | None = None) -> None:
    cmd_str = [str(x) for x in cmd]
    print("+", " ".join(cmd_str))
    rc = subprocess.call(cmd_str, cwd=str(cwd) if cwd else None)
    if rc != 0:
        raise SystemExit(rc)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    subset_dir = work_root / "subset"
    export_dir = work_root / "export"
    train_dir = work_root / "train"
    results_dir = train_dir / "results"

    if args.reuse_existing:
        subset_manifest = subset_dir / "subset_manifest.csv"
        has_subset = subset_manifest.exists()
        has_clips = (subset_dir / "clips_index.csv").exists() or (subset_dir / "clips").exists()
        has_export = (export_dir / "dataset_validation.json").exists() and (export_dir / "features_package").exists()
        has_train = (train_dir / "models" / "slt_proxy.joblib").exists() or (train_dir / "external_backend_model").exists()
        print(
            f"[resume] reuse_existing=on subset={has_subset} clips={has_clips} export={has_export} train={has_train}"
        )
    else:
        has_subset = has_clips = has_export = has_train = False

    subset_cmd: list[str | Path] = [
        py,
        repo / "scripts" / "prepare_whisperx_slt_subset.py",
        "--root",
        args.root,
        "--work-dir",
        subset_dir,
        "--sources",
        *args.sources,
        "--min-words",
        args.min_words,
        "--max-words",
        args.max_words,
        "--max-chars",
        args.max_chars,
        "--min-duration-ms",
        args.min_duration_ms,
        "--max-duration-ms",
        args.max_duration_ms,
        "--max-segments-per-episode",
        args.max_segments_per_episode,
        "--export-clips",
        "--clip-ext",
        args.clip_ext,
    ]
    subset_cmd += ["--dedup-eval-text", args.dedup_eval_text]
    if args.keep_punctuation:
        subset_cmd.append("--keep-punctuation")
    if int(args.max_clips) > 0:
        subset_cmd += ["--max-clips", args.max_clips]
    if not (args.reuse_existing and has_subset and has_clips):
        _run(subset_cmd)
    else:
        print(f"[resume] skipping subset build; found {subset_dir / 'subset_manifest.csv'}")

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
        "--limit",
        args.limit,
    ]
    if args.preprocess:
        export_cmd.append("--preprocess")
    if not (args.reuse_existing and has_export):
        _run(export_cmd)
    else:
        print(f"[resume] skipping export/features; found {export_dir / 'dataset_validation.json'}")

    if not (args.reuse_existing and (export_dir / "dataset_validation.json").exists()):
        _run(
            [
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
        )
    else:
        print(f"[resume] skipping validate; found {export_dir / 'dataset_validation.json'}")

    if not (args.reuse_existing and has_train):
        _run(
            [
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
                "--backend-loader",
                args.backend_loader,
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
            + (["--preprocess"] if args.preprocess else [])
            + (["--run-backend"] if args.run_backend else [])
        )
    else:
        print(f"[resume] skipping train; found {train_dir}")

    proxy_model = train_dir / "models" / "slt_proxy.joblib"
    _run(
        [
            py,
            repo / "scripts" / "eval_ilsut_slt.py",
            "--dataset-dir",
            export_dir,
            "--model",
            proxy_model,
            "--json-out",
            results_dir / "eval_slt.json",
            "--md-out",
            results_dir / "eval_slt.md",
        ]
    )

    summary = {
        "root": str(args.root),
        "sources": list(args.sources),
        "subset": _load_json(subset_dir / "subset_info.json"),
        "dataset_validation": _load_json(export_dir / "dataset_validation.json"),
        "train_eval": _load_json(results_dir / "train_eval_summary.json"),
        "eval_proxy": _load_json(results_dir / "eval_slt.json").get("results", {}).get("slt", {}),
        "paths": {
            "subset_dir": str(subset_dir),
            "export_dir": str(export_dir),
            "train_dir": str(train_dir),
            "proxy_model": str(proxy_model),
            "eval_json": str(results_dir / "eval_slt.json"),
            "eval_md": str(results_dir / "eval_slt.md"),
            "train_eval_json": str(results_dir / "train_eval_summary.json"),
            "backend_bundle": str(export_dir / "backend" / args.backend),
            "backend_config": str(train_dir / "external_backend_config.yaml"),
            "backend_model_dir": str(train_dir / "external_backend_model"),
        },
    }
    (work_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {work_root / 'summary.json'}")


if __name__ == "__main__":
    main()
