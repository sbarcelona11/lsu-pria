from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.slt_features import aggregate_sequence_embedding, bleu_like, normalize_slt_text, token_overlap_score
from _slt_dataset_utils import validate_slt_dataset_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an iLSU-T SLT wrapper package and a local proxy baseline.")
    p.add_argument("--subset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-dir", default="", help="Optional pre-exported dataset directory")
    p.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    p.add_argument("--backend-repo", default="")
    p.add_argument("--config-base", default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--run-backend", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print("+", " ".join(str(x) for x in cmd))
    return subprocess.call([str(x) for x in cmd], cwd=str(cwd) if cwd else None)


def _load_split_features(dataset_dir: Path, split: str) -> tuple[np.ndarray, list[str], list[dict]]:
    jsonl = dataset_dir / "features_package" / f"{split}.jsonl"
    if not jsonl.exists():
        return np.zeros((0, 0), dtype=np.float32), [], []
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    embs: list[np.ndarray] = []
    texts: list[str] = []
    for row in rows:
        feature_path = Path(str(row["feature_path"]))
        data = np.load(feature_path, allow_pickle=True)
        seq = np.asarray(data["features"], dtype=np.float32)
        embs.append(aggregate_sequence_embedding(seq, min_frames=4))
        texts.append(str(row.get("target_text", "")))
    if not embs:
        return np.zeros((0, 0), dtype=np.float32), [], rows
    return np.stack(embs, axis=0), texts, rows


def _train_proxy(train_x: np.ndarray, train_texts: list[str], seed: int) -> tuple[object, list[str]]:
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(train_texts))
    model = KNeighborsClassifier(n_neighbors=min(5, max(1, len(train_texts))), weights="distance")
    model.fit(train_x, y)
    return model, list(le.classes_)


def _predict_bundle(model, labels: list[str], x: np.ndarray) -> tuple[list[str], list[float]]:
    if x.size == 0:
        return [], []
    proba = model.predict_proba(x)
    idx = np.argmax(proba, axis=1)
    texts = [labels[int(i)] for i in idx]
    conf = [float(proba[n, int(i)]) for n, i in enumerate(idx)]
    return texts, conf


def _summarize_eval(refs: list[str], preds: list[str], confs: list[float]) -> dict:
    exact = [1.0 if normalize_slt_text(r) == normalize_slt_text(p) else 0.0 for r, p in zip(refs, preds)]
    overlap = [token_overlap_score(r, p) for r, p in zip(refs, preds)]
    bleu = [bleu_like(r, p) for r, p in zip(refs, preds)]
    return {
        "n": len(refs),
        "exact_match_rate": float(np.mean(exact)) if exact else 0.0,
        "token_overlap": float(np.mean(overlap)) if overlap else 0.0,
        "bleu_like": float(np.mean(bleu)) if bleu else 0.0,
        "avg_confidence": float(np.mean(confs)) if confs else 0.0,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (out_dir / "dataset_export")
    if not dataset_dir.exists():
        cmd = [
            py,
            repo / "scripts" / "export_ilsut_slt_dataset.py",
            "--subset-dir",
            args.subset_dir,
            "--out-dir",
            dataset_dir,
            "--mode",
            "features",
            "--sample-fps",
            args.sample_fps,
            "--max-frames",
            args.max_frames,
            "--backend",
            args.backend,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        rc = _run(cmd)
        if rc != 0:
            raise SystemExit(rc)
    validation = validate_slt_dataset_dir(dataset_dir, require_features=True)
    if not validation["valid"]:
        raise SystemExit(f"Dataset export is invalid: {validation['errors']}")

    train_x, train_texts, train_rows = _load_split_features(dataset_dir, "train")
    val_x, val_texts, val_rows = _load_split_features(dataset_dir, "val")
    test_x, test_texts, test_rows = _load_split_features(dataset_dir, "test")
    if train_x.size == 0 or not train_texts:
        raise SystemExit("No training features available under dataset export")

    backend_report = {
        "backend": args.backend,
        "backend_repo": args.backend_repo,
        "run_backend": bool(args.run_backend),
        "status": "not_run",
    }
    if args.backend_repo:
        backend_repo = Path(args.backend_repo)
        if backend_repo.exists():
            backend_dir = dataset_dir / "backend" / args.backend
            generated_cfg = out_dir / "external_backend_config.yaml"
            src_cfg = Path(args.config_base) if args.config_base else (backend_dir / "config_template.yaml")
            if src_cfg.exists():
                shutil.copyfile(src_cfg, generated_cfg)
            if args.run_backend:
                rc = _run([py, "-m", "signjoey", "train", generated_cfg], cwd=backend_repo)
                backend_report["status"] = "ok" if rc == 0 else "failed"
                backend_report["return_code"] = int(rc)
            else:
                backend_report["status"] = "packaged"
        else:
            backend_report["status"] = "missing_repo"

    proxy_model, labels = _train_proxy(train_x, train_texts, int(args.seed))
    val_preds, val_confs = _predict_bundle(proxy_model, labels, val_x)
    test_preds, test_confs = _predict_bundle(proxy_model, labels, test_x)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = models_dir / "slt_proxy.joblib"
    joblib.dump(
        {
            "model": proxy_model,
            "labels": labels,
            "sample_fps": float(args.sample_fps),
            "max_frames": int(args.max_frames),
            "min_frames": 4,
            "preprocess": bool(args.preprocess),
            "model_type": "proxy_knn",
        },
        proxy_path,
    )

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_summary = {
        "val": _summarize_eval(val_texts, val_preds, val_confs),
        "test": _summarize_eval(test_texts, test_preds, test_confs),
        "labels": labels,
        "proxy_model": str(proxy_path),
        "backend_report": backend_report,
        "dataset_validation": validation,
    }
    (results_dir / "train_eval_summary.json").write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_lines = [
        "# SLT Training Report",
        "",
        f"- Backend: `{args.backend}`",
        f"- Backend status: `{backend_report['status']}`",
        f"- Proxy model: `{proxy_path}`",
        f"- Train samples: `{len(train_rows)}`",
        f"- Val exact/token/BLEU-like: `{eval_summary['val']['exact_match_rate']:.3f}` / `{eval_summary['val']['token_overlap']:.3f}` / `{eval_summary['val']['bleu_like']:.3f}`",
        f"- Test exact/token/BLEU-like: `{eval_summary['test']['exact_match_rate']:.3f}` / `{eval_summary['test']['token_overlap']:.3f}` / `{eval_summary['test']['bleu_like']:.3f}`",
        "",
        "This phase keeps the repo usable by training a local proxy SLT baseline while packaging an external backend bundle.",
        "",
    ]
    (results_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote: {proxy_path}")
    print(f"Wrote: {results_dir / 'train_eval_summary.json'}")
    print(f"Wrote: {results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
