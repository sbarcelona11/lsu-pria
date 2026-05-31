from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
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
    p.add_argument(
        "--backend-loader",
        choices=["auto", "native", "torchtext"],
        default="auto",
        help="Backend data loader selection (default: auto). Use native for modern PyTorch/MPS.",
    )
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


def _find_feature_size(dataset_dir: Path) -> int:
    jsonl = dataset_dir / "features_package" / "train.jsonl"
    if not jsonl.exists():
        return 0
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = Path(str(row.get("feature_path", "")))
        if not p.exists():
            continue
        data = np.load(str(p), allow_pickle=True)
        seq = np.asarray(data["features"], dtype=np.float32)
        if seq.ndim == 2 and seq.shape[1] > 0:
            return int(seq.shape[1])
    return 0


def _latest_ckpt(model_dir: Path) -> str:
    ckpts = sorted(model_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else ""


def _write_neccam_slt_config(
    *,
    backend_repo: Path,
    backend_data_dir: Path,
    out_cfg: Path,
    model_dir: Path,
    feature_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    backend_loader: str,
    base_cfg: Path | None,
) -> dict:
    """
    Generate a SignJoey-compatible config for neccam/slt.

    We use the backend repo's `configs/sign.yaml` as a base and patch a few keys
    using line-level replacements to avoid needing PyYAML on the wrapper side.
    """
    if base_cfg is None:
        base_cfg = backend_repo / "configs" / "sign.yaml"
    if not base_cfg.exists():
        raise SystemExit(f"Missing backend base config: {base_cfg}")

    dev = str(device or "cpu").lower()
    if dev == "gpu":
        dev = "cuda"
    if dev not in {"cpu", "cuda", "mps", "auto"}:
        dev = "cpu"
    use_cuda = dev == "cuda"
    cfg_txt = base_cfg.read_text(encoding="utf-8")

    def _set_scalar(key: str, value: str) -> None:
        nonlocal cfg_txt
        lines = cfg_txt.splitlines()
        out = []
        replaced = False
        for ln in lines:
            if ln.lstrip().startswith(key + ":"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                out.append(f"{indent}{key}: {value}")
                replaced = True
            else:
                out.append(ln)
        if not replaced:
            out.append(f"{key}: {value}")
        cfg_txt = "\n".join(out)

    # Patch top-level name
    _set_scalar("name", f"ilsut_whisperx_slt_{int(time.time())}")

    # Data section keys we care about (assuming 4-space indentation in base file)
    def _set_data(key: str, value: str) -> None:
        nonlocal cfg_txt
        lines = cfg_txt.splitlines()
        out = []
        in_data = False
        replaced = False
        for ln in lines:
            if ln.startswith("data:"):
                in_data = True
                out.append(ln)
                continue
            if in_data and ln and not ln.startswith(" "):
                in_data = False
            if in_data and ln.lstrip().startswith(key + ":"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                out.append(f"{indent}{key}: {value}")
                replaced = True
            else:
                out.append(ln)
        if not replaced:
            # append under data:
            out2 = []
            inserted = False
            for ln in out:
                out2.append(ln)
                if ln.startswith("data:") and not inserted:
                    out2.append(f"    {key}: {value}")
                    inserted = True
            out = out2
        cfg_txt = "\n".join(out)

    _set_data("data_path", json.dumps(str(backend_data_dir)))  # keep quoting safe
    _set_data("train", "ilsut/ilsut.train")
    _set_data("dev", "ilsut/ilsut.val")
    _set_data("test", "ilsut/ilsut.test")
    _set_data("feature_size", str(int(feature_size)))
    _set_data("level", "word")
    _set_data("txt_lowercase", "true")
    _set_data("max_sent_length", "400")

    # Training section patches
    def _set_training(key: str, value: str) -> None:
        nonlocal cfg_txt
        lines = cfg_txt.splitlines()
        out = []
        in_training = False
        replaced = False
        for ln in lines:
            if ln.startswith("training:"):
                in_training = True
                out.append(ln)
                continue
            if in_training and ln and not ln.startswith(" "):
                in_training = False
            if in_training and ln.lstrip().startswith(key + ":"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                out.append(f"{indent}{key}: {value}")
                replaced = True
            else:
                out.append(ln)
        if not replaced:
            out2 = []
            inserted = False
            for ln in out:
                out2.append(ln)
                if ln.startswith("training:") and not inserted:
                    out2.append(f"    {key}: {value}")
                    inserted = True
            out = out2
        cfg_txt = "\n".join(out)

    _set_training("random_seed", str(int(seed)))
    _set_training("model_dir", json.dumps(str(model_dir)))
    _set_training("epochs", str(int(epochs)))
    _set_training("batch_size", str(int(batch_size)))
    _set_training("overwrite", "true")
    # Backwards compatible: newer forks may prefer training.device, older expects use_cuda.
    _set_training("device", dev)
    _set_training("use_cuda", "true" if use_cuda else "false")
    _set_training("eval_translation_beam_size", "1")
    _set_training("eval_translation_beam_alpha", "-1")

    def _set_data_loader(value: str) -> None:
        nonlocal cfg_txt
        lines = cfg_txt.splitlines()
        out = []
        in_data = False
        replaced = False
        for ln in lines:
            if ln.startswith("data:"):
                in_data = True
                out.append(ln)
                continue
            if in_data and ln and not ln.startswith(" "):
                in_data = False
            if in_data and ln.lstrip().startswith("loader:"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                out.append(f"{indent}loader: {value}")
                replaced = True
            else:
                out.append(ln)
        if not replaced:
            out2 = []
            inserted = False
            for ln in out:
                out2.append(ln)
                if ln.startswith("data:") and not inserted:
                    out2.append(f"    loader: {value}")
                    inserted = True
            out = out2
        cfg_txt = "\n".join(out)

    loader = str(backend_loader or "auto").lower().strip()
    if loader not in {"auto", "native", "torchtext"}:
        loader = "auto"
    if loader == "auto":
        if dev == "mps":
            loader = "native"
        else:
            # Prefer torchtext when available (legacy behavior), otherwise fall back to native.
            try:
                import torchtext  # noqa: F401
                loader = "torchtext"
            except Exception:
                loader = "native"
    _set_data_loader(loader)

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(cfg_txt + "\n", encoding="utf-8")
    return {
        "config_path": str(out_cfg),
        "model_dir": str(model_dir),
        "device": dev,
        "use_cuda": bool(use_cuda),
        "data_loader": loader,
        "base_config": str(base_cfg),
        "feature_size": int(feature_size),
    }


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
    print(
        f"[train] subset_dir={args.subset_dir} out_dir={args.out_dir} dataset_dir={args.dataset_dir or '(auto)'} "
        f"device={args.device} backend={args.backend} backend_loader={args.backend_loader} run_backend={bool(args.run_backend)}"
    )
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (out_dir / "dataset_export")
    if not dataset_dir.exists():
        print(f"[train] exporting dataset (features) -> {dataset_dir}")
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
    else:
        print(f"[train] using existing dataset_dir={dataset_dir}")
    validation = validate_slt_dataset_dir(dataset_dir, require_features=True)
    if not validation["valid"]:
        raise SystemExit(f"Dataset export is invalid: {validation['errors']}")
    print(f"[train] dataset valid rows={validation.get('rows')} features_rows={validation.get('feature_rows')}")

    train_x, train_texts, train_rows = _load_split_features(dataset_dir, "train")
    val_x, val_texts, val_rows = _load_split_features(dataset_dir, "val")
    test_x, test_texts, test_rows = _load_split_features(dataset_dir, "test")
    if train_x.size == 0 or not train_texts:
        raise SystemExit("No training features available under dataset export")
    feat_dim = int(train_x.shape[1]) if getattr(train_x, "ndim", 0) == 2 else 0
    print(f"[train] loaded features: train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} feat_dim={feat_dim}")

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
            model_dir = out_dir / "external_backend_model"
            backend_data_dir = backend_dir / "data"
            feature_size = _find_feature_size(dataset_dir) or 105

            base_cfg = Path(args.config_base) if args.config_base else None
            backend_report.update(
                _write_neccam_slt_config(
                    backend_repo=backend_repo,
                    backend_data_dir=backend_data_dir,
                    out_cfg=generated_cfg,
                    model_dir=model_dir,
                    feature_size=feature_size,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    device=str(args.device),
                    backend_loader=str(args.backend_loader),
                    base_cfg=base_cfg,
                )
            )
            if args.run_backend:
                print(
                    f"[train] running backend training (signjoey) device={backend_report.get('device')} "
                    f"loader={backend_report.get('data_loader')} cfg={generated_cfg}"
                )
                # Some forks support a torchtext-free native loader (needed for modern PyTorch/MPS).
                # Always require PyYAML; require torchtext only when using the legacy loader.
                try:
                    import yaml  # noqa: F401
                    import portalocker  # noqa: F401
                    need_torchtext = str(backend_report.get("device") or str(args.device)).lower() not in {"mps"}
                    if need_torchtext:
                        import torchtext  # noqa: F401
                except Exception as e:
                    backend_report["status"] = "missing_backend_deps"
                    backend_report["error"] = (
                        "Missing deps for neccam/slt backend. "
                        "Install pyyaml + portalocker, and torchtext if you're using the legacy torchtext loader."
                    )
                    raise SystemExit(backend_report["error"]) from e
                rc = _run([py, "-m", "signjoey", "train", generated_cfg], cwd=backend_repo)
                backend_report["status"] = "ok" if rc == 0 else "failed"
                backend_report["return_code"] = int(rc)
                if rc == 0:
                    backend_report["ckpt_latest"] = _latest_ckpt(model_dir)
            else:
                backend_report["status"] = "packaged"
        else:
            backend_report["status"] = "missing_repo"

    print("[train] training proxy baseline (KNN)")
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
