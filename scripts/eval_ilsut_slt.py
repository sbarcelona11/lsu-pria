from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.pipelines.slt import SltPipeline
from lsu_pria.slt_features import aggregate_sequence_embedding, bleu_like, normalize_slt_text, token_overlap_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an SLT model bundle on an exported iLSU-T test split.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out", default="")
    p.add_argument("--landmarks-eval-json", default="")
    p.add_argument("--cnn-eval-json", default="")
    p.add_argument("--multimodal-eval-json", default="")
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_baseline_macro(path_str: str) -> float | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    if "landmarks" in results:
        return float(results["landmarks"].get("macro_f1", 0.0))
    if "cnn" in results:
        return float(results["cnn"].get("macro_f1", 0.0))
    return None


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    test_rows = _load_jsonl(dataset_dir / "features_package" / "test.jsonl")
    if not test_rows:
        raise SystemExit("No test rows found under features_package/test.jsonl")
    pipe = SltPipeline.load(Path(args.model))

    refs: list[str] = []
    preds: list[str] = []
    confs: list[float] = []
    per_sample: list[dict] = []
    for row in test_rows:
        data = np.load(Path(row["feature_path"]), allow_pickle=True)
        seq = np.asarray(data["features"], dtype=np.float32)
        pred, conf, emb = pipe.predict_from_sequence(seq)
        ref = str(row.get("target_text", ""))
        refs.append(ref)
        preds.append(pred)
        confs.append(float(conf))
        per_sample.append(
            {
                "sample_id": row.get("sample_id", ""),
                "reference": ref,
                "prediction": pred,
                "confidence": float(conf),
                "exact_match": normalize_slt_text(ref) == normalize_slt_text(pred),
                "token_overlap": token_overlap_score(ref, pred),
                "bleu_like": bleu_like(ref, pred),
                "embedding_dim": int(emb.shape[0]),
            }
        )

    exact = [1.0 if normalize_slt_text(r) == normalize_slt_text(p) else 0.0 for r, p in zip(refs, preds)]
    overlap = [token_overlap_score(r, p) for r, p in zip(refs, preds)]
    bleu = [bleu_like(r, p) for r, p in zip(refs, preds)]
    out = {
        "dataset_dir": str(dataset_dir),
        "model": str(args.model),
        "results": {
            "slt": {
                "n_test": len(refs),
                "exact_match_rate": _mean(exact),
                "token_overlap": _mean(overlap),
                "bleu_like": _mean(bleu),
                "avg_confidence": _mean(confs),
            },
            "baselines": {
                "landmarks_macro_f1": _load_baseline_macro(args.landmarks_eval_json),
                "cnn_macro_f1": _load_baseline_macro(args.cnn_eval_json),
                "multimodal_macro_f1": _load_baseline_macro(args.multimodal_eval_json),
            },
        },
        "samples": per_sample,
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md_out:
        md = Path(args.md_out)
        lines = [
            "# SLT Evaluation",
            "",
            f"- Test samples: `{len(refs)}`",
            f"- Exact match: `{out['results']['slt']['exact_match_rate']:.3f}`",
            f"- Token overlap: `{out['results']['slt']['token_overlap']:.3f}`",
            f"- BLEU-like: `{out['results']['slt']['bleu_like']:.3f}`",
            f"- Avg confidence: `{out['results']['slt']['avg_confidence']:.3f}`",
            "",
            "## Baseline comparison",
            "",
            f"- landmarks macro-F1: `{out['results']['baselines']['landmarks_macro_f1']}`",
            f"- cnn macro-F1: `{out['results']['baselines']['cnn_macro_f1']}`",
            f"- multimodal macro-F1: `{out['results']['baselines']['multimodal_macro_f1']}`",
            "",
        ]
        md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {json_out}")
    if args.md_out:
        print(f"Wrote: {args.md_out}")


if __name__ == "__main__":
    main()
