from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render report/pitch markdown sections from a WhisperX SLT pipeline summary.json."
    )
    p.add_argument("--summary-json", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _fmt(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _load_json(path: str | Path) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_definitions() -> list[str]:
    return [
        "**Definición de métricas (texto):**",
        "- *Exact match*: predicción coincide exactamente con referencia (normalizando minúsculas/espacios/acentos).",
        "- *Token overlap*: solapamiento de tokens (0–1). Captura aciertos parciales aunque no haya match exacto.",
        "- *BLEU-like*: señal tipo BLEU sobre n‑gramas. Útil como indicativo general (no es el BLEU estándar del paper).",
        "",
    ]


def _top_errors(eval_json: dict, k: int = 6) -> list[str]:
    samples = eval_json.get("samples") if isinstance(eval_json, dict) else None
    if not isinstance(samples, list):
        return []
    rows = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        if bool(s.get("exact_match", False)):
            continue
        ref = str(s.get("reference", "")).strip()
        pred = str(s.get("prediction", "")).strip()
        if not ref or not pred:
            continue
        rows.append(
            {
                "sample_id": str(s.get("sample_id", "")),
                "ref": ref,
                "pred": pred,
                "overlap": float(s.get("token_overlap", 0.0) or 0.0),
                "bleu": float(s.get("bleu_like", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda r: (r["overlap"], r["bleu"]))
    if not rows:
        return []
    out = ["**Ejemplos de errores (proxy, test)** *(inspección cualitativa)*:"]
    for r in rows[: max(1, int(k))]:
        ref = r["ref"][:90] + ("…" if len(r["ref"]) > 90 else "")
        pred = r["pred"][:90] + ("…" if len(r["pred"]) > 90 else "")
        out.append(f"- `{r['sample_id']}` overlap `{r['overlap']:.2f}` | ref: “{ref}” | pred: “{pred}”")
    out.append("")
    return out


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = summary.get("sources", [])
    subset = summary.get("subset", {}) or {}
    subset_rows = int(subset.get("rows", 0) or 0)
    subset_splits = subset.get("splits", {}) or subset.get("split_counts", {}) or {}
    subset_episodes = subset.get("episodes", None)
    subset_cfg = (subset.get("config", {}) or {})

    validation = summary.get("dataset_validation", {}) or {}
    valid = bool(validation.get("valid", False))
    validation_rows = int(validation.get("rows", 0) or 0)
    validation_splits = validation.get("split_counts", {}) or {}

    train_eval = summary.get("train_eval", {}) or {}
    backend_report = (train_eval.get("backend_report", {}) or {})
    backend_status = backend_report.get("status", "unknown")

    eval_proxy = summary.get("eval_proxy", {}) or summary.get("eval_slt", {}) or {}

    exact = _fmt(eval_proxy.get("exact_match_rate", 0.0))
    overlap = _fmt(eval_proxy.get("token_overlap", 0.0))
    bleu = _fmt(eval_proxy.get("bleu_like", 0.0))
    avg_conf = _fmt(eval_proxy.get("avg_confidence", 0.0))

    paths = summary.get("paths", {}) or {}
    export_dir = Path(paths.get("export_dir", "")) if paths.get("export_dir") else None
    export_info = _load_json(export_dir / "dataset_export_info.json") if export_dir else {}
    eval_json = _load_json(paths.get("eval_json", "")) if paths.get("eval_json") else {}

    sample_fps = export_info.get("sample_fps", subset_cfg.get("sample_fps", "N/A"))
    max_frames = export_info.get("max_frames", subset_cfg.get("max_frames", "N/A"))
    preprocess = export_info.get("preprocess", subset_cfg.get("preprocess", "N/A"))

    backend_ckpt = str(backend_report.get("ckpt_latest") or "")
    backend_use_cuda = backend_report.get("use_cuda", None)

    report_lines = [
        "### Resultados y evaluación (SLT con iLSU‑T + WhisperX)",
        "",
        "Este experimento trata iLSU‑T como un problema de **traducción** (señas → texto).",
        "Se usa WhisperX para construir pares supervisados por segmento:",
        "- input: clip de video (ventana temporal del intérprete)",
        "- target: texto WhisperX del segmento",
        "",
        f"Fuentes utilizadas: `{', '.join(sources) if sources else 'N/A'}`.",
        f"Muestras exportadas: `{subset_rows}` (validación dataset rows: `{validation_rows}`) con splits `{subset_splits or validation_splits}` y `group_id` por episodio para evitar leakage (`dataset_validation.valid = {valid}`).",
    ]
    if subset_episodes is not None:
        report_lines.append(f"Episodios/`group_id` únicos: `{subset_episodes}`.")
    report_lines += [
        "",
        "#### Configuración del dataset (segmentos)",
        f"- duración: `{subset_cfg.get('min_duration_ms', 'N/A')}`–`{subset_cfg.get('max_duration_ms', 'N/A')}` ms",
        f"- longitud: `{subset_cfg.get('min_words', 'N/A')}`–`{subset_cfg.get('max_words', 'N/A')}` palabras, máx `{subset_cfg.get('max_chars', 'N/A')}` chars",
        f"- tope por episodio: `{subset_cfg.get('max_segments_per_episode', 0)}` (0 = sin tope)",
        "",
        "#### Features (clip → secuencia)",
        f"- sample_fps: `{sample_fps}` | max_frames: `{max_frames}` | preprocess: `{preprocess}`",
        "",
        "#### Métricas (test, baseline proxy)",
        f"- exact match rate: `{exact:.3f}`",
        f"- token overlap: `{overlap:.3f}`",
        f"- BLEU-like: `{bleu:.3f}`",
        f"- confianza promedio: `{avg_conf:.3f}`",
        "",
        "#### Entrenamiento generativo (opcional)",
        f"- backend (neccam/slt SignJoey): `{backend_status}`",
        (f"- ckpt (latest): `{backend_ckpt}`" if backend_ckpt else "- ckpt: `N/A`"),
        (f"- use_cuda: `{backend_use_cuda}`" if backend_use_cuda is not None else "- use_cuda: `N/A`"),
        "",
        "#### Justificación metodológica",
        "- iLSU‑T es un dataset de traducción; un pipeline *clip → texto* es coherente con el objetivo.",
        "- WhisperX habilita construir un dataset reproducible sin anotación manual frame‑a‑frame (pero introduce ruido de alineamiento).",
        "- Por eso se reporta una métrica cuantitativa + inspección cualitativa de errores.",
        "",
        "#### Qué modelo se usa en la demo",
        "- Para asegurar reproducibilidad y latencia, la demo usa por defecto un **baseline proxy** (KNN sobre embeddings agregados).",
        "- Si el modelo generativo está entrenado (Colab/GPU), el backend puede exponer inferencia generativa para análisis offline (video → texto).",
        "",
    ] + _metric_definitions()

    if eval_json:
        report_lines += _top_errors(eval_json, k=6)

    pitch_lines = [
        "## Slide extra — Resultados SLT (iLSU‑T + WhisperX)",
        "",
        f"- Fuentes: `{', '.join(sources) if sources else 'N/A'}`",
        f"- Dataset: `{subset_rows}` muestras | splits: `{subset_splits or validation_splits}` | válido: `{valid}`",
        f"- Segmentos: dur `{subset_cfg.get('min_duration_ms', 'N/A')}`–`{subset_cfg.get('max_duration_ms', 'N/A')}`ms | words `{subset_cfg.get('min_words', 'N/A')}`–`{subset_cfg.get('max_words', 'N/A')}`",
        f"- Features: fps `{sample_fps}` | frames `{max_frames}` | preprocess `{preprocess}`",
        f"- Métricas test (proxy): EM `{exact:.3f}` | overlap `{overlap:.3f}` | BLEU-like `{bleu:.3f}` | conf `{avg_conf:.3f}`",
        f"- Backend generativo (SignJoey): `{backend_status}`",
        "",
    ]

    (out_dir / "report_results_section.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / "pitch_results_section.md").write_text("\n".join(pitch_lines), encoding="utf-8")
    print(f"Wrote: {out_dir / 'report_results_section.md'}")
    print(f"Wrote: {out_dir / 'pitch_results_section.md'}")


if __name__ == "__main__":
    main()

