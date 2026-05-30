from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render report/pitch markdown sections from an iLSU-T SLT summary.json.")
    p.add_argument("--summary-json", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(summary.get("support", {}).get("recommended_labels", []))
    split_counts = summary.get("subset", {}).get("split_counts", {})
    split_episode_counts = summary.get("subset", {}).get("split_episode_counts", {})
    eval_slt = summary.get("eval_slt", {})
    valid = summary.get("dataset_validation", {}).get("valid", False)
    backend_status = summary.get("train_eval", {}).get("backend_report", {}).get("status", "unknown")

    report_lines = [
        "### Resultados SLT con iLSU-T",
        "",
        f"Como experimento complementario, se preparó un subset de iLSU-T usando las fuentes `{', '.join(summary.get('sources', []))}` y un vocabulario acotado de `{', '.join(labels) if labels else 'N/A'}`.",
        "",
        f"El subset resultante quedó compuesto por `{summary.get('subset', {}).get('rows', 0)}` muestras distribuidas en train/val/test como `{split_counts}` y sin mezcla de episodios entre splits (`dataset_validation.valid = {valid}`).",
        "",
        "Las métricas del baseline SLT sobre el split de test fueron:",
        f"- exact match rate: `{float(eval_slt.get('exact_match_rate', 0.0)):.3f}`",
        f"- token overlap: `{float(eval_slt.get('token_overlap', 0.0)):.3f}`",
        f"- BLEU-like: `{float(eval_slt.get('bleu_like', 0.0)):.3f}`",
        f"- confianza promedio: `{float(eval_slt.get('avg_confidence', 0.0)):.3f}`",
        "",
        f"En esta fase, el backend externo quedó en estado `{backend_status}`, mientras que el repositorio mantuvo un baseline local `slt_proxy.joblib` para asegurar reproducibilidad e integración con la app.",
        "",
        "Estos resultados deben leerse como una primera aproximación offline sobre clips completos, no como traducción abierta en tiempo real desde webcam.",
        "",
    ]

    pitch_lines = [
        "## Slide extra — SLT con iLSU-T",
        "",
        f"- Subset iLSU-T preparado con `{', '.join(summary.get('sources', []))}`",
        f"- Labels con mejor soporte: `{', '.join(labels) if labels else 'N/A'}`",
        f"- Split: train `{split_counts.get('train', 0)}`, val `{split_counts.get('val', 0)}`, test `{split_counts.get('test', 0)}`",
        f"- Episodios por split: `{split_episode_counts}`",
        f"- Exact match / token overlap / BLEU-like: `{float(eval_slt.get('exact_match_rate', 0.0)):.3f}` / `{float(eval_slt.get('token_overlap', 0.0)):.3f}` / `{float(eval_slt.get('bleu_like', 0.0)):.3f}`",
        f"- Dataset válido: `{valid}`",
        f"- Estado backend SLT: `{backend_status}`",
        "",
    ]

    (out_dir / "report_results_section.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / "pitch_results_section.md").write_text("\n".join(pitch_lines), encoding="utf-8")
    print(f"Wrote: {out_dir / 'report_results_section.md'}")
    print(f"Wrote: {out_dir / 'pitch_results_section.md'}")


if __name__ == "__main__":
    main()
