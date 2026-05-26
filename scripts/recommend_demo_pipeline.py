from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a recommended demo setup from compare_video_pipelines.json.")
    p.add_argument("--compare-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="8000")
    p.add_argument("--landmarks-model", default="")
    p.add_argument("--cnn-model", default="")
    p.add_argument("--sequence-model", default="")
    p.add_argument("--multimodal-model", default="")
    p.add_argument("--cases-json", default="")
    return p.parse_args()


def _model_arg(best: str, args: argparse.Namespace) -> list[str]:
    if best == "landmarks" and args.landmarks_model:
        return ["--landmarks-model", args.landmarks_model]
    if best == "cnn" and args.cnn_model:
        return ["--cnn-model", args.cnn_model]
    if best == "sequence" and args.sequence_model:
        return ["--sequence-model", args.sequence_model]
    if best == "multimodal" and args.multimodal_model:
        return ["--multimodal-model", args.multimodal_model]
    return []


def _shell_join(parts: list[str]) -> str:
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    compare = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    best = str(compare.get("best_pipeline") or "")
    summary = compare.get("summary") or []
    best_row = next((row for row in summary if row.get("pipeline") == best), {})

    web_cmd = [
        "python3",
        "vcpria.py",
        "web",
        *_model_arg(best, args),
        "--host",
        args.host,
        "--port",
        args.port,
        "--open-browser",
    ]
    validate_cmd = [
        "python3",
        "vcpria.py",
        "validate-videos",
        "--pipeline",
        best,
        *_model_arg(best, args),
    ]
    if args.cases_json:
        validate_cmd += ["--cases-json", args.cases_json]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_pipeline": best,
        "best_summary": best_row,
        "web_command": web_cmd,
        "validate_command": validate_cmd,
    }
    json_path = out_dir / "recommended_demo_pipeline.json"
    md_path = out_dir / "recommended_demo_pipeline.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# Recommended demo pipeline\n")
    md.append(f"- Best pipeline: `{best}`")
    if best_row:
        md.append(f"- Exact match rate: `{float(best_row.get('exact_match_rate', 0.0)):.3f}`")
        md.append(f"- Token match rate: `{float(best_row.get('token_match_rate', 0.0)):.3f}`")
        md.append(f"- Avg confidence: `{float(best_row.get('avg_confidence', 0.0)):.3f}`")
    md.append("")
    md.append("## Launch web demo")
    md.append("```bash")
    md.append(_shell_join(web_cmd))
    md.append("```")
    md.append("")
    md.append("## Re-run validation")
    md.append("```bash")
    md.append(_shell_join(validate_cmd))
    md.append("```")
    md.append("")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
