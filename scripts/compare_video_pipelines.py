from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple pipelines over the same validation videos.")
    p.add_argument("--pipelines", nargs="+", choices=["landmarks", "cnn", "sequence", "multimodal", "slt"], required=True)
    p.add_argument("--landmarks-model", default="")
    p.add_argument("--cnn-model", default="")
    p.add_argument("--sequence-model", default="")
    p.add_argument("--multimodal-model", default="")
    p.add_argument("--slt-model", default="")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--cases-json", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["both", "words", "spelling"], default="both")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--skin-mask", action="store_true")
    p.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    p.add_argument("--use-tracker", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.75)
    p.add_argument("--stable-frames-min", type=int, default=6)
    p.add_argument("--pause-ms-min", type=int, default=350)
    p.add_argument("--cooldown-ms", type=int, default=800)
    p.add_argument("--sample-fps", type=float, default=0.0)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _required_model_arg(pipeline: str, args: argparse.Namespace) -> list[str]:
    if pipeline == "landmarks":
        if not args.landmarks_model:
            raise SystemExit("Missing --landmarks-model for landmarks pipeline")
        return ["--landmarks-model", args.landmarks_model]
    if pipeline == "cnn":
        if not args.cnn_model:
            raise SystemExit("Missing --cnn-model for cnn pipeline")
        return ["--cnn-model", args.cnn_model]
    if pipeline == "sequence":
        if not args.sequence_model:
            raise SystemExit("Missing --sequence-model for sequence pipeline")
        return ["--sequence-model", args.sequence_model]
    if pipeline == "slt":
        if not args.slt_model:
            raise SystemExit("Missing --slt-model for slt pipeline")
        return ["--slt-model", args.slt_model]
    if not args.multimodal_model:
        raise SystemExit("Missing --multimodal-model for multimodal pipeline")
    return ["--multimodal-model", args.multimodal_model]


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "cases": 0,
            "exact_match_rate": 0.0,
            "token_match_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_predictions": 0.0,
            "avg_frames_used": 0.0,
        }
    exact = sum(1 for r in rows if r.get("exact_match"))
    token = sum(1 for r in rows if r.get("token_match"))
    confs = [float(r.get("avg_confidence", 0.0) or 0.0) for r in rows]
    preds = [float(r.get("predictions_count", 0.0) or 0.0) for r in rows]
    frames = [float(r.get("frames_used", 0.0) or 0.0) for r in rows]
    return {
        "cases": n,
        "exact_match_rate": exact / n,
        "token_match_rate": token / n,
        "avg_confidence": sum(confs) / n,
        "avg_predictions": sum(preds) / n,
        "avg_frames_used": sum(frames) / n,
    }


def _to_markdown(summary_rows: list[dict]) -> str:
    headers = ["pipeline", "cases", "exact_match_rate", "token_match_rate", "avg_confidence", "avg_predictions", "avg_frames_used"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pipeline"]),
                    str(row["cases"]),
                    f'{row["exact_match_rate"]:.3f}',
                    f'{row["token_match_rate"]:.3f}',
                    f'{row["avg_confidence"]:.3f}',
                    f'{row["avg_predictions"]:.1f}',
                    f'{row["avg_frames_used"]:.1f}',
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    detailed: dict[str, list[dict]] = {}
    for pipeline in args.pipelines:
        pipeline_out = out_dir / pipeline
        cmd = [
            py,
            str(repo / "scripts" / "validate_videos.py"),
            "--pipeline",
            pipeline,
            "--out-dir",
            str(pipeline_out),
            "--mode",
            args.mode,
            "--mask-space",
            args.mask_space,
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--stable-frames-min",
            str(args.stable_frames_min),
            "--pause-ms-min",
            str(args.pause_ms_min),
            "--cooldown-ms",
            str(args.cooldown_ms),
            "--sample-fps",
            str(args.sample_fps),
            *_required_model_arg(pipeline, args),
        ]
        if args.videos:
            cmd += ["--videos", *args.videos]
        if args.cases_json:
            cmd += ["--cases-json", args.cases_json]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.skin_mask:
            cmd.append("--skin-mask")
        if args.use_tracker:
            cmd.append("--use-tracker")
        _run(cmd)

        rows = json.loads((pipeline_out / "video_validation.json").read_text(encoding="utf-8"))
        detailed[pipeline] = rows
        summary = _summarize(rows)
        summary["pipeline"] = pipeline
        summaries.append(summary)

    summaries.sort(key=lambda r: (r["exact_match_rate"], r["token_match_rate"], r["avg_confidence"]), reverse=True)
    best = summaries[0]["pipeline"] if summaries else ""

    summary_json = out_dir / "compare_video_pipelines.json"
    summary_md = out_dir / "compare_video_pipelines.md"
    summary_json.write_text(
        json.dumps({"best_pipeline": best, "summary": summaries, "details": detailed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_md.write_text(_to_markdown(summaries), encoding="utf-8")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    if best:
        print(f"Recommended pipeline: {best}")


if __name__ == "__main__":
    main()
