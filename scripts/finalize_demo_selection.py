from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run video comparison and generate the final demo recommendation.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases-json", default="")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--frame-work-dir", default="", help="Workspace with models/landmarks.joblib, cnn.pt, sequence.joblib")
    p.add_argument("--multimodal-work-dir", default="", help="Workspace with models/multimodal_sequence.joblib")
    p.add_argument("--landmarks-model", default="")
    p.add_argument("--cnn-model", default="")
    p.add_argument("--sequence-model", default="")
    p.add_argument("--multimodal-model", default="")
    p.add_argument("--pipelines", nargs="*", choices=["landmarks", "cnn", "sequence", "multimodal"], default=None)
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
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="8000")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _pick_model(work_dir: str, name: str) -> str:
    if not work_dir:
        return ""
    cand = Path(work_dir) / "models" / name
    return str(cand) if cand.exists() else ""


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarks_model = args.landmarks_model or _pick_model(args.frame_work_dir, "landmarks.joblib")
    cnn_model = args.cnn_model or _pick_model(args.frame_work_dir, "cnn.pt")
    sequence_model = args.sequence_model or _pick_model(args.frame_work_dir, "sequence.joblib")
    multimodal_model = args.multimodal_model or _pick_model(args.multimodal_work_dir, "multimodal_sequence.joblib")

    pipelines = list(args.pipelines or [])
    if not pipelines:
        if landmarks_model:
            pipelines.append("landmarks")
        if cnn_model:
            pipelines.append("cnn")
        if sequence_model:
            pipelines.append("sequence")
        if multimodal_model:
            pipelines.append("multimodal")
    if not pipelines:
        raise SystemExit("No pipelines available. Provide model paths or work dirs.")

    compare_dir = out_dir / "compare"
    compare_cmd = [
        py,
        str(repo / "scripts" / "compare_video_pipelines.py"),
        "--pipelines",
        *pipelines,
        "--out-dir",
        str(compare_dir),
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
    ]
    if landmarks_model:
        compare_cmd += ["--landmarks-model", landmarks_model]
    if cnn_model:
        compare_cmd += ["--cnn-model", cnn_model]
    if sequence_model:
        compare_cmd += ["--sequence-model", sequence_model]
    if multimodal_model:
        compare_cmd += ["--multimodal-model", multimodal_model]
    if args.cases_json:
        compare_cmd += ["--cases-json", args.cases_json]
    if args.videos:
        compare_cmd += ["--videos", *args.videos]
    if args.preprocess:
        compare_cmd.append("--preprocess")
    if args.skin_mask:
        compare_cmd.append("--skin-mask")
    if args.use_tracker:
        compare_cmd.append("--use-tracker")
    _run(compare_cmd)

    recommend_dir = out_dir / "recommendation"
    recommend_cmd = [
        py,
        str(repo / "scripts" / "recommend_demo_pipeline.py"),
        "--compare-json",
        str(compare_dir / "compare_video_pipelines.json"),
        "--out-dir",
        str(recommend_dir),
        "--host",
        args.host,
        "--port",
        args.port,
    ]
    if landmarks_model:
        recommend_cmd += ["--landmarks-model", landmarks_model]
    if cnn_model:
        recommend_cmd += ["--cnn-model", cnn_model]
    if sequence_model:
        recommend_cmd += ["--sequence-model", sequence_model]
    if multimodal_model:
        recommend_cmd += ["--multimodal-model", multimodal_model]
    if args.cases_json:
        recommend_cmd += ["--cases-json", args.cases_json]
    _run(recommend_cmd)
    print(f"Prepared final demo selection: {out_dir}")


if __name__ == "__main__":
    main()
