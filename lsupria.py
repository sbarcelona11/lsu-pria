from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _python() -> str:
    return sys.executable or "python"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run(args: list[str]) -> int:
    return subprocess.call(args)


def main() -> None:
    p = argparse.ArgumentParser(prog="lsupria")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("web", help="Run FastAPI web demo")
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--slt-gen-backend-repo", default="")
    sp.add_argument("--slt-gen-config", default="")
    sp.add_argument("--slt-gen-ckpt", default="")
    sp.add_argument("--slt-gen-model-dir", default="")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--open-browser", action="store_true")

    sp = sub.add_parser("app-dev", help="Run backend and frontend dev servers together")
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--stack-work-dir", default="")
    sp.add_argument("--api-host", default="127.0.0.1")
    sp.add_argument("--api-port", type=int, default=8000)
    sp.add_argument("--ui-port", type=int, default=5173)
    sp.add_argument("--install-ui", action="store_true")

    sp = sub.add_parser("demo", help="Run OpenCV desktop demo")
    sp.add_argument("--pipeline", choices=["landmarks", "cnn", "sequence", "multimodal"], default="landmarks")
    sp.add_argument("--model", required=True)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--width", type=int, default=1280)
    sp.add_argument("--height", type=int, default=720)
    sp.add_argument("--max-fps", type=float, default=0.0)

    sp = sub.add_parser("collect", help="Collect labeled webcam samples")
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels", nargs="+", required=True)
    sp.add_argument("--subject-id", default="S1")
    sp.add_argument("--camera", type=int, default=0)

    sp = sub.add_parser("collect-seq", help="Collect short sequences for dynamic gestures")
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels", nargs="+", required=True)
    sp.add_argument("--subject-id", default="S1")
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--duration", type=float, default=2.0)

    sp = sub.add_parser("collect-mm-seq", help="Collect multimodal sequences (hands + pose + face)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels", nargs="+", required=True)
    sp.add_argument("--subject-id", default="S1")
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--duration", type=float, default=2.5)

    sp = sub.add_parser("train-landmarks", help="Train landmarks baseline model")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("train-cnn", help="Train CNN transfer-learning model")
    sp.add_argument("--img-dir", default="")
    sp.add_argument("--csv", default="")
    sp.add_argument("--image-col", default="auto")
    sp.add_argument("--out", required=True)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--unfreeze-backbone", action="store_true")
    sp.add_argument("--group-col", default="")

    sp = sub.add_parser("train-seq", help="Train temporal baseline from landmark sequences")
    sp.add_argument("--seq-dir", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("train-mm-seq", help="Train multimodal temporal baseline")
    sp.add_argument("--seq-dir", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("train-mm-stack", help="Run the recommended local multimodal training stack")
    sp.add_argument("--seq-dir", default="")
    sp.add_argument("--seq-dirs", nargs="*", default=None)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--window", type=int, default=24)
    sp.add_argument("--min-frames", type=int, default=8)
    sp.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="subject_id")

    sp = sub.add_parser("train-videos-stack", help="Prepare labeled videos and run the frame-based training stack")
    sp.add_argument("--videos-root", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    sp.add_argument("--subject-default", default="video")
    sp.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    sp.add_argument("--fps", type=float, default=4.0)
    sp.add_argument("--max-per-video", type=int, default=120)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--camera-like", action="store_true")
    sp.add_argument("--cnn-image-col", default="img_raw_path")
    sp.add_argument("--group-col", default="group_id")
    sp.add_argument("--cnn-epochs", type=int, default=10)
    sp.add_argument("--cnn-device", default="cpu")
    sp.add_argument("--skip-ablation", action="store_true")
    sp.add_argument("--ablation-seconds", type=float, default=10.0)
    sp.add_argument("--camera", type=int, default=0)

    sp = sub.add_parser("train-videos-mm-stack", help="Prepare labeled videos and run the multimodal temporal training stack")
    sp.add_argument("--videos-root", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    sp.add_argument("--subject-default", default="video")
    sp.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    sp.add_argument("--fps", type=float, default=8.0)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--window", type=int, default=24)
    sp.add_argument("--min-frames", type=int, default=8)
    sp.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="group_id")

    sp = sub.add_parser("train-stack", help="Run the recommended local end-to-end training stack")
    sp.add_argument("--csv", default="")
    sp.add_argument("--csvs", nargs="*", default=None)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--cnn-image-col", default="img_raw_path")
    sp.add_argument("--group-col", default="subject_id")
    sp.add_argument("--cnn-epochs", type=int, default=10)
    sp.add_argument("--cnn-device", default="cpu")
    sp.add_argument("--skip-ablation", action="store_true")
    sp.add_argument("--ablation-seconds", type=float, default=10.0)
    sp.add_argument("--camera", type=int, default=0)

    sp = sub.add_parser("demo-stack-web", help="Launch the web demo from a train-stack workspace")
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--open-browser", action="store_true")

    sp = sub.add_parser("ilsut-train", help="Prepare iLSU-T weak labels and train project models")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--pipelines", nargs="+", choices=["landmarks", "cnn"], default=["cnn"])
    sp.add_argument("--fps", type=float, default=5.0)
    sp.add_argument("--max-per-seg", type=int, default=40)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--camera-like", action="store_true")
    sp.add_argument("--group-col", default="group_id")
    sp.add_argument("--cnn-epochs", type=int, default=10)
    sp.add_argument("--cnn-device", default="cpu")
    sp.add_argument("--cnn-use-masked", action="store_true")
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--min-label-count", type=int, default=0)
    sp.add_argument("--skip-ablation", action="store_true")
    sp.add_argument("--ablation-seconds", type=float, default=10.0)
    sp.add_argument("--camera", type=int, default=0)

    sp = sub.add_parser("ilsut-train-mm", help="Prepare iLSU-T weak labels and train multimodal temporal baseline")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--fps", type=float, default=6.0)
    sp.add_argument("--max-per-seg", type=int, default=40)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--min-label-count", type=int, default=0)
    sp.add_argument("--window", type=int, default=24)
    sp.add_argument("--min-frames", type=int, default=8)
    sp.add_argument("--group-col", choices=["group_id", "subject_id", "none"], default="group_id")

    sp = sub.add_parser("ilsut-preset", help="Run a practical preset for iLSU-T training on extracted data")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", default="deliverables/ilsut_keywords.example.json")
    sp.add_argument("--source", default="source2")
    sp.add_argument("--mode", choices=["frame", "multimodal", "both"], default="both")
    sp.add_argument("--preset", choices=["quick", "standard"], default="quick")
    sp.add_argument("--work-root", default="runs/ilsut_presets")
    sp.add_argument("--cnn-device", default="cpu")

    sp = sub.add_parser("ilsut-download", help="Download iLSU-T archives from the local manifest")
    sp.add_argument("--manifest", default="deliverables/ilsut_downloads.json")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--skip-existing", action="store_true")

    sp = sub.add_parser("ilsut-extract", help="Extract downloaded iLSU-T .rar archives")
    sp.add_argument("--archives-dir", required=True)
    sp.add_argument("--out-root", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--skip-existing", action="store_true")

    sp = sub.add_parser("ilsut-convert-videos", help="Convert extracted iLSU-T .avi videos to mp4 or mkv")
    sp.add_argument("--root", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--input-ext", default=".avi")
    sp.add_argument("--output-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--skip-existing", action="store_true")
    sp.add_argument("--tool", choices=["auto", "ffmpeg", "opencv"], default="auto")
    sp.add_argument("--quality", choices=["copy", "fast", "balanced"], default="balanced")
    sp.add_argument("--delete-source", action="store_true", help="Delete source video after successful conversion")

    sp = sub.add_parser("ilsut-build-episodes-csv", help="Generate an episodes.csv-like file from extracted iLSU-T files")
    sp.add_argument("--root", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--video-dir-name", default="episodes")
    sp.add_argument("--whisperx-dir-name", default="whisperx")
    sp.add_argument("--include-unmatched", action="store_true")
    sp.add_argument("--strict", action="store_true")

    sp = sub.add_parser("ilsut-analyze-support", help="Summarize weak-label support by class for extracted iLSU-T sources")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--min-label-count", type=int, default=20)

    sp = sub.add_parser("ilsut-prepare-slt-subset", help="Prepare an iLSU-T clip-level subset with train/val/test splits")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--min-label-count", type=int, default=20)
    sp.add_argument("--labels-json", default="")
    sp.add_argument("--test-size", type=float, default=0.2)
    sp.add_argument("--val-size", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--export-clips", action="store_true")
    sp.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--max-clips", type=int, default=0)
    sp.add_argument(
        "--dedup-eval-text",
        choices=["off", "train_exact", "train_val_exact"],
        default="off",
        help="Drop duplicated target_text from val/test to reduce evaluation leakage (default: off)",
    )

    sp = sub.add_parser(
        "ilsut-prepare-slt-subset-whisperx",
        help="Prepare an SLT subset directly from WhisperX segments (clip -> target_text), for generative SLT",
    )
    sp.add_argument("--root", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--video-dir-name", default="episodes")
    sp.add_argument("--whisperx-dir-name", default="whisperx")
    sp.add_argument("--test-size", type=float, default=0.2)
    sp.add_argument("--val-size", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--min-words", type=int, default=1)
    sp.add_argument("--max-words", type=int, default=60)
    sp.add_argument("--max-chars", type=int, default=240)
    sp.add_argument("--min-duration-ms", type=int, default=700)
    sp.add_argument("--max-duration-ms", type=int, default=25000)
    sp.add_argument("--keep-punctuation", action="store_true")
    sp.add_argument("--max-segments-per-episode", type=int, default=0)
    sp.add_argument("--export-clips", action="store_true")
    sp.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--max-clips", type=int, default=0)
    sp.add_argument(
        "--dedup-eval-text",
        choices=["off", "train_exact", "train_val_exact"],
        default="off",
        help="Drop duplicated target_text from val/test to reduce evaluation leakage (default: off)",
    )

    sp = sub.add_parser("ilsut-audit-keywords", help="Audit which transcript variants are matched by the current iLSU-T keywords rules")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--sources", nargs="*", default=None)
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--top-k", type=int, default=10)

    sp = sub.add_parser("ilsut-extract-slt-features", help="Extract clip-level multimodal SLT features from a prepared iLSU-T subset")
    sp.add_argument("--subset-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--sample-fps", type=float, default=6.0)
    sp.add_argument("--max-frames", type=int, default=0)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    sp.add_argument("--limit", type=int, default=0)

    sp = sub.add_parser("ilsut-export-slt-dataset", help="Export a prepared iLSU-T subset as an SLT dataset package")
    sp.add_argument("--subset-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--mode", choices=["features", "raw"], default="features")
    sp.add_argument("--sample-fps", type=float, default=6.0)
    sp.add_argument("--max-frames", type=int, default=0)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--max-clips", type=int, default=0)
    sp.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    sp.add_argument("--limit", type=int, default=0)

    sp = sub.add_parser("train-ilsut-slt", help="Train the SLT wrapper package and a local proxy baseline")
    sp.add_argument("--subset-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--dataset-dir", default="")
    sp.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    sp.add_argument("--backend-repo", default="")
    sp.add_argument("--config-base", default="")
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--sample-fps", type=float, default=6.0)
    sp.add_argument("--max-frames", type=int, default=0)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--run-backend", action="store_true")

    sp = sub.add_parser("eval-ilsut-slt", help="Evaluate an SLT model bundle on an exported iLSU-T test split")
    sp.add_argument("--dataset-dir", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--json-out", required=True)
    sp.add_argument("--md-out", default="")
    sp.add_argument("--landmarks-eval-json", default="")
    sp.add_argument("--cnn-eval-json", default="")
    sp.add_argument("--multimodal-eval-json", default="")

    sp = sub.add_parser("validate-ilsut-slt-dataset", help="Validate an exported iLSU-T SLT dataset package")
    sp.add_argument("--dataset-dir", required=True)
    sp.add_argument("--json-out", default="")
    sp.add_argument("--md-out", default="")
    sp.add_argument("--require-features", action="store_true")

    sp = sub.add_parser("run-ilsut-slt-pipeline", help="Run support -> subset -> export -> validate -> train -> eval for iLSU-T SLT")
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", default="deliverables/ilsut_keywords.focused.json")
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--work-root", default="runs/ilsut_slt_pipeline")
    sp.add_argument("--preset", choices=["custom", "quick", "standard"], default="custom")
    sp.add_argument("--episodes-csv", default="auto")
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")
    sp.add_argument("--manifest-limit", type=int, default=0)
    sp.add_argument("--min-label-count", type=int, default=20)
    sp.add_argument("--labels-json", default="")
    sp.add_argument("--sample-fps", type=float, default=6.0)
    sp.add_argument("--max-frames", type=int, default=48)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--max-clips", type=int, default=0)
    sp.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    sp.add_argument("--backend-repo", default="")
    sp.add_argument("--config-base", default="")
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--run-backend", action="store_true")

    sp = sub.add_parser(
        "run-whisperx-slt-pipeline",
        help="Run SLT pipeline directly from WhisperX segments (clip -> target_text), for generative SLT training",
    )
    sp.add_argument("--root", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--work-root", default="runs/whisperx_slt_pipeline")
    sp.add_argument("--min-words", type=int, default=1)
    sp.add_argument("--max-words", type=int, default=60)
    sp.add_argument("--max-chars", type=int, default=240)
    sp.add_argument("--min-duration-ms", type=int, default=700)
    sp.add_argument("--max-duration-ms", type=int, default=25000)
    sp.add_argument("--keep-punctuation", action="store_true")
    sp.add_argument("--max-segments-per-episode", type=int, default=0)
    sp.add_argument("--sample-fps", type=float, default=6.0)
    sp.add_argument("--max-frames", type=int, default=48)
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    sp.add_argument("--max-clips", type=int, default=0)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument(
        "--dedup-eval-text",
        choices=["off", "train_exact", "train_val_exact"],
        default="train_exact",
        help="Drop duplicated target_text from val/test to reduce evaluation leakage (default: train_exact)",
    )
    sp.add_argument("--backend", choices=["neccam_slt"], default="neccam_slt")
    sp.add_argument("--backend-repo", default="")
    sp.add_argument("--config-base", default="")
    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--run-backend", action="store_true")

    sp = sub.add_parser("render-ilsut-slt-summary", help="Render report-friendly artifacts from an iLSU-T SLT summary.json")
    sp.add_argument("--summary-json", required=True)
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("render-ilsut-slt-sections", help="Render markdown sections for report/pitch from an iLSU-T SLT summary.json")
    sp.add_argument("--summary-json", required=True)
    sp.add_argument("--out-dir", required=True)

    sp = sub.add_parser("validate-videos", help="Run the recognition pipeline over validation videos")
    sp.add_argument("--pipeline", choices=["landmarks", "cnn", "sequence", "multimodal", "slt"], default="landmarks")
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--videos", nargs="*", default=None)
    sp.add_argument("--cases-json", default="")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--mode", choices=["both", "words", "spelling"], default="both")
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    sp.add_argument("--use-tracker", action="store_true")
    sp.add_argument("--confidence-threshold", type=float, default=0.75)
    sp.add_argument("--stable-frames-min", type=int, default=6)
    sp.add_argument("--pause-ms-min", type=int, default=350)
    sp.add_argument("--cooldown-ms", type=int, default=800)
    sp.add_argument("--sample-fps", type=float, default=0.0)

    sp = sub.add_parser("compare-video-pipelines", help="Compare multiple pipelines over the same validation videos")
    sp.add_argument("--pipelines", nargs="+", choices=["landmarks", "cnn", "sequence", "multimodal", "slt"], required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--videos", nargs="*", default=None)
    sp.add_argument("--cases-json", default="")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--mode", choices=["both", "words", "spelling"], default="both")
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    sp.add_argument("--use-tracker", action="store_true")
    sp.add_argument("--confidence-threshold", type=float, default=0.75)
    sp.add_argument("--stable-frames-min", type=int, default=6)
    sp.add_argument("--pause-ms-min", type=int, default=350)
    sp.add_argument("--cooldown-ms", type=int, default=800)
    sp.add_argument("--sample-fps", type=float, default=0.0)

    sp = sub.add_parser("recommend-demo-pipeline", help="Generate a recommended demo setup from video comparison results")
    sp.add_argument("--compare-json", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--cases-json", default="")

    sp = sub.add_parser("finalize-demo-selection", help="Run comparison + recommendation for the final demo in one shot")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--cases-json", default="")
    sp.add_argument("--videos", nargs="*", default=None)
    sp.add_argument("--frame-work-dir", default="")
    sp.add_argument("--multimodal-work-dir", default="")
    sp.add_argument("--slt-work-dir", default="")
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--multimodal-model", default="")
    sp.add_argument("--slt-model", default="")
    sp.add_argument("--pipelines", nargs="*", choices=["landmarks", "cnn", "sequence", "multimodal", "slt"], default=None)
    sp.add_argument("--mode", choices=["both", "words", "spelling"], default="both")
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    sp.add_argument("--use-tracker", action="store_true")
    sp.add_argument("--confidence-threshold", type=float, default=0.75)
    sp.add_argument("--stable-frames-min", type=int, default=6)
    sp.add_argument("--pause-ms-min", type=int, default=350)
    sp.add_argument("--cooldown-ms", type=int, default=800)
    sp.add_argument("--sample-fps", type=float, default=0.0)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    sp = sub.add_parser("eval-split", help="Evaluate with random/group split")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--cnn-image-col", default="auto")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--json-out", default="")

    sp = sub.add_parser("ablation", help="Run ablation (split macro-F1 + FPS profiling)")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--seconds", type=float, default=10.0)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--fps-preprocess", choices=["on", "off"], default="on")
    sp.add_argument("--fps-tracker", choices=["on", "off"], default="on")
    sp.add_argument("--fps-skin-mask", choices=["on", "off"], default="off")
    sp.add_argument("--fps-mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    sp.add_argument("--out-md", default="results/ablation_table.md")
    sp.add_argument("--out-json", default="results/ablation_results.json")

    sp = sub.add_parser("ablation-grid", help="Sweep OpenCV toggles (Macro-F1 + FPS)")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--seconds", type=float, default=10.0)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--out-md", default="results/ablation_grid.md")
    sp.add_argument("--out-json", default="results/ablation_grid.json")

    sp = sub.add_parser("dataset-stats", help="Print dataset stats and optional markdown")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--by", choices=["label", "label_subject"], default="label_subject")
    sp.add_argument("--out-md", default="")

    sp = sub.add_parser("validate-multisubject", help="Validate min samples per label per subject")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--min-per-label-per-subject", type=int, default=30)
    sp.add_argument("--out-md", default="")

    sp = sub.add_parser("merge-csvs", help="Merge multiple landmarks.csv into one")
    sp.add_argument("--out", required=True)
    sp.add_argument("--inputs", nargs="+", required=True)

    sp = sub.add_parser("report", help="Generate quick report (md + plot)")
    sp.add_argument("--ablation-json", default="results/ablation_results.json")
    sp.add_argument("--ablation-table", default="results/ablation_table.md")
    sp.add_argument("--dataset-stats-md", default="results/dataset_stats.md")
    sp.add_argument("--out-md", default="results/report.md")
    sp.add_argument("--out-fig", default="results/precision_vs_fps.png")

    sp = sub.add_parser("deliverables", help="Build PDFs + zip and run checks")
    sp.add_argument("--config", default="deliverables/config.json")
    sp.add_argument("--include-data", action="store_true")
    sp.add_argument("--include-models", action="store_true")
    sp.add_argument("--set-title", default="")
    sp.add_argument("--set-course", default="")
    sp.add_argument("--set-group", default="")
    sp.add_argument("--set-date", default="")
    sp.add_argument("--set-pitch-minutes", type=int, default=0)
    sp.add_argument("--set-members", nargs="*", default=None)
    sp.add_argument("--slt-summary-json", default="")
    sp.add_argument("--slt-report-section", default="")
    sp.add_argument("--slt-pitch-section", default="")

    args = p.parse_args()
    repo = _repo_root()
    py = _python()

    if args.cmd == "web":
        cmd = [
            py,
            str(repo / "scripts" / "run_webapp.py"),
            "--host",
            str(args.host),
            "--port",
            str(args.port),
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
        if args.slt_gen_backend_repo and args.slt_gen_config and args.slt_gen_ckpt:
            cmd += ["--slt-gen-backend-repo", args.slt_gen_backend_repo]
            cmd += ["--slt-gen-config", args.slt_gen_config]
            cmd += ["--slt-gen-ckpt", args.slt_gen_ckpt]
            if args.slt_gen_model_dir:
                cmd += ["--slt-gen-model-dir", args.slt_gen_model_dir]
        if args.open_browser:
            cmd += ["--open-browser"]
        raise SystemExit(_run(cmd))

    if args.cmd == "app-dev":
        cmd = [
            py,
            str(repo / "scripts" / "app_dev.py"),
            "--api-host",
            str(args.api_host),
            "--api-port",
            str(args.api_port),
            "--ui-port",
            str(args.ui_port),
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
        if args.stack_work_dir:
            cmd += ["--stack-work-dir", args.stack_work_dir]
        if args.install_ui:
            cmd += ["--install-ui"]
        raise SystemExit(_run(cmd))

    if args.cmd == "demo":
        cmd = [
            py,
            str(repo / "main.py"),
            "--pipeline",
            args.pipeline,
            "--model",
            args.model,
            "--camera",
            int(args.camera),
            "--width",
            int(args.width),
            "--height",
            int(args.height),
            "--max-fps",
            float(args.max_fps),
        ]
        cmd = [str(x) for x in cmd]
        raise SystemExit(_run(cmd))

    if args.cmd == "collect":
        cmd = [
            py,
            str(repo / "scripts" / "collect_data.py"),
            "--out",
            args.out,
            "--subject-id",
            args.subject_id,
            "--camera",
            args.camera,
            "--labels",
            *args.labels,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "collect-seq":
        cmd = [
            py,
            str(repo / "scripts" / "collect_sequence.py"),
            "--out",
            args.out,
            "--subject-id",
            args.subject_id,
            "--camera",
            int(args.camera),
            "--duration",
            float(args.duration),
            "--labels",
            *args.labels,
        ]
        cmd = [str(x) for x in cmd]
        raise SystemExit(_run(cmd))

    if args.cmd == "collect-mm-seq":
        cmd = [
            py,
            str(repo / "scripts" / "collect_multimodal_sequence.py"),
            "--out",
            args.out,
            "--subject-id",
            args.subject_id,
            "--camera",
            int(args.camera),
            "--duration",
            float(args.duration),
            "--labels",
            *args.labels,
        ]
        cmd = [str(x) for x in cmd]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-landmarks":
        cmd = [py, str(repo / "scripts" / "train_landmarks.py"), "--csv", args.csv, "--out", args.out]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-cnn":
        cmd = [
            py,
            str(repo / "scripts" / "train_cnn.py"),
            "--out",
            args.out,
            "--epochs",
            args.epochs,
            "--device",
            args.device,
        ]
        if args.img_dir:
            cmd += ["--img-dir", args.img_dir]
        if args.csv:
            cmd += ["--csv", args.csv, "--image-col", args.image_col]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        if args.unfreeze_backbone:
            cmd += ["--unfreeze-backbone"]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-seq":
        cmd = [
            py,
            str(repo / "scripts" / "train_sequence.py"),
            "--seq-dir",
            args.seq_dir,
            "--out",
            args.out,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-mm-seq":
        cmd = [
            py,
            str(repo / "scripts" / "train_multimodal_sequence.py"),
            "--seq-dir",
            args.seq_dir,
            "--out",
            args.out,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-mm-stack":
        cmd = [
            py,
            str(repo / "scripts" / "train_multimodal_stack.py"),
            "--work-dir",
            args.work_dir,
            "--window",
            args.window,
            "--min-frames",
            args.min_frames,
            "--group-col",
            args.group_col,
        ]
        if args.seq_dir:
            cmd += ["--seq-dir", args.seq_dir]
        if args.seq_dirs:
            cmd += ["--seq-dirs", *args.seq_dirs]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-videos-stack":
        cmd = [
            py,
            str(repo / "scripts" / "train_videos_stack.py"),
            "--videos-root",
            args.videos_root,
            "--work-dir",
            args.work_dir,
            "--layout",
            args.layout,
            "--subject-default",
            args.subject_default,
            "--fps",
            args.fps,
            "--max-per-video",
            args.max_per_video,
            "--cnn-image-col",
            args.cnn_image_col,
            "--group-col",
            args.group_col,
            "--cnn-epochs",
            args.cnn_epochs,
            "--cnn-device",
            args.cnn_device,
            "--ablation-seconds",
            args.ablation_seconds,
            "--camera",
            args.camera,
            "--exts",
            *args.exts,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.skin_mask:
            cmd.append("--skin-mask")
        if args.camera_like:
            cmd.append("--camera-like")
        if args.skip_ablation:
            cmd.append("--skip-ablation")
        raise SystemExit(_run(cmd))

    if args.cmd == "train-videos-mm-stack":
        cmd = [
            py,
            str(repo / "scripts" / "train_videos_mm_stack.py"),
            "--videos-root",
            args.videos_root,
            "--work-dir",
            args.work_dir,
            "--layout",
            args.layout,
            "--subject-default",
            args.subject_default,
            "--fps",
            args.fps,
            "--window",
            args.window,
            "--min-frames",
            args.min_frames,
            "--group-col",
            args.group_col,
            "--exts",
            *args.exts,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        raise SystemExit(_run(cmd))

    if args.cmd == "train-stack":
        cmd = [
            py,
            str(repo / "scripts" / "train_stack.py"),
            "--work-dir",
            args.work_dir,
            "--cnn-image-col",
            args.cnn_image_col,
            "--group-col",
            args.group_col,
            "--cnn-epochs",
            args.cnn_epochs,
            "--cnn-device",
            args.cnn_device,
            "--ablation-seconds",
            args.ablation_seconds,
            "--camera",
            args.camera,
        ]
        if args.csv:
            cmd += ["--csv", args.csv]
        if args.csvs:
            cmd += ["--csvs", *args.csvs]
        if args.skip_ablation:
            cmd.append("--skip-ablation")
        raise SystemExit(_run(cmd))

    if args.cmd == "demo-stack-web":
        cmd = [
            py,
            str(repo / "scripts" / "launch_stack_demo.py"),
            "--work-dir",
            args.work_dir,
            "--host",
            args.host,
            "--port",
            args.port,
        ]
        if args.open_browser:
            cmd.append("--open-browser")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-train":
        cmd = [
            py,
            str(repo / "scripts" / "train_ilsut.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--fps",
            args.fps,
            "--max-per-seg",
            args.max_per_seg,
            "--group-col",
            args.group_col,
            "--cnn-epochs",
            args.cnn_epochs,
            "--cnn-device",
            args.cnn_device,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
            "--pipelines",
            *args.pipelines,
            "--ablation-seconds",
            args.ablation_seconds,
            "--camera",
            args.camera,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.skin_mask:
            cmd.append("--skin-mask")
        if args.camera_like:
            cmd.append("--camera-like")
        if args.cnn_use_masked:
            cmd.append("--cnn-use-masked")
        if args.skip_ablation:
            cmd.append("--skip-ablation")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-train-mm":
        cmd = [
            py,
            str(repo / "scripts" / "train_ilsut_multimodal.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--fps",
            args.fps,
            "--max-per-seg",
            args.max_per_seg,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
            "--window",
            args.window,
            "--min-frames",
            args.min_frames,
            "--group-col",
            args.group_col,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        if args.preprocess:
            cmd.append("--preprocess")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-preset":
        cmd = [
            py,
            str(repo / "scripts" / "run_ilsut_preset.py"),
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--source",
            args.source,
            "--mode",
            args.mode,
            "--preset",
            args.preset,
            "--work-root",
            args.work_root,
            "--cnn-device",
            args.cnn_device,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-download":
        cmd = [
            py,
            str(repo / "scripts" / "download_ilsut.py"),
            "--manifest",
            args.manifest,
            "--out-dir",
            args.out_dir,
            "--sources",
            *args.sources,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-extract":
        cmd = [
            py,
            str(repo / "scripts" / "extract_ilsut.py"),
            "--archives-dir",
            args.archives_dir,
            "--out-root",
            args.out_root,
            "--sources",
            *args.sources,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-convert-videos":
        cmd = [
            py,
            str(repo / "scripts" / "convert_ilsut_videos.py"),
            "--root",
            args.root,
            "--sources",
            *args.sources,
            "--input-ext",
            args.input_ext,
            "--output-ext",
            args.output_ext,
            "--tool",
            args.tool,
            "--quality",
            args.quality,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.delete_source:
            cmd.append("--delete-source")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-build-episodes-csv":
        cmd = [
            py,
            str(repo / "scripts" / "build_ilsut_episodes_csv.py"),
            "--root",
            args.root,
            "--out",
            args.out,
            "--video-dir-name",
            args.video_dir_name,
            "--whisperx-dir-name",
            args.whisperx_dir_name,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        if args.include_unmatched:
            cmd.append("--include-unmatched")
        if args.strict:
            cmd.append("--strict")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-prepare-slt-subset-whisperx":
        sources = list(args.sources or [])
        cmd = [
            py,
            str(repo / "scripts" / "prepare_whisperx_slt_subset.py"),
            "--root",
            args.root,
            "--work-dir",
            args.work_dir,
            "--episodes-csv",
            args.episodes_csv,
            "--video-dir-name",
            args.video_dir_name,
            "--whisperx-dir-name",
            args.whisperx_dir_name,
            "--test-size",
            args.test_size,
            "--val-size",
            args.val_size,
            "--seed",
            args.seed,
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
            "--clip-ext",
            args.clip_ext,
            "--dedup-eval-text",
            args.dedup_eval_text,
        ]
        if sources:
            cmd += ["--sources", *sources]
        if args.keep_punctuation:
            cmd.append("--keep-punctuation")
        if args.export_clips:
            cmd.append("--export-clips")
        if int(args.max_clips) > 0:
            cmd += ["--max-clips", args.max_clips]
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-analyze-support":
        cmd = [
            py,
            str(repo / "scripts" / "analyze_ilsut_support.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-prepare-slt-subset":
        cmd = [
            py,
            str(repo / "scripts" / "prepare_ilsut_slt_subset.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
            "--labels-json",
            args.labels_json,
            "--test-size",
            args.test_size,
            "--val-size",
            args.val_size,
            "--seed",
            args.seed,
            "--clip-ext",
            args.clip_ext,
            "--max-clips",
            args.max_clips,
            "--dedup-eval-text",
            args.dedup_eval_text,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        if args.export_clips:
            cmd.append("--export-clips")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-audit-keywords":
        cmd = [
            py,
            str(repo / "scripts" / "audit_ilsut_keywords.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--top-k",
            args.top_k,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-extract-slt-features":
        cmd = [
            py,
            str(repo / "scripts" / "extract_ilsut_slt_features.py"),
            "--subset-dir",
            args.subset_dir,
            "--out-dir",
            args.out_dir,
            "--sample-fps",
            args.sample_fps,
            "--max-frames",
            args.max_frames,
            "--limit",
            args.limit,
        ]
        if args.splits:
            cmd += ["--splits", *args.splits]
        if args.preprocess:
            cmd.append("--preprocess")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-export-slt-dataset":
        cmd = [
            py,
            str(repo / "scripts" / "export_ilsut_slt_dataset.py"),
            "--subset-dir",
            args.subset_dir,
            "--out-dir",
            args.out_dir,
            "--mode",
            args.mode,
            "--sample-fps",
            args.sample_fps,
            "--max-frames",
            args.max_frames,
            "--clip-ext",
            args.clip_ext,
            "--max-clips",
            args.max_clips,
            "--backend",
            args.backend,
            "--limit",
            args.limit,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        raise SystemExit(_run(cmd))

    if args.cmd == "train-ilsut-slt":
        cmd = [
            py,
            str(repo / "scripts" / "train_ilsut_slt.py"),
            "--subset-dir",
            args.subset_dir,
            "--out-dir",
            args.out_dir,
            "--dataset-dir",
            args.dataset_dir,
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
            cmd.append("--preprocess")
        if args.run_backend:
            cmd.append("--run-backend")
        raise SystemExit(_run(cmd))

    if args.cmd == "eval-ilsut-slt":
        cmd = [
            py,
            str(repo / "scripts" / "eval_ilsut_slt.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--model",
            args.model,
            "--json-out",
            args.json_out,
            "--md-out",
            args.md_out,
            "--landmarks-eval-json",
            args.landmarks_eval_json,
            "--cnn-eval-json",
            args.cnn_eval_json,
            "--multimodal-eval-json",
            args.multimodal_eval_json,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "validate-ilsut-slt-dataset":
        cmd = [
            py,
            str(repo / "scripts" / "validate_ilsut_slt_dataset.py"),
            "--dataset-dir",
            args.dataset_dir,
        ]
        if args.json_out:
            cmd += ["--json-out", args.json_out]
        if args.md_out:
            cmd += ["--md-out", args.md_out]
        if args.require_features:
            cmd.append("--require-features")
        raise SystemExit(_run(cmd))

    if args.cmd == "run-ilsut-slt-pipeline":
        cmd = [
            py,
            str(repo / "scripts" / "run_ilsut_slt_pipeline.py"),
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-root",
            args.work_root,
            "--preset",
            args.preset,
            "--episodes-csv",
            args.episodes_csv,
            "--path-mode",
            args.path_mode,
            "--manifest-limit",
            args.manifest_limit,
            "--min-label-count",
            args.min_label_count,
            "--labels-json",
            args.labels_json,
            "--sample-fps",
            args.sample_fps,
            "--max-frames",
            args.max_frames,
            "--clip-ext",
            args.clip_ext,
            "--max-clips",
            args.max_clips,
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
            "--sources",
            *args.sources,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.run_backend:
            cmd.append("--run-backend")
        raise SystemExit(_run(cmd))

    if args.cmd == "run-whisperx-slt-pipeline":
        cmd = [
            py,
            str(repo / "scripts" / "run_whisperx_slt_pipeline.py"),
            "--root",
            args.root,
            "--sources",
            *args.sources,
            "--work-root",
            args.work_root,
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
            "--sample-fps",
            args.sample_fps,
            "--max-frames",
            args.max_frames,
            "--clip-ext",
            args.clip_ext,
            "--max-clips",
            args.max_clips,
            "--limit",
            args.limit,
            "--dedup-eval-text",
            args.dedup_eval_text,
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
        ]
        if args.keep_punctuation:
            cmd.append("--keep-punctuation")
        if args.preprocess:
            cmd.append("--preprocess")
        if args.run_backend:
            cmd.append("--run-backend")
        raise SystemExit(_run(cmd))

    if args.cmd == "render-ilsut-slt-summary":
        cmd = [
            py,
            str(repo / "scripts" / "render_ilsut_slt_summary.py"),
            "--summary-json",
            args.summary_json,
            "--out-dir",
            args.out_dir,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "render-ilsut-slt-sections":
        cmd = [
            py,
            str(repo / "scripts" / "render_ilsut_slt_sections.py"),
            "--summary-json",
            args.summary_json,
            "--out-dir",
            args.out_dir,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "validate-videos":
        cmd = [
            py,
            str(repo / "scripts" / "validate_videos.py"),
            "--pipeline",
            args.pipeline,
            "--out-dir",
            args.out_dir,
            "--mode",
            args.mode,
            "--mask-space",
            args.mask_space,
            "--confidence-threshold",
            args.confidence_threshold,
            "--stable-frames-min",
            args.stable_frames_min,
            "--pause-ms-min",
            args.pause_ms_min,
            "--cooldown-ms",
            args.cooldown_ms,
            "--sample-fps",
            args.sample_fps,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
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
        raise SystemExit(_run(cmd))

    if args.cmd == "compare-video-pipelines":
        cmd = [
            py,
            str(repo / "scripts" / "compare_video_pipelines.py"),
            "--pipelines",
            *args.pipelines,
            "--out-dir",
            args.out_dir,
            "--mode",
            args.mode,
            "--mask-space",
            args.mask_space,
            "--confidence-threshold",
            args.confidence_threshold,
            "--stable-frames-min",
            args.stable_frames_min,
            "--pause-ms-min",
            args.pause_ms_min,
            "--cooldown-ms",
            args.cooldown_ms,
            "--sample-fps",
            args.sample_fps,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
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
        raise SystemExit(_run(cmd))

    if args.cmd == "recommend-demo-pipeline":
        cmd = [
            py,
            str(repo / "scripts" / "recommend_demo_pipeline.py"),
            "--compare-json",
            args.compare_json,
            "--out-dir",
            args.out_dir,
            "--host",
            args.host,
            "--port",
            args.port,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
        if args.cases_json:
            cmd += ["--cases-json", args.cases_json]
        raise SystemExit(_run(cmd))

    if args.cmd == "finalize-demo-selection":
        cmd = [
            py,
            str(repo / "scripts" / "finalize_demo_selection.py"),
            "--out-dir",
            args.out_dir,
            "--mode",
            args.mode,
            "--mask-space",
            args.mask_space,
            "--confidence-threshold",
            args.confidence_threshold,
            "--stable-frames-min",
            args.stable_frames_min,
            "--pause-ms-min",
            args.pause_ms_min,
            "--cooldown-ms",
            args.cooldown_ms,
            "--sample-fps",
            args.sample_fps,
            "--host",
            args.host,
            "--port",
            args.port,
        ]
        if args.cases_json:
            cmd += ["--cases-json", args.cases_json]
        if args.videos:
            cmd += ["--videos", *args.videos]
        if args.frame_work_dir:
            cmd += ["--frame-work-dir", args.frame_work_dir]
        if args.multimodal_work_dir:
            cmd += ["--multimodal-work-dir", args.multimodal_work_dir]
        if args.slt_work_dir:
            cmd += ["--slt-work-dir", args.slt_work_dir]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.multimodal_model:
            cmd += ["--multimodal-model", args.multimodal_model]
        if args.slt_model:
            cmd += ["--slt-model", args.slt_model]
        if args.pipelines:
            cmd += ["--pipelines", *args.pipelines]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.skin_mask:
            cmd.append("--skin-mask")
        if args.use_tracker:
            cmd.append("--use-tracker")
        raise SystemExit(_run(cmd))

    if args.cmd == "eval-split":
        cmd = [
            py,
            str(repo / "scripts" / "eval_split.py"),
            "--csv",
            args.csv,
            "--cnn-image-col",
            args.cnn_image_col,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        if args.json_out:
            cmd += ["--json-out", args.json_out]
        raise SystemExit(_run(cmd))

    if args.cmd == "ablation":
        cmd = [
            py,
            str(repo / "scripts" / "run_ablation.py"),
            "--csv",
            args.csv,
            "--seconds",
            args.seconds,
            "--camera",
            args.camera,
            "--fps-preprocess",
            args.fps_preprocess,
            "--fps-tracker",
            args.fps_tracker,
            "--fps-skin-mask",
            args.fps_skin_mask,
            "--fps-mask-space",
            args.fps_mask_space,
            "--out-md",
            args.out_md,
            "--out-json",
            args.out_json,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        raise SystemExit(_run(cmd))

    if args.cmd == "ablation-grid":
        cmd = [
            py,
            str(repo / "scripts" / "run_ablation_grid.py"),
            "--csv",
            args.csv,
            "--seconds",
            args.seconds,
            "--camera",
            args.camera,
            "--out-md",
            args.out_md,
            "--out-json",
            args.out_json,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        raise SystemExit(_run(cmd))

    if args.cmd == "dataset-stats":
        cmd = [py, str(repo / "scripts" / "dataset_stats.py"), "--csv", args.csv, "--by", args.by]
        if args.out_md:
            cmd += ["--out-md", args.out_md]
        raise SystemExit(_run(cmd))

    if args.cmd == "validate-multisubject":
        cmd = [
            py,
            str(repo / "scripts" / "validate_multisubject.py"),
            "--csv",
            args.csv,
            "--min-per-label-per-subject",
            args.min_per_label_per_subject,
        ]
        if args.out_md:
            cmd += ["--out-md", args.out_md]
        raise SystemExit(_run(cmd))

    if args.cmd == "merge-csvs":
        cmd = [py, str(repo / "scripts" / "merge_subject_csvs.py"), "--out", args.out, "--inputs", *args.inputs]
        raise SystemExit(_run(cmd))

    if args.cmd == "report":
        cmd = [
            py,
            str(repo / "scripts" / "make_report.py"),
            "--ablation-json",
            args.ablation_json,
            "--ablation-table",
            args.ablation_table,
            "--dataset-stats-md",
            args.dataset_stats_md,
            "--out-md",
            args.out_md,
            "--out-fig",
            args.out_fig,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "deliverables":
        # Optionally update config before building.
        if (
            args.set_title
            or args.set_course
            or args.set_group
            or args.set_date
            or (args.set_members is not None and len(args.set_members) > 0)
            or int(args.set_pitch_minutes) > 0
        ):
            cmd0 = [
                py,
                str(repo / "scripts" / "update_deliverables_config.py"),
                "--config",
                args.config,
            ]
            if args.set_title:
                cmd0 += ["--title", args.set_title]
            if args.set_course:
                cmd0 += ["--course", args.set_course]
            if args.set_group:
                cmd0 += ["--group", args.set_group]
            if args.set_date:
                cmd0 += ["--date", args.set_date]
            if int(args.set_pitch_minutes) > 0:
                cmd0 += ["--pitch-minutes", args.set_pitch_minutes]
            if args.set_members is not None and len(args.set_members) > 0:
                cmd0 += ["--members", *args.set_members]
            rc0 = _run(cmd0)
            if rc0 != 0:
                raise SystemExit(rc0)

        slt_report_section = args.slt_report_section
        slt_pitch_section = args.slt_pitch_section
        if args.slt_summary_json:
            slt_sections_dir = repo / "deliverables" / "out" / "_slt_sections"
            # Render SLT sections. Support both the iLSU-T keyword-based summary and the WhisperX segment-based summary.
            try:
                slt_summary = json.loads(Path(args.slt_summary_json).read_text(encoding="utf-8"))
            except Exception:
                slt_summary = {}
            renderer = "render_whisperx_slt_sections.py" if "eval_proxy" in slt_summary else "render_ilsut_slt_sections.py"
            cmd_sections = [py, str(repo / "scripts" / renderer), "--summary-json", args.slt_summary_json, "--out-dir", str(slt_sections_dir)]
            rc_sections = _run(cmd_sections)
            if rc_sections != 0:
                raise SystemExit(rc_sections)
            slt_report_section = str(slt_sections_dir / "report_results_section.md")
            slt_pitch_section = str(slt_sections_dir / "pitch_results_section.md")

        cmd1 = [py, str(repo / "scripts" / "build_deliverables.py"), "--config", args.config]
        if slt_report_section:
            cmd1 += ["--slt-report-section", slt_report_section]
        if slt_pitch_section:
            cmd1 += ["--slt-pitch-section", slt_pitch_section]
        rc1 = _run(cmd1)
        if rc1 != 0:
            raise SystemExit(rc1)
        cmd2 = [py, str(repo / "scripts" / "package_submission.py")]
        if args.include_data:
            cmd2.append("--include-data")
        if args.include_models:
            cmd2.append("--include-models")
        rc2 = _run(cmd2)
        if rc2 != 0:
            raise SystemExit(rc2)
        cmd3 = [py, str(repo / "scripts" / "check_deliverables.py"), "--config", args.config]
        raise SystemExit(_run(cmd3))


if __name__ == "__main__":
    main()
