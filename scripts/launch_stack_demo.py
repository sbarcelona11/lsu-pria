from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the web demo using the models produced by a training workspace.")
    p.add_argument("--work-dir", required=True, help="Workspace previously created by train-stack")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="8000")
    p.add_argument("--open-browser", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir)
    models_dir = work_dir / "models"

    landmarks_model = models_dir / "landmarks.joblib"
    cnn_model = models_dir / "cnn.pt"
    sequence_model = models_dir / "sequence.joblib"
    multimodal_model = models_dir / "multimodal_sequence.joblib"
    if not any(p.exists() for p in (landmarks_model, cnn_model, sequence_model, multimodal_model)):
        raise SystemExit(f"No known models found under: {models_dir}")

    cmd = [
        sys.executable or "python",
        str(repo / "scripts" / "run_webapp.py"),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    if landmarks_model.exists():
        cmd += ["--landmarks-model", str(landmarks_model)]
    if cnn_model.exists():
        cmd += ["--cnn-model", str(cnn_model)]
    if sequence_model.exists():
        cmd += ["--sequence-model", str(sequence_model)]
    if multimodal_model.exists():
        cmd += ["--multimodal-model", str(multimodal_model)]
    if args.open_browser:
        cmd.append("--open-browser")

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
