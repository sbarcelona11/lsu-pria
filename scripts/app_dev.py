from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backend and frontend dev servers together.")
    p.add_argument("--landmarks-model", default="")
    p.add_argument("--cnn-model", default="")
    p.add_argument("--sequence-model", default="")
    p.add_argument("--multimodal-model", default="")
    p.add_argument("--stack-work-dir", default="", help="If set, auto-pick models from <work-dir>/models/")
    p.add_argument("--api-host", default="127.0.0.1")
    p.add_argument("--api-port", default="8000")
    p.add_argument("--ui-port", default="5173")
    p.add_argument("--install-ui", action="store_true", help="Run `npm install` in web-ui before starting")
    return p.parse_args()


def _resolve_models(repo: Path, args: argparse.Namespace) -> tuple[str, str, str, str]:
    landmarks = args.landmarks_model
    cnn = args.cnn_model
    sequence = args.sequence_model
    multimodal = args.multimodal_model
    if args.stack_work_dir:
        models_dir = Path(args.stack_work_dir) / "models"
        if not landmarks:
            cand = models_dir / "landmarks.joblib"
            if cand.exists():
                landmarks = str(cand)
        if not cnn:
            cand = models_dir / "cnn.pt"
            if cand.exists():
                cnn = str(cand)
        if not sequence:
            cand = models_dir / "sequence.joblib"
            if cand.exists():
                sequence = str(cand)
        if not multimodal:
            cand = models_dir / "multimodal_sequence.joblib"
            if cand.exists():
                multimodal = str(cand)
    return landmarks, cnn, sequence, multimodal


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    web_ui = repo / "web-ui"
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("`npm` is required to run the frontend dev server.")

    if not (web_ui / "package.json").exists():
        raise SystemExit(f"Missing frontend package.json: {web_ui / 'package.json'}")

    if args.install_ui:
        rc = subprocess.call([npm, "install"], cwd=str(web_ui))
        if rc != 0:
            raise SystemExit(rc)
    elif not (web_ui / "node_modules").exists():
        raise SystemExit("web-ui/node_modules is missing. Run with --install-ui or install dependencies manually.")

    landmarks, cnn, sequence, multimodal = _resolve_models(repo, args)

    backend_cmd = [
        sys.executable or "python",
        str(repo / "scripts" / "run_webapp.py"),
        "--host",
        args.api_host,
        "--port",
        args.api_port,
    ]
    if landmarks:
        backend_cmd += ["--landmarks-model", landmarks]
    if cnn:
        backend_cmd += ["--cnn-model", cnn]
    if sequence:
        backend_cmd += ["--sequence-model", sequence]
    if multimodal:
        backend_cmd += ["--multimodal-model", multimodal]

    frontend_cmd = [npm, "run", "dev", "--", "--port", args.ui_port]

    env = os.environ.copy()
    env["VITE_API_BASE"] = f"http://{args.api_host}:{args.api_port}"

    backend = subprocess.Popen(backend_cmd, cwd=str(repo))
    frontend = subprocess.Popen(frontend_cmd, cwd=str(web_ui), env=env)

    print(f"Backend: http://{args.api_host}:{args.api_port}")
    print(f"Frontend: http://127.0.0.1:{args.ui_port}")

    def _handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        _terminate(frontend)
        _terminate(backend)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            if backend.poll() is not None:
                _terminate(frontend)
                raise SystemExit(backend.returncode or 0)
            if frontend.poll() is not None:
                _terminate(backend)
                raise SystemExit(frontend.returncode or 0)
            time.sleep(0.5)
    finally:
        _terminate(frontend)
        _terminate(backend)


if __name__ == "__main__":
    main()
