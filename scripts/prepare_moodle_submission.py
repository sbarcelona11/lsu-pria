from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="deliverables/config.json")
    p.add_argument("--include-data", action="store_true")
    p.add_argument("--include-models", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"

    _run([py, str(repo / "scripts" / "build_deliverables.py"), "--config", args.config])
    pkg = [py, str(repo / "scripts" / "package_submission.py")]
    if args.include_data:
        pkg.append("--include-data")
    if args.include_models:
        pkg.append("--include-models")
    _run(pkg)
    _run([py, str(repo / "scripts" / "check_deliverables.py"), "--config", args.config])

    out = repo / "deliverables" / "out"
    print("READY:")
    print("-", out / "informe.pdf")
    print("-", out / "pitch.pdf")
    print("-", out / "entrega_moodle.zip")


if __name__ == "__main__":
    main()

