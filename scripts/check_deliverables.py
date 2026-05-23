from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="deliverables/config.json")
    p.add_argument("--out-dir", default="deliverables/out")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / args.config
    out_dir = repo / args.out_dir
    problems: list[str] = []

    if not cfg_path.exists():
        problems.append(f"Missing config: {cfg_path}")
        cfg = {}
    else:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(f"Using config: {cfg_path}")

    def _must(k: str) -> None:
        v = cfg.get(k)
        if not v:
            problems.append(f"Config missing/empty: {k}")
        if isinstance(v, str) and "Completar" in v:
            problems.append(f"Config placeholder not filled: {k}")
        if isinstance(v, list) and any(isinstance(x, str) and "Completar" in x for x in v):
            problems.append(f"Config placeholder not filled: {k}")

    for k in ["title", "course", "group", "members", "date"]:
        _must(k)

    report_pdf = out_dir / "informe.pdf"
    pitch_pdf = out_dir / "pitch.pdf"
    pitch_pptx = out_dir / "pitch.pptx"
    zip_path = out_dir / "entrega_moodle.zip"
    for p in [report_pdf, pitch_pdf, pitch_pptx, zip_path]:
        if not p.exists():
            problems.append(f"Missing output: {p}")
        elif p.stat().st_size < 1000:
            problems.append(f"Output too small/suspicious: {p}")

    if problems:
        print("DELIVERABLES CHECK: FAIL")
        for pr in problems:
            print("-", pr)
        # Non-zero exit so CI/users can catch it, but keep artifacts generated.
        raise SystemExit(2)

    print("DELIVERABLES CHECK: OK")
    print("-", report_pdf)
    print("-", pitch_pdf)
    print("-", pitch_pptx)
    print("-", zip_path)


if __name__ == "__main__":
    main()
