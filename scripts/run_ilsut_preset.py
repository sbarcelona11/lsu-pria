from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a practical preset for iLSU-T training on the extracted dataset.")
    p.add_argument("--root", required=True)
    p.add_argument("--keywords", default="deliverables/ilsut_keywords.example.json")
    p.add_argument("--source", default="source2")
    p.add_argument("--mode", choices=["frame", "multimodal", "both"], default="both")
    p.add_argument("--preset", choices=["quick", "standard"], default="quick")
    p.add_argument("--work-root", default="runs/ilsut_presets")
    p.add_argument("--cnn-device", default="cpu")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    rc = subprocess.call([str(x) for x in cmd])
    if rc != 0:
        raise SystemExit(rc)


def _frame_params(preset: str) -> dict[str, str | float | int | bool]:
    if preset == "standard":
        return {
            "fps": 4.0,
            "max_per_seg": 24,
            "cnn_epochs": 8,
            "skip_ablation": False,
            "manifest_limit": 0,
            "min_label_count": 0,
        }
    return {
        "fps": 2.0,
        "max_per_seg": 12,
        "cnn_epochs": 3,
        "skip_ablation": True,
        "manifest_limit": 12,
        "min_label_count": 20,
    }


def _multimodal_params(preset: str) -> dict[str, str | float | int]:
    if preset == "standard":
        return {"fps": 5.0, "max_per_seg": 24, "window": 24, "min_frames": 8, "manifest_limit": 0, "min_label_count": 0}
    return {"fps": 3.0, "max_per_seg": 16, "window": 20, "min_frames": 6, "manifest_limit": 12, "min_label_count": 20}


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    if args.mode in {"frame", "both"}:
        fp = _frame_params(args.preset)
        work_dir = work_root / f"{args.source}_{args.preset}_frame"
        cmd: list[object] = [
            py,
            repo / "scripts" / "train_ilsut.py",
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            work_dir,
            "--sources",
            args.source,
            "--pipelines",
            "cnn",
            "landmarks",
            "--fps",
            fp["fps"],
            "--max-per-seg",
            fp["max_per_seg"],
            "--preprocess",
            "--skin-mask",
            "--camera-like",
            "--cnn-epochs",
            fp["cnn_epochs"],
            "--cnn-device",
            args.cnn_device,
            "--manifest-limit",
            fp["manifest_limit"],
            "--min-label-count",
            fp["min_label_count"],
        ]
        if bool(fp["skip_ablation"]):
            cmd.append("--skip-ablation")
        _run([str(x) for x in cmd])

    if args.mode in {"multimodal", "both"}:
        mp = _multimodal_params(args.preset)
        work_dir = work_root / f"{args.source}_{args.preset}_multimodal"
        cmd = [
            py,
            repo / "scripts" / "train_ilsut_multimodal.py",
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            work_dir,
            "--sources",
            args.source,
            "--fps",
            mp["fps"],
            "--max-per-seg",
            mp["max_per_seg"],
            "--preprocess",
            "--window",
            mp["window"],
            "--min-frames",
            mp["min_frames"],
            "--manifest-limit",
            mp["manifest_limit"],
            "--min-label-count",
            mp["min_label_count"],
        ]
        _run([str(x) for x in cmd])


if __name__ == "__main__":
    main()
