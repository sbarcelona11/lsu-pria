from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


@dataclass(frozen=True)
class FpsConfig:
    preprocess: str  # on/off
    tracker: str  # on/off
    skin_mask: str  # on/off
    mask_space: str  # ycrcb/hsv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--landmarks-model", type=str, default="")
    p.add_argument("--cnn-model", type=str, default="")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--group-col", type=str, default="", help="e.g. subject_id")
    p.add_argument("--out-md", type=str, default="results/ablation_grid.md")
    p.add_argument("--out-json", type=str, default="results/ablation_grid.json")
    p.add_argument(
        "--fps-preprocess",
        nargs="+",
        default=["on", "off"],
        choices=["on", "off"],
        help="Values to sweep",
    )
    p.add_argument("--fps-tracker", nargs="+", default=["on", "off"], choices=["on", "off"])
    p.add_argument("--fps-skin-mask", nargs="+", default=["off", "on"], choices=["on", "off"])
    p.add_argument("--fps-mask-space", nargs="+", default=["ycrcb", "hsv"], choices=["ycrcb", "hsv"])
    p.add_argument(
        "--cnn-image-cols",
        nargs="+",
        default=["auto", "img_masked_path", "img_raw_path", "img_path"],
        help="Order of preference; unavailable cols are skipped.",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _python() -> str:
    return sys.executable or "python"


def _detect_cols(csv_path: Path) -> set[str]:
    try:
        cols = set(pd.read_csv(csv_path, nrows=1).columns.tolist())
    except Exception:
        cols = set()
    return cols


def _choose_cnn_cols(available: set[str], requested: list[str]) -> list[str]:
    out = []
    for c in requested:
        if c == "auto":
            out.append("auto")
        elif c in available:
            out.append(c)
    # ensure unique while preserving order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    cols = _detect_cols(csv_path)
    cnn_cols = _choose_cnn_cols(cols, list(args.cnn_image_cols))

    fps_grid = [
        FpsConfig(p, t, s, m)
        for (p, t, s, m) in itertools.product(args.fps_preprocess, args.fps_tracker, args.fps_skin_mask, args.fps_mask_space)
    ]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_json.parent

    results: list[dict] = []

    # Precompute eval macro-f1 (not dependent on fps toggles).
    eval_payloads = {}
    if args.landmarks_model:
        eval_out = work_dir / "grid_eval_landmarks.json"
        cmd = [
            _python(),
            "scripts/eval_split.py",
            "--csv",
            str(csv_path),
            "--test-size",
            str(args.test_size),
            "--seed",
            str(args.seed),
            "--json-out",
            str(eval_out),
            "--landmarks-model",
            args.landmarks_model,
        ]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        _run(cmd)
        eval_payloads["landmarks"] = json.loads(eval_out.read_text(encoding="utf-8"))["results"]["landmarks"]

    if args.cnn_model:
        for col in cnn_cols:
            eval_out = work_dir / f"grid_eval_cnn_{col}.json"
            cmd = [
                _python(),
                "scripts/eval_split.py",
                "--csv",
                str(csv_path),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(args.seed),
                "--json-out",
                str(eval_out),
                "--cnn-model",
                args.cnn_model,
                "--cnn-image-col",
                col,
            ]
            if args.group_col:
                cmd += ["--group-col", args.group_col]
            _run(cmd)
            eval_payloads[f"cnn:{col}"] = json.loads(eval_out.read_text(encoding="utf-8"))["results"]["cnn"]

    # FPS profiling across toggles.
    for fps_cfg in fps_grid:
        if args.landmarks_model:
            fps_out = work_dir / f"grid_fps_landmarks_{fps_cfg.preprocess}_{fps_cfg.tracker}_{fps_cfg.skin_mask}_{fps_cfg.mask_space}.json"
            cmd = [
                _python(),
                "scripts/profile_fps.py",
                "--pipeline",
                "landmarks",
                "--model",
                args.landmarks_model,
                "--seconds",
                str(args.seconds),
                "--camera",
                str(args.camera),
                "--json-out",
                str(fps_out),
            ]
            if fps_cfg.preprocess == "off":
                cmd += ["--no-preprocess"]
            if fps_cfg.tracker == "off":
                cmd += ["--no-tracker"]
            if fps_cfg.skin_mask == "on":
                cmd += ["--skin-mask", "--mask-space", fps_cfg.mask_space]
            else:
                cmd += ["--mask-space", fps_cfg.mask_space]
            _run(cmd)
            fps_payload = json.loads(fps_out.read_text(encoding="utf-8"))
            results.append(
                {
                    "name": "landmarks",
                    "pipeline": "landmarks",
                    "macro_f1": eval_payloads.get("landmarks", {}).get("macro_f1"),
                    "fps": fps_payload.get("fps"),
                    "fps_config": fps_cfg.__dict__,
                    "eval": eval_payloads.get("landmarks", {}),
                }
            )

        if args.cnn_model:
            for col in cnn_cols:
                fps_out = work_dir / f"grid_fps_cnn_{col}_{fps_cfg.preprocess}_{fps_cfg.tracker}_{fps_cfg.skin_mask}_{fps_cfg.mask_space}.json"
                cmd = [
                    _python(),
                    "scripts/profile_fps.py",
                    "--pipeline",
                    "cnn",
                    "--model",
                    args.cnn_model,
                    "--seconds",
                    str(args.seconds),
                    "--camera",
                    str(args.camera),
                    "--json-out",
                    str(fps_out),
                ]
                if fps_cfg.preprocess == "off":
                    cmd += ["--no-preprocess"]
                if fps_cfg.tracker == "off":
                    cmd += ["--no-tracker"]
                if fps_cfg.skin_mask == "on":
                    cmd += ["--skin-mask", "--mask-space", fps_cfg.mask_space]
                else:
                    cmd += ["--mask-space", fps_cfg.mask_space]
                _run(cmd)
                fps_payload = json.loads(fps_out.read_text(encoding="utf-8"))
                eval_key = f"cnn:{col}"
                results.append(
                    {
                        "name": f"cnn({col})",
                        "pipeline": "cnn",
                        "cnn_image_col": col,
                        "macro_f1": eval_payloads.get(eval_key, {}).get("macro_f1"),
                        "fps": fps_payload.get("fps"),
                        "fps_config": fps_cfg.__dict__,
                        "eval": eval_payloads.get(eval_key, {}),
                    }
                )

    payload = {
        "csv": str(csv_path),
        "seconds": float(args.seconds),
        "camera": int(args.camera),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "group_col": args.group_col,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    df = pd.DataFrame(results)
    # Flatten fps_config for table readability.
    if "fps_config" in df.columns:
        fps_df = pd.json_normalize(df["fps_config"])
        fps_df.columns = [f"fps_{c}" for c in fps_df.columns]
        df = pd.concat([df.drop(columns=["fps_config"]), fps_df], axis=1)

    df = df.sort_values(["macro_f1", "fps"], ascending=False)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

