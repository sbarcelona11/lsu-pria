from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


@dataclass
class AblationCase:
    name: str
    pipeline: str  # landmarks | cnn
    model: str
    cnn_image_col: str = "img_masked_path"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--landmarks-model", type=str, default="")
    p.add_argument("--cnn-model", type=str, default="")
    p.add_argument("--cnn-image-col", type=str, default="img_masked_path")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out-md", type=str, default="results/ablation_table.md")
    p.add_argument("--out-json", type=str, default="results/ablation_results.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--group-col", type=str, default="", help="e.g. subject_id for group split")
    p.add_argument("--fps-preprocess", choices=["on", "off"], default="on")
    p.add_argument("--fps-tracker", choices=["on", "off"], default="on")
    p.add_argument("--fps-skin-mask", choices=["on", "off"], default="off")
    p.add_argument("--fps-mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    return p.parse_args()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _python() -> str:
    return sys.executable or "python"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Detect available image columns to decide CNN ablations.
    try:
        import pandas as _pd

        _cols = set(_pd.read_csv(csv_path, nrows=1).columns.tolist())
    except Exception:
        _cols = set()

    cases: list[AblationCase] = []
    if args.landmarks_model:
        cases.append(AblationCase(name="landmarks", pipeline="landmarks", model=args.landmarks_model))
    if args.cnn_model:
        # Always include masked+raw if possible for ablation.
        if "img_masked_path" in _cols:
            cases.append(AblationCase(name="cnn_masked", pipeline="cnn", model=args.cnn_model, cnn_image_col="img_masked_path"))
        if "img_raw_path" in _cols:
            cases.append(AblationCase(name="cnn_raw", pipeline="cnn", model=args.cnn_model, cnn_image_col="img_raw_path"))
        if not any(c.pipeline == "cnn" for c in cases) and "img_path" in _cols:
            # Backward compatible with older CSV schema.
            cases.append(AblationCase(name="cnn_legacy", pipeline="cnn", model=args.cnn_model, cnn_image_col="img_path"))
    if not cases:
        raise SystemExit("Provide at least one model: --landmarks-model and/or --cnn-model")

    results = {"csv": str(csv_path), "cases": []}
    results["fps_config"] = {
        "preprocess": args.fps_preprocess,
        "tracker": args.fps_tracker,
        "skin_mask": args.fps_skin_mask,
        "mask_space": args.fps_mask_space,
        "seconds": float(args.seconds),
        "camera": int(args.camera),
    }

    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Evaluation macro-F1 per model (split)
    eval_json = out_dir / "eval_split.json"
    eval_payloads: dict[str, dict] = {}

    if args.landmarks_model:
        eval_cmd = [
            _python(),
            "scripts/eval_split.py",
            "--csv",
            str(csv_path),
            "--test-size",
            str(args.test_size),
            "--seed",
            str(args.seed),
            "--json-out",
            str(out_dir / "eval_landmarks.json"),
            "--landmarks-model",
            args.landmarks_model,
        ]
        if args.group_col:
            eval_cmd += ["--group-col", args.group_col]
        print("Running:", " ".join(eval_cmd))
        _run(eval_cmd)
        eval_payloads["landmarks"] = json.loads((out_dir / "eval_landmarks.json").read_text(encoding="utf-8"))["results"]["landmarks"]

    if args.cnn_model:
        cols = [c.cnn_image_col for c in cases if c.pipeline == "cnn"]
        for col in cols:
            eval_cmd = [
                _python(),
                "scripts/eval_split.py",
                "--csv",
                str(csv_path),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(args.seed),
                "--json-out",
                str(out_dir / f"eval_cnn_{col}.json"),
                "--cnn-model",
                args.cnn_model,
                "--cnn-image-col",
                col,
            ]
            if args.group_col:
                eval_cmd += ["--group-col", args.group_col]
            print("Running:", " ".join(eval_cmd))
            _run(eval_cmd)
            eval_payloads[f"cnn_{col}"] = json.loads((out_dir / f"eval_cnn_{col}.json").read_text(encoding="utf-8"))["results"]["cnn"]

    # 2) FPS profiling per model (webcam)
    fps_payloads: dict[str, dict] = {}
    for c in cases:
        fps_json = out_dir / f"fps_{c.name}.json"
        fps_cmd = [
            _python(),
            "scripts/profile_fps.py",
            "--pipeline",
            c.pipeline,
            "--model",
            c.model,
            "--seconds",
            str(args.seconds),
            "--camera",
            str(args.camera),
            "--json-out",
            str(fps_json),
        ]
        if args.fps_preprocess == "off":
            fps_cmd += ["--no-preprocess"]
        if args.fps_tracker == "off":
            fps_cmd += ["--no-tracker"]
        if args.fps_skin_mask == "on":
            fps_cmd += ["--skin-mask", "--mask-space", args.fps_mask_space]
        else:
            fps_cmd += ["--mask-space", args.fps_mask_space]
        print("Running:", " ".join(fps_cmd))
        _run(fps_cmd)
        fps_payloads[c.name] = json.loads(fps_json.read_text(encoding="utf-8"))

    rows = []
    for c in cases:
        if c.pipeline == "landmarks":
            macro_f1 = eval_payloads.get("landmarks", {}).get("macro_f1")
        else:
            macro_f1 = eval_payloads.get(f"cnn_{c.cnn_image_col}", {}).get("macro_f1")
        fps = fps_payloads[c.name]["fps"]
        rows.append({"name": c.name, "pipeline": c.pipeline, "macro_f1": macro_f1, "fps": fps})
        results["cases"].append({"name": c.name, "pipeline": c.pipeline, "macro_f1": macro_f1, "fps": fps})

    df = pd.DataFrame(rows).sort_values(["macro_f1", "fps"], ascending=False)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
