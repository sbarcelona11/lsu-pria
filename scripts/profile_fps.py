from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.demo import DemoRunner, DemoToggles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["landmarks", "cnn"], default="landmarks")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--preprocess", action="store_true", default=True)
    p.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    p.add_argument("--tracker", action="store_true", default=True)
    p.add_argument("--no-tracker", dest="tracker", action="store_false")
    p.add_argument("--skin-mask", action="store_true", default=False)
    p.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    p.add_argument("--json-out", type=str, default="", help="Write a JSON summary to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    toggles = DemoToggles(
        preprocess=bool(args.preprocess),
        show_skin_mask=bool(args.skin_mask),
        use_tracker=bool(args.tracker),
        mask_space=str(args.mask_space),
    )
    runner = DemoRunner(pipeline=args.pipeline, model_path=model_path, toggles=toggles)

    t_end = time.time() + args.seconds
    n = 0
    last_t = time.time()
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        dt = now - last_t
        last_t = now
        _ = runner.process_frame(frame, dt=dt)
        n += 1

    cap.release()
    elapsed = args.seconds
    fps = n / max(1e-6, elapsed)
    payload = {
        "pipeline": args.pipeline,
        "model": str(model_path),
        "camera": int(args.camera),
        "seconds": float(elapsed),
        "frames": int(n),
        "fps": float(fps),
        "width": int(args.width),
        "height": int(args.height),
        "preprocess": bool(args.preprocess),
        "tracker": bool(args.tracker),
        "skin_mask": bool(args.skin_mask),
        "mask_space": str(args.mask_space),
    }
    print(json.dumps(payload, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
