from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from vc_pria.demo import DemoRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["landmarks", "cnn", "sequence"], default="landmarks")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-fps", type=float, default=0.0, help="0 = sin límite")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    runner = DemoRunner(pipeline=args.pipeline, model_path=model_path)

    last_frame_t = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_frame_t
        last_frame_t = now

        out_bgr = runner.process_frame(frame_bgr, dt=dt)
        cv2.imshow("VC-pria demo", out_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        runner.handle_key(key)

        if args.max_fps and args.max_fps > 0:
            elapsed = time.time() - now
            sleep_s = max(0.0, (1.0 / args.max_fps) - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
