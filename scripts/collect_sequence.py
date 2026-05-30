from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.hand import HandDetector
from lsu_pria.opencv_utils import apply_clahe, maybe_denoise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect short landmark sequences for dynamic gestures.")
    p.add_argument("--out", required=True, help="Output dir (will create sequences/<label>/*.npz)")
    p.add_argument("--labels", nargs="+", required=True, help="List of labels. Key is first letter of each label.")
    p.add_argument("--subject-id", default="S1")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--duration", type=float, default=2.0, help="Seconds per recorded sample")
    p.add_argument("--target-fps", type=float, default=12.0)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--preprocess", action="store_true", default=True)
    p.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(max_num_hands=1)

    key_to_label: dict[str, str] = {}
    for lab in args.labels:
        k = lab[0].lower()
        if k in key_to_label:
            raise SystemExit(f"Duplicate key '{k}' for labels '{key_to_label[k]}' and '{lab}'")
        key_to_label[k] = lab

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    recording = False
    rec_label = ""
    rec_start = 0.0
    last_save = 0.0
    buf_lm: list[np.ndarray] = []
    buf_ts: list[float] = []
    buf_hand: list[str] = []

    def _reset_recording() -> None:
        nonlocal recording, rec_label, rec_start, buf_lm, buf_ts, buf_hand
        recording = False
        rec_label = ""
        rec_start = 0.0
        buf_lm = []
        buf_ts = []
        buf_hand = []

    def _save_sample() -> None:
        nonlocal last_save
        if not rec_label or len(buf_lm) < int(args.min_frames):
            return
        stamp = f"{time.time():.6f}"
        out = (seq_dir / rec_label)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{stamp}.npz"
        arr = np.stack(buf_lm, axis=0).astype(np.float32)  # (T,21,3)
        ts = np.array(buf_ts, dtype=np.float64)
        handedness = np.array(buf_hand, dtype=object)
        np.savez_compressed(
            str(path),
            landmarks=arr,
            ts=ts,
            handedness=handedness,
            label=rec_label,
            subject_id=str(args.subject_id),
        )
        last_save = time.time()
        print(f"Saved: {path} frames={arr.shape[0]}")

    target_dt = 1.0 / max(1e-6, float(args.target_fps))
    last_frame_t = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_frame_t
        last_frame_t = now

        work = frame_bgr.copy()
        if args.preprocess:
            work = apply_clahe(work)
            work = maybe_denoise(work)

        frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = detector.detect(frame_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key != 255:
            ch = chr(key).lower()
            if ch in key_to_label and not recording:
                recording = True
                rec_label = key_to_label[ch]
                rec_start = now
                buf_lm = []
                buf_ts = []
                buf_hand = []
                print(f"Recording: {rec_label} for {args.duration:.1f}s ...")
            if ch == "r":
                _reset_recording()

        if res is not None and res.bbox is not None:
            x, y, w, h = res.bbox
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detector.draw_landmarks(frame_bgr, res.landmarks)

        if recording and res is not None and res.landmarks is not None:
            # Basic rate limit
            if not buf_ts or (now - buf_ts[-1]) >= target_dt:
                buf_lm.append(res.landmarks.astype(np.float32))
                buf_ts.append(now)
                buf_hand.append(str(res.handedness or ""))

            if now - rec_start >= float(args.duration):
                _save_sample()
                _reset_recording()

        msg = f"Keys: start={list(key_to_label.keys())}  r=reset  q/esc=quit"
        if recording:
            msg += f"  REC {rec_label} t={now - rec_start:.1f}/{args.duration:.1f}s frames={len(buf_lm)}"
        cv2.putText(frame_bgr, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("collect_sequence", frame_bgr)

        # Avoid runaway CPU if camera is too fast.
        if dt < 0.002:
            time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

