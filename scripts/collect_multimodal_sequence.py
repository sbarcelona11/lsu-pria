from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.multimodal import HolisticDetector
from vc_pria.opencv_utils import apply_clahe, maybe_denoise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect multimodal sequences (hands + pose + face) for temporal training.")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--subject-id", default="S1")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--duration", type=float, default=2.5)
    p.add_argument("--target-fps", type=float, default=12.0)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--preprocess", action="store_true", default=True)
    p.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    seq_dir = out_dir / "multimodal_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    detector = HolisticDetector()
    key_to_label: dict[str, str] = {}
    for lab in args.labels:
        key = lab[0].lower()
        if key in key_to_label:
            raise SystemExit(f"Duplicate key '{key}' for labels '{key_to_label[key]}' and '{lab}'")
        key_to_label[key] = lab

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    recording = False
    rec_label = ""
    rec_start = 0.0
    left_buf: list[np.ndarray] = []
    right_buf: list[np.ndarray] = []
    pose_buf: list[np.ndarray] = []
    face_buf: list[np.ndarray] = []
    ts_buf: list[float] = []
    target_dt = 1.0 / max(1e-6, float(args.target_fps))

    def _reset() -> None:
        nonlocal recording, rec_label, rec_start, left_buf, right_buf, pose_buf, face_buf, ts_buf
        recording = False
        rec_label = ""
        rec_start = 0.0
        left_buf, right_buf, pose_buf, face_buf, ts_buf = [], [], [], [], []

    def _stack_optional(buf: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        if not buf:
            return np.zeros((0, *shape), dtype=np.float32)
        return np.stack(buf, axis=0).astype(np.float32)

    def _save() -> None:
        if not rec_label or len(ts_buf) < int(args.min_frames):
            return
        stamp = f"{time.time():.6f}"
        out = seq_dir / rec_label
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{stamp}.npz"
        np.savez_compressed(
            str(path),
            left_hand=_stack_optional(left_buf, (21, 3)),
            right_hand=_stack_optional(right_buf, (21, 3)),
            pose=_stack_optional(pose_buf, (11, 3)),
            face=_stack_optional(face_buf, (6, 3)),
            ts=np.array(ts_buf, dtype=np.float64),
            label=rec_label,
            subject_id=str(args.subject_id),
        )
        print(f"Saved: {path} frames={len(ts_buf)}")

    last_sample_t = 0.0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        now = time.time()
        work = frame_bgr.copy()
        if args.preprocess:
            work = apply_clahe(work)
            work = maybe_denoise(work)
        res = detector.detect(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key != 255:
            ch = chr(key).lower()
            if ch in key_to_label and not recording:
                recording = True
                rec_label = key_to_label[ch]
                rec_start = now
                left_buf, right_buf, pose_buf, face_buf, ts_buf = [], [], [], [], []
                print(f"Recording: {rec_label} for {args.duration:.1f}s ...")
            if ch == "r":
                _reset()

        if recording and (not ts_buf or (now - last_sample_t) >= target_dt):
            left_buf.append(res.left_hand if res.left_hand is not None else np.zeros((21, 3), dtype=np.float32))
            right_buf.append(res.right_hand if res.right_hand is not None else np.zeros((21, 3), dtype=np.float32))
            pose_buf.append(res.pose if res.pose is not None else np.zeros((11, 3), dtype=np.float32))
            face_buf.append(res.face if res.face is not None else np.zeros((6, 3), dtype=np.float32))
            ts_buf.append(now)
            last_sample_t = now

            if now - rec_start >= float(args.duration):
                _save()
                _reset()

        msg = f"Keys: start={list(key_to_label.keys())} r=reset q/esc=quit"
        if recording:
            msg += f" REC {rec_label} t={now - rec_start:.1f}/{args.duration:.1f}s frames={len(ts_buf)}"
        cv2.putText(frame_bgr, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("collect_multimodal_sequence", frame_bgr)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
