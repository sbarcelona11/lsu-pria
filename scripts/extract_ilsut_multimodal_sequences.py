from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.multimodal import HolisticDetector
from vc_pria.opencv_utils import apply_clahe, maybe_denoise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract multimodal temporal sequences from iLSU-T manifest segments.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--max-per-seg", type=int, default=40)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _read_segment(cap: cv2.VideoCapture, start_ms: int, end_ms: int, target_fps: float, max_per_seg: int):
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(1e-6, target_fps))))
    start_frame = max(0, int(round((start_ms / 1000.0) * src_fps)))
    end_frame = max(start_frame, int(round((end_ms / 1000.0) * src_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    saved = 0
    while frame_idx <= end_frame and saved < max_per_seg:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame_idx - start_frame) % step == 0:
            yield frame, frame_idx
            saved += 1
        frame_idx += 1


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    seq_dir = out_dir / "multimodal_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    detector = HolisticDetector()

    wrote = 0
    rows_meta: list[dict] = []
    for idx, row in df.iterrows():
        if args.limit and idx >= int(args.limit):
            break
        video_path = Path(str(row["video_path"]))
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        try:
            left_buf: list[np.ndarray] = []
            right_buf: list[np.ndarray] = []
            pose_buf: list[np.ndarray] = []
            face_buf: list[np.ndarray] = []
            ts_buf: list[float] = []
            for frame_bgr, frame_idx in _read_segment(
                cap,
                int(row["start_ms"]),
                int(row["end_ms"]),
                float(args.fps),
                int(args.max_per_seg),
            ):
                work = frame_bgr
                if args.preprocess:
                    work = apply_clahe(work)
                    work = maybe_denoise(work)
                res = detector.detect(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
                left_buf.append(res.left_hand if res.left_hand is not None else np.zeros((21, 3), dtype=np.float32))
                right_buf.append(res.right_hand if res.right_hand is not None else np.zeros((21, 3), dtype=np.float32))
                pose_buf.append(res.pose if res.pose is not None else np.zeros((11, 3), dtype=np.float32))
                face_buf.append(res.face if res.face is not None else np.zeros((6, 3), dtype=np.float32))
                ts_buf.append(float(frame_idx))

            if not ts_buf:
                continue
            label = str(row["label"])
            group_id = str(row.get("group_id", row.get("episode_id", f"row_{idx}")))
            sample_id = f"{time.time():.6f}_{idx}"
            label_dir = seq_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            out_path = label_dir / f"{sample_id}.npz"
            np.savez_compressed(
                str(out_path),
                left_hand=np.stack(left_buf, axis=0).astype(np.float32),
                right_hand=np.stack(right_buf, axis=0).astype(np.float32),
                pose=np.stack(pose_buf, axis=0).astype(np.float32),
                face=np.stack(face_buf, axis=0).astype(np.float32),
                ts=np.array(ts_buf, dtype=np.float64),
                label=label,
                subject_id=str(row.get("source", "iLSUT")),
                group_id=group_id,
                episode_id=str(row.get("episode_id", "")),
                video_path=str(video_path),
            )
            rows_meta.append(
                {
                    "path": str(out_path),
                    "label": label,
                    "group_id": group_id,
                    "subject_id": str(row.get("source", "iLSUT")),
                    "episode_id": str(row.get("episode_id", "")),
                    "frames": len(ts_buf),
                }
            )
            wrote += 1
        finally:
            cap.release()

    meta_path = out_dir / "multimodal_sequences_index.json"
    meta_path.write_text(json.dumps(rows_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote sequences: {wrote}")
    print(f"Wrote index: {meta_path}")


if __name__ == "__main__":
    main()
