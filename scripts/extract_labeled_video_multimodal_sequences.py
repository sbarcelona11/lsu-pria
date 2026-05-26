from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.multimodal import HolisticDetector
from vc_pria.opencv_utils import apply_clahe, maybe_denoise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract multimodal temporal sequences from labeled videos.")
    p.add_argument("--videos-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    p.add_argument("--subject-default", default="video")
    p.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _iter_videos(root: Path, exts: list[str]):
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in wanted:
            yield path


def _infer_meta(path: Path, root: Path, layout: str, subject_default: str) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    label = path.parent.name
    subject_id = subject_default
    if layout == "subject_label" or (layout == "auto" and len(parts) >= 3):
        subject_id = path.parent.parent.name
        label = path.parent.name
    elif layout == "label" or (layout == "auto" and len(parts) >= 2):
        label = path.parent.name
    video_id = rel.with_suffix("").as_posix().replace("/", "__")
    return str(label), str(subject_id), str(video_id)


def main() -> None:
    args = parse_args()
    videos_root = Path(args.videos_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    seq_root = out_dir / "multimodal_sequences"
    seq_root.mkdir(parents=True, exist_ok=True)
    detector = HolisticDetector()

    videos = list(_iter_videos(videos_root, list(args.exts)))
    if args.limit:
        videos = videos[: int(args.limit)]
    if not videos:
        raise SystemExit(f"No videos found under {videos_root}")

    rows_meta: list[dict] = []
    wrote = 0
    for video_path in videos:
        label, subject_id, video_id = _infer_meta(video_path, videos_root, args.layout, args.subject_default)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, int(round(src_fps / max(1e-6, float(args.fps)))))
            frame_idx = 0
            left_buf: list[np.ndarray] = []
            right_buf: list[np.ndarray] = []
            pose_buf: list[np.ndarray] = []
            face_buf: list[np.ndarray] = []
            ts_buf: list[float] = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue
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
                frame_idx += 1
        finally:
            cap.release()

        if len(ts_buf) < int(args.min_frames):
            continue

        out_label = seq_root / label
        out_label.mkdir(parents=True, exist_ok=True)
        out_path = out_label / f"{video_id}.npz"
        np.savez_compressed(
            str(out_path),
            left_hand=np.stack(left_buf, axis=0).astype(np.float32),
            right_hand=np.stack(right_buf, axis=0).astype(np.float32),
            pose=np.stack(pose_buf, axis=0).astype(np.float32),
            face=np.stack(face_buf, axis=0).astype(np.float32),
            ts=np.array(ts_buf, dtype=np.float64),
            label=label,
            subject_id=subject_id,
            group_id=video_id,
            video_id=video_id,
            video_path=str(video_path),
        )
        rows_meta.append(
            {
                "path": str(out_path),
                "label": label,
                "subject_id": subject_id,
                "group_id": video_id,
                "video_id": video_id,
                "frames": len(ts_buf),
            }
        )
        wrote += 1

    index_path = out_dir / "multimodal_sequences_index.json"
    index_path.write_text(json.dumps(rows_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote sequences: {wrote}")
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
