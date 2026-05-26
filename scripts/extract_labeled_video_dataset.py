from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.hand import HandDetector
from vc_pria.opencv_utils import SkinMaskConfig, apply_clahe, maybe_denoise, skin_mask_ycrcb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frame-level training data from labeled videos.")
    p.add_argument("--videos-root", required=True, help="Dataset root, e.g. data/videos/<label>/*.mp4")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layout", choices=["auto", "label", "subject_label"], default="auto")
    p.add_argument("--subject-default", default="video")
    p.add_argument("--exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
    p.add_argument("--fps", type=float, default=4.0, help="Sampling fps per video")
    p.add_argument("--max-per-video", type=int, default=120)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--skin-mask", action="store_true")
    p.add_argument("--save-masked", action="store_true")
    p.add_argument("--save-landmarks", action="store_true", default=True)
    p.add_argument("--camera-like", action="store_true", help="Resize ROI to 224x224")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


@dataclass
class VideoMeta:
    label: str
    subject_id: str
    group_id: str
    video_id: str
    video_path: Path


@dataclass
class LmRow:
    label: str
    subject_id: str
    group_id: str
    video_id: str
    video_path: str
    frame_idx: int
    img_raw_path: str
    img_masked_path: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    score: float
    landmarks: list[float]


def _iter_videos(root: Path, exts: list[str]):
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in wanted:
            yield path


def _infer_meta(path: Path, root: Path, layout: str, subject_default: str) -> VideoMeta:
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
    return VideoMeta(
        label=str(label),
        subject_id=str(subject_id),
        group_id=str(video_id),
        video_id=str(video_id),
        video_path=path,
    )


def _crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    if w <= 2 or h <= 2:
        return None
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 <= 2 or y1 - y0 <= 2:
        return None
    return frame_bgr[y0:y1, x0:x1]


def main() -> None:
    args = parse_args()
    videos_root = Path(args.videos_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "images_raw"
    masked_dir = out_dir / "images_masked"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masked or args.skin_mask:
        masked_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(max_num_hands=1)
    skin_cfg = SkinMaskConfig()
    lm_rows: list[LmRow] = []
    sample_rows: list[dict] = []

    videos = list(_iter_videos(videos_root, list(args.exts)))
    if args.limit:
        videos = videos[: int(args.limit)]
    if not videos:
        raise SystemExit(f"No videos found under {videos_root}")

    saved_frames = 0
    for vid_idx, video_path in enumerate(videos, start=1):
        meta = _infer_meta(video_path, videos_root, args.layout, args.subject_default)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, int(round(src_fps / max(1e-6, float(args.fps)))))
            frame_idx = 0
            taken = 0
            while taken < int(args.max_per_video):
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

                mask = None
                if args.skin_mask:
                    mask = skin_mask_ycrcb(work, skin_cfg)

                hand = detector.detect(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
                if hand is None or hand.bbox is None or (hand.score or 0.0) < float(args.min_conf):
                    frame_idx += 1
                    continue

                roi = _crop(work, hand.bbox)
                if roi is None or not roi.size:
                    frame_idx += 1
                    continue

                roi_masked = None
                if mask is not None:
                    x, y, w, h = hand.bbox
                    roi_m = mask[y : y + h, x : x + w]
                    roi_m = cv2.resize(roi_m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=roi_m)

                if args.camera_like:
                    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                    if roi_masked is not None:
                        roi_masked = cv2.resize(roi_masked, (224, 224), interpolation=cv2.INTER_AREA)

                stamp = f"{meta.video_id}_{frame_idx}"
                raw_path = raw_dir / meta.label / f"{stamp}.jpg"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(raw_path), roi)

                masked_path = ""
                if roi_masked is not None and (args.save_masked or args.skin_mask):
                    out_m = masked_dir / meta.label / f"{stamp}.jpg"
                    out_m.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_m), roi_masked)
                    masked_path = str(out_m)

                sample_rows.append(
                    {
                        "label": meta.label,
                        "subject_id": meta.subject_id,
                        "group_id": meta.group_id,
                        "video_id": meta.video_id,
                        "video_path": str(meta.video_path),
                        "frame_idx": int(frame_idx),
                        "bbox_x": int(hand.bbox[0]),
                        "bbox_y": int(hand.bbox[1]),
                        "bbox_w": int(hand.bbox[2]),
                        "bbox_h": int(hand.bbox[3]),
                        "score": float(hand.score or 0.0),
                        "img_raw_path": str(raw_path),
                        "img_masked_path": masked_path,
                    }
                )

                if args.save_landmarks and hand.landmarks is not None:
                    lm_rows.append(
                        LmRow(
                            label=meta.label,
                            subject_id=meta.subject_id,
                            group_id=meta.group_id,
                            video_id=meta.video_id,
                            video_path=str(meta.video_path),
                            frame_idx=int(frame_idx),
                            img_raw_path=str(raw_path),
                            img_masked_path=masked_path,
                            bbox_x=int(hand.bbox[0]),
                            bbox_y=int(hand.bbox[1]),
                            bbox_w=int(hand.bbox[2]),
                            bbox_h=int(hand.bbox[3]),
                            score=float(hand.score or 0.0),
                            landmarks=hand.landmarks.reshape(-1).astype(float).tolist(),
                        )
                    )

                taken += 1
                saved_frames += 1
                frame_idx += 1
        finally:
            cap.release()

        if vid_idx % 10 == 0:
            print(f"videos={vid_idx}/{len(videos)} saved_frames={saved_frames}")

    samples_path = out_dir / "samples.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "subject_id",
            "group_id",
            "video_id",
            "video_path",
            "frame_idx",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
            "img_raw_path",
            "img_masked_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sample_rows:
            w.writerow(row)
    print(f"Wrote: {samples_path} ({len(sample_rows)} rows)")

    if args.save_landmarks and lm_rows:
        lm_path = out_dir / "landmarks.csv"
        fieldnames = (
            ["label", "subject_id", "group_id", "video_id", "video_path", "frame_idx"]
            + [f"lm_{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
            + ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "img_raw_path", "img_masked_path", "score", "landmarks"]
        )
        with lm_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in lm_rows:
                wide = {f"lm_{i}_{c}": row.landmarks[(i * 3) + j] for i in range(21) for j, c in enumerate(("x", "y", "z"))}
                w.writerow(
                    {
                        "label": row.label,
                        "subject_id": row.subject_id,
                        "group_id": row.group_id,
                        "video_id": row.video_id,
                        "video_path": row.video_path,
                        "frame_idx": row.frame_idx,
                        **wide,
                        "bbox_x": row.bbox_x,
                        "bbox_y": row.bbox_y,
                        "bbox_w": row.bbox_w,
                        "bbox_h": row.bbox_h,
                        "img_raw_path": row.img_raw_path,
                        "img_masked_path": row.img_masked_path,
                        "score": row.score,
                        "landmarks": json.dumps(row.landmarks),
                    }
                )
        print(f"Wrote: {lm_path} ({len(lm_rows)} rows)")

    print(f"Done. videos={len(videos)} saved_frames={saved_frames} out_dir={out_dir}")


if __name__ == "__main__":
    main()
