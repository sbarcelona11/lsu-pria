from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.hand import HandDetector
from lsu_pria.opencv_utils import apply_clahe, maybe_denoise, skin_mask_ycrcb, SkinMaskConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract ROI images (and optional landmarks) from iLSU-T weakly labeled manifest.\n"
            "This script expects a manifest produced by ilsut_make_manifest.py."
        )
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True, help="Output dataset folder (will create images_raw/ and landmarks.csv)")
    p.add_argument("--fps", type=float, default=5.0, help="Sampling fps within each segment")
    p.add_argument("--max-per-seg", type=int, default=40, help="Max frames extracted per segment")
    p.add_argument("--min-conf", type=float, default=0.5, help="Min MediaPipe score to accept ROI")
    p.add_argument("--preprocess", action="store_true", help="Apply CLAHE+denoise before detection/cropping")
    p.add_argument("--skin-mask", action="store_true", help="Compute skin mask and apply to ROI (for images_masked/)")
    p.add_argument("--save-masked", action="store_true", help="Also write masked ROI images in images_masked/")
    p.add_argument("--save-landmarks", action="store_true", help="Write landmarks.csv compatible with scripts/train_landmarks.py")
    p.add_argument("--camera-like", action="store_true", help="Resize ROIs to 224x224 (same as realtime CNN)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of manifest rows")
    return p.parse_args()


@dataclass
class LmRow:
    label: str
    subject_id: str
    group_id: str
    img_raw_path: str
    img_masked_path: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    score: float
    landmarks: list[float]
    source: str
    episode_id: str
    video_path: str
    start_ms: int
    end_ms: int


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


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
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "images_raw"
    masked_dir = out_dir / "images_masked"
    _safe_mkdir(raw_dir)
    if args.save_masked or args.skin_mask:
        _safe_mkdir(masked_dir)

    detector = HandDetector(max_num_hands=1)
    skin_cfg = SkinMaskConfig()

    lm_rows: list[LmRow] = []
    sample_rows: list[dict] = []

    n_rows = 0
    n_saved = 0
    for row in _iter_manifest(manifest_path):
        n_rows += 1
        if args.limit and n_rows > int(args.limit):
            break

        label = str(row.get("label") or "").strip() or "unknown"
        episode_id = str(row.get("episode_id") or "EP").strip()
        group_id = str(row.get("group_id") or row.get("source") or episode_id or "EP").strip()
        subject_id = group_id
        source = str(row.get("source") or "").strip()
        video_path = Path(str(row.get("video_path") or "")).expanduser()
        try:
            start_ms = int(float(row.get("start_ms") or "0"))
            end_ms = int(float(row.get("end_ms") or "0"))
        except Exception:
            continue

        if not video_path.exists():
            continue
        if end_ms <= start_ms:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, int(round(fps / float(args.fps))))
            start_frame = int(round((start_ms / 1000.0) * fps))
            end_frame = int(round((end_ms / 1000.0) * fps))
            if end_frame <= start_frame:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            frames_taken = 0
            frame_idx = start_frame

            while frame_idx <= end_frame and frames_taken < int(args.max_per_seg):
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if frame_idx != start_frame and (frame_idx - start_frame) % step != 0:
                    frame_idx += 1
                    continue

                work = frame_bgr
                if args.preprocess:
                    work = apply_clahe(work)
                    work = maybe_denoise(work)

                mask = None
                if args.skin_mask:
                    mask = skin_mask_ycrcb(work, skin_cfg)

                frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
                hand = detector.detect(frame_rgb)
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

                # Save images
                stamp = f"{video_path.stem}_{start_ms}_{frame_idx}"
                out_raw = raw_dir / label / f"{stamp}.jpg"
                _safe_mkdir(out_raw.parent)
                cv2.imwrite(str(out_raw), roi)

                out_masked = ""
                if roi_masked is not None and (args.save_masked or args.skin_mask):
                    out_m = masked_dir / label / f"{stamp}.jpg"
                    _safe_mkdir(out_m.parent)
                    cv2.imwrite(str(out_m), roi_masked)
                    out_masked = str(out_m)

                if args.save_landmarks and hand.landmarks is not None:
                    lm_flat = hand.landmarks.reshape(-1).astype(float).tolist()
                    x, y, w, h = hand.bbox
                    lm_rows.append(
                        LmRow(
                            label=label,
                            subject_id=subject_id,
                            group_id=group_id,
                            img_raw_path=str(out_raw),
                            img_masked_path=out_masked,
                            bbox_x=int(x),
                            bbox_y=int(y),
                            bbox_w=int(w),
                            bbox_h=int(h),
                            score=float(hand.score or 0.0),
                            landmarks=lm_flat,
                            source=source,
                            episode_id=episode_id,
                            video_path=str(video_path),
                            start_ms=int(start_ms),
                            end_ms=int(end_ms),
                        )
                    )

                sample_rows.append(
                    {
                        "label": label,
                        "subject_id": subject_id,
                        "group_id": group_id,
                        "source": source,
                        "episode_id": episode_id,
                        "video_path": str(video_path),
                        "start_ms": int(start_ms),
                        "end_ms": int(end_ms),
                        "frame_idx": int(frame_idx),
                        "bbox_x": int(hand.bbox[0]),
                        "bbox_y": int(hand.bbox[1]),
                        "bbox_w": int(hand.bbox[2]),
                        "bbox_h": int(hand.bbox[3]),
                        "score": float(hand.score or 0.0),
                        "img_raw_path": str(out_raw),
                        "img_masked_path": out_masked,
                    }
                )

                frames_taken += 1
                n_saved += 1
                frame_idx += 1

        finally:
            cap.release()

        if n_rows % 20 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] rows={n_rows} saved_frames={n_saved}")

    if sample_rows:
        samples_path = out_dir / "samples.csv"
        sample_fieldnames = [
            "label",
            "subject_id",
            "group_id",
            "source",
            "episode_id",
            "video_path",
            "start_ms",
            "end_ms",
            "frame_idx",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
            "img_raw_path",
            "img_masked_path",
        ]
        with samples_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sample_fieldnames)
            w.writeheader()
            for r in sample_rows:
                w.writerow(r)
        print(f"Wrote: {samples_path} ({len(sample_rows)} rows)")

    if args.save_landmarks and lm_rows:
        lm_path = out_dir / "landmarks.csv"
        lm_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["label", "subject_id", "group_id", "source", "episode_id", "video_path", "start_ms", "end_ms"]
            + [f"lm_{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
            + ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "img_raw_path", "img_masked_path", "score", "landmarks"]
        )
        with lm_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in lm_rows:
                wide_landmarks = {f"lm_{i}_{c}": r.landmarks[(i * 3) + j] for i in range(21) for j, c in enumerate(("x", "y", "z"))}
                w.writerow(
                    {
                        "label": r.label,
                        "subject_id": r.subject_id,
                        "group_id": r.group_id,
                        "source": r.source,
                        "episode_id": r.episode_id,
                        "video_path": r.video_path,
                        "start_ms": r.start_ms,
                        "end_ms": r.end_ms,
                        **wide_landmarks,
                        "bbox_x": r.bbox_x,
                        "bbox_y": r.bbox_y,
                        "bbox_w": r.bbox_w,
                        "bbox_h": r.bbox_h,
                        "img_raw_path": r.img_raw_path,
                        "img_masked_path": r.img_masked_path,
                        "score": r.score,
                        "landmarks": json.dumps(r.landmarks),
                    }
                )
        print(f"Wrote: {lm_path} ({len(lm_rows)} rows)")

    print(f"Done. manifest_rows={n_rows} saved_frames={n_saved} out_dir={out_dir}")


if __name__ == "__main__":
    main()
