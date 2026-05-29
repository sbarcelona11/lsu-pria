from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.hand import HandDetector
from lsu_pria.opencv_utils import SkinMaskConfig, apply_clahe, maybe_denoise, skin_mask_ycrcb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--subject-id", type=str, default="S1", help="Identifier for group split (e.g. S1, S2)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--preprocess", action="store_true", default=True)
    p.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    p.add_argument("--save-every", type=int, default=2, help="save 1 sample every N frames while label held")
    p.add_argument("--save-raw-roi", action="store_true", default=True)
    p.add_argument("--no-save-raw-roi", dest="save_raw_roi", action="store_false")
    p.add_argument("--save-masked-roi", action="store_true", default=True)
    p.add_argument("--no-save-masked-roi", dest="save_masked_roi", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_raw_dir = out_dir / "images_raw"
    img_masked_dir = out_dir / "images_masked"
    if args.save_raw_roi:
        img_raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masked_roi:
        img_masked_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "landmarks.csv"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = HandDetector(max_num_hands=1)
    skin_cfg = SkinMaskConfig()

    key_to_label = {}
    for lab in args.labels:
        k = lab[0].lower()
        if k in key_to_label:
            raise SystemExit(f"Duplicate key '{k}' for labels '{key_to_label[k]}' and '{lab}'")
        key_to_label[k] = lab

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if csv_path.stat().st_size == 0:
            writer.writerow(
                ["ts", "label", "handedness", "subject_id"]
                + [f"lm_{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
                + ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "img_raw_path", "img_masked_path"]
            )

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            work = frame_bgr.copy()
            if args.preprocess:
                work = apply_clahe(work)
                work = maybe_denoise(work)

            frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = detector.detect(frame_rgb)

            label = None
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key != 255:
                ch = chr(key).lower()
                label = key_to_label.get(ch)

            if res is not None and res.bbox is not None:
                x, y, w, h = res.bbox
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detector.draw_landmarks(frame_bgr, res.landmarks)

            if label and res is not None and res.landmarks is not None and (frame_idx % args.save_every == 0):
                ts = time.time()
                raw_path = ""
                masked_path = ""

                roi = None
                if res.bbox is not None:
                    x, y, w, h = res.bbox
                    roi = work[y : y + h, x : x + w]
                if roi is None or roi.size == 0:
                    roi = work

                if args.save_raw_roi:
                    lab_dir = img_raw_dir / label
                    lab_dir.mkdir(parents=True, exist_ok=True)
                    pth = lab_dir / f"{ts:.6f}.jpg"
                    cv2.imwrite(str(pth), roi)
                    raw_path = str(pth)

                if args.save_masked_roi:
                    lab_dir = img_masked_dir / label
                    lab_dir.mkdir(parents=True, exist_ok=True)
                    pth = lab_dir / f"{ts:.6f}.jpg"
                    mask = skin_mask_ycrcb(roi, skin_cfg)
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
                    cv2.imwrite(str(pth), roi_masked)
                    masked_path = str(pth)

                lm_flat = res.landmarks.reshape(-1).tolist()
                bx = by = bw = bh = ""
                if res.bbox is not None:
                    bx, by, bw, bh = res.bbox
                writer.writerow([ts, label, res.handedness, args.subject_id] + lm_flat + [bx, by, bw, bh, raw_path, masked_path])
                f.flush()

            cv2.putText(frame_bgr, f"Hold key to label: {list(key_to_label.keys())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("collect_data", frame_bgr)
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
