from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from vc_pria.hand import HandDetector, HandResult
from vc_pria.multimodal import HolisticDetector, multimodal_bbox
from vc_pria.opencv_utils import SkinMaskConfig, apply_clahe, maybe_denoise, skin_mask_hsv, skin_mask_ycrcb
from vc_pria.pipelines.cnn import CnnPipeline
from vc_pria.pipelines.landmarks import LandmarksPipeline
from vc_pria.pipelines.multimodal_sequence import MultimodalSequencePipeline
from vc_pria.pipelines.sequence import SequencePipeline
from vc_pria.tracking import RoiTracker
from vc_pria.webapp.composer import ComposeConfig, ComposeMode, ComposeState


@dataclass
class VideoCase:
    video_path: str
    title: str = ""
    source_url: str = ""
    expected_text: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the recognition pipeline over local validation videos.")
    p.add_argument("--pipeline", choices=["landmarks", "cnn", "sequence", "multimodal"], default="landmarks")
    p.add_argument("--landmarks-model", default="")
    p.add_argument("--cnn-model", default="")
    p.add_argument("--sequence-model", default="")
    p.add_argument("--multimodal-model", default="")
    p.add_argument("--videos", nargs="*", default=None, help="Local video files")
    p.add_argument("--cases-json", default="", help="JSON file with [{video_path,title,source_url,expected_text}]")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["both", "words", "spelling"], default="both")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--skin-mask", action="store_true")
    p.add_argument("--mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    p.add_argument("--use-tracker", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.75)
    p.add_argument("--stable-frames-min", type=int, default=6)
    p.add_argument("--pause-ms-min", type=int, default=350)
    p.add_argument("--cooldown-ms", type=int, default=800)
    p.add_argument("--sample-fps", type=float, default=0.0, help="0 = use all frames, otherwise sample to this FPS")
    return p.parse_args()


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.lower().strip().split())
    return s


def _load_cases(args: argparse.Namespace) -> list[VideoCase]:
    cases: list[VideoCase] = []
    if args.cases_json:
        raw = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit("cases-json must be a JSON array")
        for item in raw:
            if not isinstance(item, dict):
                continue
            cases.append(
                VideoCase(
                    video_path=str(item.get("video_path") or ""),
                    title=str(item.get("title") or ""),
                    source_url=str(item.get("source_url") or ""),
                    expected_text=str(item.get("expected_text") or ""),
                )
            )
    if args.videos:
        for p in args.videos:
            cases.append(VideoCase(video_path=str(p)))
    cases = [c for c in cases if c.video_path]
    if not cases:
        raise SystemExit("Provide --videos and/or --cases-json")
    return cases


def _load_pipeline(args: argparse.Namespace):
    if args.pipeline == "landmarks":
        if not args.landmarks_model:
            raise SystemExit("Missing --landmarks-model")
        return LandmarksPipeline.load(Path(args.landmarks_model))
    if args.pipeline == "cnn":
        if not args.cnn_model:
            raise SystemExit("Missing --cnn-model")
        return CnnPipeline.load(Path(args.cnn_model))
    if args.pipeline == "multimodal":
        if not args.multimodal_model:
            raise SystemExit("Missing --multimodal-model")
        return MultimodalSequencePipeline.load(Path(args.multimodal_model))
    if not args.sequence_model:
        raise SystemExit("Missing --sequence-model")
    return SequencePipeline.load(Path(args.sequence_model))


def _process_video(case: VideoCase, args: argparse.Namespace, pipeline) -> dict:
    path = Path(case.video_path)
    if not path.exists():
        raise SystemExit(f"Missing video: {path}")
    if hasattr(pipeline, "reset"):
        try:
            pipeline.reset()
        except Exception:
            pass
    if hasattr(pipeline, "_ema"):
        try:
            pipeline._ema = None
        except Exception:
            pass

    detector = HandDetector(max_num_hands=1)
    holistic = HolisticDetector()
    tracker = RoiTracker()
    skin_cfg = SkinMaskConfig()
    composer = ComposeState(
        config=ComposeConfig(
            confidence_threshold=args.confidence_threshold,
            stable_frames_min=args.stable_frames_min,
            pause_ms_min=args.pause_ms_min,
            cooldown_ms=args.cooldown_ms,
        ),
        mode=ComposeMode(name=args.mode),
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1
        if args.sample_fps and args.sample_fps > 0:
            step = max(1, int(round(src_fps / float(args.sample_fps))))

        frames = 0
        used_frames = 0
        predictions = 0
        conf_sum = 0.0
        tokens: list[str] = []
        last_tracker_status = "off"

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames += 1
            if step > 1 and (frames - 1) % step != 0:
                continue

            ts_ms = int(round(((frames - 1) / float(src_fps)) * 1000.0))
            work = frame_bgr
            if args.preprocess:
                work = apply_clahe(work)
                work = maybe_denoise(work)

            mask = None
            if args.skin_mask:
                if args.mask_space == "hsv":
                    mask = skin_mask_hsv(work, skin_cfg)
                else:
                    mask = skin_mask_ycrcb(work, skin_cfg)

            frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            hand = None
            holistic_res = None
            if args.pipeline == "multimodal":
                holistic_res = holistic.detect(frame_rgb)
                bbox = multimodal_bbox(holistic_res, work.shape[1], work.shape[0]) if holistic_res is not None else None
            else:
                hand = detector.detect(frame_rgb)
                bbox = hand.bbox if hand is not None else None

            if args.pipeline == "multimodal" and bbox is not None and args.use_tracker:
                tracker.update_from_detection(work, bbox)
                last_tracker_status = tracker.status
            elif args.pipeline == "multimodal" and args.use_tracker:
                tracked = tracker.track(work)
                last_tracker_status = tracker.status
                if tracked is not None:
                    bbox = tracked
            elif hand is not None and bbox is not None and args.use_tracker:
                tracker.update_from_detection(work, bbox)
                last_tracker_status = tracker.status
            elif hand is None and args.use_tracker:
                tracked = tracker.track(work)
                last_tracker_status = tracker.status
                if tracked is not None:
                    hand = HandResult(landmarks=None, handedness=None, bbox=tracked, score=0.0)
            else:
                last_tracker_status = "off"

            no_hand = hand is None and not (holistic_res is not None and holistic_res.any_hand())
            label = "no_hand"
            conf = 0.0
            if args.pipeline == "multimodal":
                label, conf = pipeline.predict_multimodal(
                    holistic_res.left_hand if holistic_res is not None else None,
                    holistic_res.right_hand if holistic_res is not None else None,
                    holistic_res.pose if holistic_res is not None else None,
                    holistic_res.face if holistic_res is not None else None,
                )
                no_hand = label == "no_hand"
            elif hand is not None:
                label, conf = pipeline.predict(work, hand, skin_mask=mask)
                no_hand = label == "no_hand" or (hand.landmarks is None and args.pipeline == "landmarks")

            new_token = composer.update(label=label, confidence=conf, no_hand=no_hand, ts_ms=ts_ms)
            if new_token:
                tokens.append(new_token)
            if label != "no_hand":
                predictions += 1
                conf_sum += float(conf)
            used_frames += 1

        predicted_text = composer.text
        expected_norm = _normalize_text(case.expected_text)
        predicted_norm = _normalize_text(predicted_text)
        exact_match = bool(expected_norm) and expected_norm == predicted_norm
        token_match = bool(expected_norm) and all(tok in predicted_norm for tok in expected_norm.split())

        return {
            "video_path": str(path),
            "title": case.title or path.stem,
            "source_url": case.source_url,
            "expected_text": case.expected_text,
            "predicted_text": predicted_text,
            "predicted_tokens": tokens,
            "frames_total": frames,
            "frames_used": used_frames,
            "predictions_count": predictions,
            "avg_confidence": (conf_sum / predictions) if predictions else 0.0,
            "tracker_status_last": last_tracker_status,
            "exact_match": exact_match,
            "token_match": token_match,
        }
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args)
    pipeline = _load_pipeline(args)
    results = [_process_video(case, args, pipeline) for case in cases]

    summary_json = out_dir / "video_validation.json"
    summary_csv = out_dir / "video_validation.csv"
    summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "title",
        "video_path",
        "source_url",
        "expected_text",
        "predicted_text",
        "frames_total",
        "frames_used",
        "predictions_count",
        "avg_confidence",
        "tracker_status_last",
        "exact_match",
        "token_match",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
