from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .features import extract_multimodal_frame_features
from .multimodal import HolisticDetector, multimodal_bbox, multimodal_primary_hand_landmarks
from .opencv_utils import apply_clahe, maybe_denoise


@dataclass
class SltFeatureExtraction:
    features: np.ndarray
    ts_ms: np.ndarray
    bboxes: list[tuple[int, int, int, int] | None]
    landmarks_px: list[list[list[int]] | None]
    frames_total: int
    frames_used: int


def aggregate_sequence_embedding(seq_feats: np.ndarray, min_frames: int = 4) -> np.ndarray:
    arr = np.asarray(seq_feats, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if arr.shape[0] < int(min_frames):
        pad = np.repeat(arr[:1], int(min_frames) - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    delta = arr[-1] - arr[0]
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)


def extract_slt_features_from_capture(
    cap: cv2.VideoCapture,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    sample_fps: float = 6.0,
    max_frames: int = 0,
    preprocess: bool = False,
    include_debug: bool = False,
) -> SltFeatureExtraction:
    detector = HolisticDetector()
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    step = 1 if sample_fps <= 0 else max(1, int(round(src_fps / max(1e-6, float(sample_fps)))))

    if start_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, int(start_ms)))

    features: list[np.ndarray] = []
    ts_ms: list[int] = []
    bboxes: list[tuple[int, int, int, int] | None] = []
    landmarks_px: list[list[list[int]] | None] = []
    frames_total = 0
    frames_used = 0
    start_frame_idx: Optional[int] = None

    while True:
        pos_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        if end_ms is not None and pos_ms > int(end_ms):
            break
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frames_total += 1
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or frames_total)
        if start_frame_idx is None:
            start_frame_idx = frame_idx
        if step > 1 and ((frame_idx - start_frame_idx) % step != 0):
            continue
        if max_frames and frames_used >= int(max_frames):
            break

        work = frame_bgr
        if preprocess:
            work = apply_clahe(work)
            work = maybe_denoise(work)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = detector.detect(rgb)
        feat = extract_multimodal_frame_features(res.left_hand, res.right_hand, res.pose, res.face)
        features.append(feat)
        ts_ms.append(pos_ms)
        frames_used += 1

        if include_debug:
            bbox = multimodal_bbox(res, work.shape[1], work.shape[0]) if res is not None else None
            bboxes.append(bbox)
            primary = multimodal_primary_hand_landmarks(res) if res is not None else None
            if primary is not None:
                pts = np.stack(
                    [
                        np.clip(primary[:, 0] * float(work.shape[1]), 0, max(0, work.shape[1] - 1)),
                        np.clip(primary[:, 1] * float(work.shape[0]), 0, max(0, work.shape[0] - 1)),
                    ],
                    axis=1,
                )
                landmarks_px.append(pts.astype(int).tolist())
            else:
                landmarks_px.append(None)

    if not include_debug:
        bboxes = [None] * len(features)
        landmarks_px = [None] * len(features)

    if not features:
        return SltFeatureExtraction(
            features=np.zeros((0, 0), dtype=np.float32),
            ts_ms=np.zeros((0,), dtype=np.int64),
            bboxes=[],
            landmarks_px=[],
            frames_total=frames_total,
            frames_used=0,
        )
    return SltFeatureExtraction(
        features=np.stack(features, axis=0).astype(np.float32),
        ts_ms=np.asarray(ts_ms, dtype=np.int64),
        bboxes=bboxes,
        landmarks_px=landmarks_px,
        frames_total=frames_total,
        frames_used=frames_used,
    )


def extract_slt_features_from_video(
    video_path: Path,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    sample_fps: float = 6.0,
    max_frames: int = 0,
    preprocess: bool = False,
    include_debug: bool = False,
) -> SltFeatureExtraction:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        return extract_slt_features_from_capture(
            cap,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_fps=sample_fps,
            max_frames=max_frames,
            preprocess=preprocess,
            include_debug=include_debug,
        )
    finally:
        cap.release()


def extract_slt_features_from_frames(
    frames_bgr: list[np.ndarray],
    ts_ms: list[int],
    *,
    sample_fps: float = 6.0,
    max_frames: int = 0,
    preprocess: bool = False,
    include_debug: bool = False,
) -> SltFeatureExtraction:
    """
    Extract SLT features from an in-memory sequence of frames with timestamps.
    This is used by the realtime-ish SLT web endpoint (rolling window).
    """
    if not frames_bgr or not ts_ms or len(frames_bgr) != len(ts_ms):
        return SltFeatureExtraction(
            features=np.zeros((0, 0), dtype=np.float32),
            ts_ms=np.zeros((0,), dtype=np.int64),
            bboxes=[],
            landmarks_px=[],
            frames_total=len(frames_bgr),
            frames_used=0,
        )
    detector = HolisticDetector()
    step_ms = 0 if sample_fps <= 0 else int(round(1000.0 / max(1e-6, float(sample_fps))))

    features: list[np.ndarray] = []
    out_ts: list[int] = []
    bboxes: list[tuple[int, int, int, int] | None] = []
    landmarks_px: list[list[list[int]] | None] = []
    frames_used = 0
    last_kept_ms: int | None = None

    for frame_bgr, t_ms in zip(frames_bgr, ts_ms):
        if max_frames and frames_used >= int(max_frames):
            break
        if step_ms > 0 and last_kept_ms is not None and int(t_ms) - int(last_kept_ms) < step_ms:
            continue

        work = frame_bgr
        if preprocess:
            work = apply_clahe(work)
            work = maybe_denoise(work)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = detector.detect(rgb)
        feat = extract_multimodal_frame_features(res.left_hand, res.right_hand, res.pose, res.face)
        features.append(feat)
        out_ts.append(int(t_ms))
        frames_used += 1
        last_kept_ms = int(t_ms)

        if include_debug:
            bbox = multimodal_bbox(res, work.shape[1], work.shape[0]) if res is not None else None
            bboxes.append(bbox)
            primary = multimodal_primary_hand_landmarks(res) if res is not None else None
            if primary is not None:
                pts = np.stack(
                    [
                        np.clip(primary[:, 0] * float(work.shape[1]), 0, max(0, work.shape[1] - 1)),
                        np.clip(primary[:, 1] * float(work.shape[0]), 0, max(0, work.shape[0] - 1)),
                    ],
                    axis=1,
                )
                landmarks_px.append(pts.astype(int).tolist())
            else:
                landmarks_px.append(None)

    if not include_debug:
        bboxes = [None] * len(features)
        landmarks_px = [None] * len(features)

    if not features:
        return SltFeatureExtraction(
            features=np.zeros((0, 0), dtype=np.float32),
            ts_ms=np.zeros((0,), dtype=np.int64),
            bboxes=[],
            landmarks_px=[],
            frames_total=len(frames_bgr),
            frames_used=0,
        )
    return SltFeatureExtraction(
        features=np.stack(features, axis=0).astype(np.float32),
        ts_ms=np.asarray(out_ts, dtype=np.int64),
        bboxes=bboxes,
        landmarks_px=landmarks_px,
        frames_total=len(frames_bgr),
        frames_used=frames_used,
    )


def normalize_slt_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def token_overlap_score(reference: str, hypothesis: str) -> float:
    ref = normalize_slt_text(reference).split()
    hyp = normalize_slt_text(hypothesis).split()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    inter = 0
    hyp_counts: dict[str, int] = {}
    for tok in hyp:
        hyp_counts[tok] = hyp_counts.get(tok, 0) + 1
    for tok in ref:
        if hyp_counts.get(tok, 0) > 0:
            inter += 1
            hyp_counts[tok] -= 1
    precision = inter / len(hyp)
    recall = inter / len(ref)
    if precision + recall <= 1e-9:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def bleu_like(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref = normalize_slt_text(reference).split()
    hyp = normalize_slt_text(hypothesis).split()
    if not ref or not hyp:
        return 0.0
    if ref == hyp:
        return 1.0

    def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]

    max_order = max(1, min(int(max_n), len(ref), len(hyp)))
    precisions: list[float] = []
    for n in range(1, max_order + 1):
        ref_ngrams = ngrams(ref, n)
        hyp_ngrams = ngrams(hyp, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        ref_counts: dict[tuple[str, ...], int] = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        match = 0
        for ng in hyp_ngrams:
            if ref_counts.get(ng, 0) > 0:
                match += 1
                ref_counts[ng] -= 1
        precisions.append(match / len(hyp_ngrams))
    if any(p <= 0 for p in precisions):
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precisions) / max_order)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - (len(ref) / max(1, len(hyp))))
    return float(bp * geo)
