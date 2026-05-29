from __future__ import annotations

import copy
import json
import shutil
import subprocess
import time
import unicodedata
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from ..hand import HandDetector, HandResult
from ..multimodal import HolisticDetector, multimodal_bbox, multimodal_primary_hand_landmarks
from ..opencv_utils import SkinMaskConfig, apply_clahe, maybe_denoise, skin_mask_hsv, skin_mask_ycrcb
from ..pipelines.multimodal_sequence import MultimodalSequencePipeline
from ..pipelines.sequence import SequencePipeline
from ..pipelines.slt import SltPipeline
from ..tracking import RoiTracker
from .composer import ComposeConfig, ComposeMode, ComposeState


@dataclass
class VideoAnalysisConfig:
    pipeline_name: str = "landmarks"
    mode: str = "both"
    preprocess: bool = True
    skin_mask: bool = False
    mask_space: str = "ycrcb"
    use_tracker: bool = True
    confidence_threshold: float = 0.75
    stable_frames_min: int = 6
    pause_ms_min: int = 350
    cooldown_ms: int = 800
    sample_fps: float = 0.0
    max_frames: int = 0


def normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().strip().split())


def tools_status() -> dict[str, bool]:
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp_exe": shutil.which("yt-dlp") is not None,
        "yt_dlp_module": _has_yt_dlp_module(),
    }


def _has_yt_dlp_module() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _looks_like_youtube(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return "youtube.com" in host or "youtu.be" in host


def download_video_source(source_url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _looks_like_youtube(source_url):
        path = _download_youtube(source_url, out_dir)
        if path is None:
            raise RuntimeError("No se pudo descargar el video de YouTube. Instalá `yt-dlp` o subí un archivo local.")
        return path
    return _download_direct_http(source_url, out_dir)


def _download_youtube(source_url: str, out_dir: Path) -> Optional[Path]:
    outtmpl = str(out_dir / "source.%(ext)s")
    try:
        import yt_dlp

        opts = {
            "outtmpl": outtmpl,
            "format": "mp4/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source_url, download=True)
            downloaded = ydl.prepare_filename(info)
        path = Path(downloaded)
        if path.suffix.lower() != ".mp4":
            mp4 = path.with_suffix(".mp4")
            if mp4.exists():
                path = mp4
        if path.exists():
            return path
    except Exception:
        pass

    tool = shutil.which("yt-dlp")
    if not tool:
        return None
    cmd = [
        tool,
        "-f",
        "mp4/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        outtmpl,
        source_url,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        return None
    matches = sorted(out_dir.glob("source.*"))
    return matches[0] if matches else None


def _download_direct_http(source_url: str, out_dir: Path) -> Path:
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    out_path = out_dir / f"source{suffix}"
    with requests.get(source_url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return out_path


def load_video_pipeline(pipelines, pipeline_name: str):
    if pipeline_name == "landmarks":
        base = pipelines.landmarks
    elif pipeline_name == "cnn":
        base = pipelines.cnn
    elif pipeline_name == "sequence":
        base = pipelines.sequence
    elif pipeline_name == "multimodal":
        base = pipelines.multimodal
    elif pipeline_name == "slt":
        base = pipelines.slt
    else:
        raise ValueError(f"unknown pipeline: {pipeline_name}")
    if base is None:
        raise ValueError(f"pipeline not loaded: {pipeline_name}")
    pipe = copy.deepcopy(base)
    if hasattr(pipe, "_ema"):
        pipe._ema = None
    if isinstance(pipe, (SequencePipeline, MultimodalSequencePipeline)):
        pipe.reset()
    return pipe


def _draw_overlay(
    frame_bgr: np.ndarray,
    label: str,
    confidence: float,
    bbox: Optional[tuple[int, int, int, int]],
    landmarks_px: Optional[list[list[int]]],
    tracker_status: str,
    no_hand: bool,
    compose_text: str,
) -> np.ndarray:
    out = frame_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        color = (80, 80, 255) if no_hand else (80, 255, 120)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    if landmarks_px:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for a, b in connections:
            if a < len(landmarks_px) and b < len(landmarks_px):
                pa = tuple(int(v) for v in landmarks_px[a])
                pb = tuple(int(v) for v in landmarks_px[b])
                cv2.line(out, pa, pb, (255, 180, 80), 2, cv2.LINE_AA)
        for idx, p in enumerate(landmarks_px):
            color = (120, 220, 255) if idx == 0 else (255, 255, 255)
            cv2.circle(out, tuple(int(v) for v in p), 3, color, -1, cv2.LINE_AA)

    overlay_lines = [
        f"pred={label} conf={confidence:.2f} tracker={tracker_status}",
        compose_text[-72:] if compose_text else "(sin texto confirmado)",
    ]
    y = 24
    for line in overlay_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 245, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def analyze_video(
    video_path: Path,
    pipeline,
    detector: HandDetector,
    skin_cfg: SkinMaskConfig,
    config: VideoAnalysisConfig,
    output_video_path: Optional[Path] = None,
) -> dict:
    if config.pipeline_name == "slt":
        if not isinstance(pipeline, SltPipeline):
            raise RuntimeError("SLT pipeline not loaded")
        started = time.perf_counter()
        pred = pipeline.predict_video_file(
            video_path,
            sample_fps=float(config.sample_fps) if config.sample_fps > 0 else None,
            max_frames=int(config.max_frames) if config.max_frames > 0 else None,
            preprocess=bool(config.preprocess),
        )
        cap_render = cv2.VideoCapture(str(video_path))
        writer = None
        try:
            if output_video_path is not None and cap_render.isOpened():
                width = int(cap_render.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                height = int(cap_render.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                fps = float(cap_render.get(cv2.CAP_PROP_FPS) or 25.0)
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                while True:
                    ok, frame_bgr = cap_render.read()
                    if not ok or frame_bgr is None:
                        break
                    rendered = frame_bgr.copy()
                    cv2.putText(rendered, f"SLT: {pred.text or '(sin texto)'}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 3, cv2.LINE_AA)
                    cv2.putText(rendered, f"SLT: {pred.text or '(sin texto)'}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 255), 1, cv2.LINE_AA)
                    cv2.putText(rendered, f"conf={pred.confidence:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 10, 10), 3, cv2.LINE_AA)
                    cv2.putText(rendered, f"conf={pred.confidence:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 240, 200), 1, cv2.LINE_AA)
                    writer.write(rendered)
        finally:
            cap_render.release()
            if writer is not None:
                writer.release()
        elapsed_s = max(1e-6, time.perf_counter() - started)
        return {
            "video_path": str(video_path),
            "predicted_text": pred.text,
            "predicted_tokens": pred.token_sequence,
            "frames_total": pred.frames_total,
            "frames_used": pred.frames_used,
            "predictions_count": 1 if pred.text else 0,
            "avg_confidence": float(pred.confidence),
            "tracker_status_last": "off",
            "elapsed_s": elapsed_s,
            "effective_fps": pred.frames_used / elapsed_s if elapsed_s > 0 else 0.0,
            "config": asdict(config),
            "log": [
                {
                    "ts_ms": 0,
                    "label": pred.text,
                    "confidence": float(pred.confidence),
                    "no_hand": False,
                    "tracker_status": "off",
                    "new_token": pred.text,
                }
            ],
        }

    holistic = HolisticDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    sample_step = 1
    output_fps = src_fps
    if config.sample_fps and config.sample_fps > 0:
        sample_step = max(1, int(round(src_fps / float(config.sample_fps))))
        output_fps = max(1.0, float(config.sample_fps))

    writer = None
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))

    tracker = RoiTracker()
    composer = ComposeState(
        config=ComposeConfig(
            confidence_threshold=config.confidence_threshold,
            stable_frames_min=config.stable_frames_min,
            pause_ms_min=config.pause_ms_min,
            cooldown_ms=config.cooldown_ms,
        ),
        mode=ComposeMode(name=config.mode),
    )

    frames_total = 0
    frames_used = 0
    predictions_count = 0
    conf_sum = 0.0
    tokens: list[str] = []
    last_tracker_status = "off"
    started = time.perf_counter()
    log_rows: list[dict] = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames_total += 1
            if config.max_frames and frames_used >= config.max_frames:
                break
            if sample_step > 1 and (frames_total - 1) % sample_step != 0:
                continue

            ts_ms = int(round(((frames_total - 1) / float(src_fps)) * 1000.0))
            work = frame_bgr
            if config.preprocess:
                work = apply_clahe(work)
                work = maybe_denoise(work)

            mask = None
            if config.skin_mask:
                mask = skin_mask_hsv(work, skin_cfg) if config.mask_space == "hsv" else skin_mask_ycrcb(work, skin_cfg)

            frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            hand = None
            holistic_res = None
            bbox = None
            if config.pipeline_name == "multimodal":
                holistic_res = holistic.detect(frame_rgb)
                bbox = multimodal_bbox(holistic_res, width, height) if holistic_res is not None else None
            else:
                hand = detector.detect(frame_rgb)
                bbox = hand.bbox if hand is not None else None

            if config.pipeline_name == "multimodal" and bbox is not None and config.use_tracker:
                tracker.update_from_detection(work, bbox)
                last_tracker_status = tracker.status
            elif config.pipeline_name == "multimodal" and bbox is None and config.use_tracker:
                tracked = tracker.track(work)
                last_tracker_status = tracker.status
                if tracked is not None:
                    bbox = tracked
            elif hand is not None and bbox is not None and config.use_tracker:
                tracker.update_from_detection(work, bbox)
                last_tracker_status = tracker.status
            elif hand is None and config.use_tracker:
                tracked = tracker.track(work)
                last_tracker_status = tracker.status
                if tracked is not None:
                    hand = HandResult(landmarks=None, handedness=None, bbox=tracked, score=0.0)
                    bbox = tracked
            else:
                last_tracker_status = "off"

            no_hand = hand is None and not (holistic_res is not None and holistic_res.any_hand())
            label = "no_hand"
            confidence = 0.0
            if config.pipeline_name == "multimodal":
                label, confidence = pipeline.predict_multimodal(
                    holistic_res.left_hand if holistic_res is not None else None,
                    holistic_res.right_hand if holistic_res is not None else None,
                    holistic_res.pose if holistic_res is not None else None,
                    holistic_res.face if holistic_res is not None else None,
                )
                no_hand = label == "no_hand"
            elif hand is not None:
                label, confidence = pipeline.predict(work, hand, skin_mask=mask)
                no_hand = label == "no_hand" or (hand.landmarks is None and config.pipeline_name == "landmarks")
            if label != "no_hand":
                predictions_count += 1
                conf_sum += float(confidence)

            new_token = composer.update(label=label, confidence=confidence, no_hand=no_hand, ts_ms=ts_ms)
            if new_token:
                tokens.append(new_token)

            landmarks_px = None
            if config.pipeline_name == "multimodal" and holistic_res is not None:
                primary = multimodal_primary_hand_landmarks(holistic_res)
                if primary is not None:
                    pts = primary[:, :2].copy()
                    pts[:, 0] *= width
                    pts[:, 1] *= height
                    landmarks_px = pts.astype(np.int32).tolist()
            elif hand is not None and hand.landmarks is not None:
                pts = hand.landmarks[:, :2].copy()
                pts[:, 0] *= width
                pts[:, 1] *= height
                landmarks_px = pts.astype(np.int32).tolist()

            if writer is not None:
                rendered = _draw_overlay(
                    frame_bgr=frame_bgr,
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    landmarks_px=landmarks_px,
                    tracker_status=last_tracker_status,
                    no_hand=no_hand,
                    compose_text=composer.text,
                )
                writer.write(rendered)

            log_rows.append(
                {
                    "ts_ms": ts_ms,
                    "label": label,
                    "confidence": float(confidence),
                    "no_hand": bool(no_hand),
                    "tracker_status": last_tracker_status,
                    "new_token": new_token,
                }
            )
            frames_used += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed_s = max(1e-6, time.perf_counter() - started)
    return {
        "video_path": str(video_path),
        "predicted_text": composer.text,
        "predicted_tokens": tokens,
        "frames_total": frames_total,
        "frames_used": frames_used,
        "predictions_count": predictions_count,
        "avg_confidence": (conf_sum / predictions_count) if predictions_count else 0.0,
        "tracker_status_last": last_tracker_status,
        "elapsed_s": elapsed_s,
        "effective_fps": frames_used / elapsed_s,
        "config": asdict(config),
        "log": log_rows,
    }
