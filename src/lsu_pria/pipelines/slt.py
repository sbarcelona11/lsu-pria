from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ..slt_features import (
    aggregate_sequence_embedding,
    extract_slt_features_from_frames,
    extract_slt_features_from_video,
    normalize_slt_text,
)


@dataclass
class SltPrediction:
    text: str
    confidence: float
    token_sequence: list[str]
    embedding: np.ndarray
    frames_total: int
    frames_used: int


@dataclass
class SltPipeline:
    model: object
    labels: list[str]
    sample_fps: float = 6.0
    max_frames: int = 0
    min_frames: int = 4
    preprocess: bool = True
    model_type: str = "proxy_knn"

    @classmethod
    def load(cls, path: Path) -> "SltPipeline":
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            labels=list(payload["labels"]),
            sample_fps=float(payload.get("sample_fps", 6.0)),
            max_frames=int(payload.get("max_frames", 0)),
            min_frames=int(payload.get("min_frames", 4)),
            preprocess=bool(payload.get("preprocess", True)),
            model_type=str(payload.get("model_type", "proxy_knn")),
        )

    def predict_from_sequence(self, seq_features: np.ndarray) -> tuple[str, float, np.ndarray]:
        emb = aggregate_sequence_embedding(seq_features, min_frames=self.min_frames)
        if emb.size == 0:
            return "", 0.0, emb
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(emb[None, :])[0]
            idx = int(np.argmax(proba))
            return self.labels[idx], float(proba[idx]), emb
        pred = self.model.predict(emb[None, :])[0]
        pred_text = str(pred)
        confidence = 1.0 if normalize_slt_text(pred_text) else 0.0
        return pred_text, confidence, emb

    def predict_video_file(
        self,
        video_path: Path,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        preprocess: bool | None = None,
    ) -> SltPrediction:
        extraction = extract_slt_features_from_video(
            video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_fps=float(self.sample_fps if sample_fps is None else sample_fps),
            max_frames=int(self.max_frames if max_frames is None else max_frames),
            preprocess=bool(self.preprocess if preprocess is None else preprocess),
            include_debug=False,
        )
        text, conf, emb = self.predict_from_sequence(extraction.features)
        toks = normalize_slt_text(text).split() if text else []
        return SltPrediction(
            text=text,
            confidence=conf,
            token_sequence=toks,
            embedding=emb,
            frames_total=extraction.frames_total,
            frames_used=extraction.frames_used,
        )

    def predict_frames(
        self,
        frames_bgr: list[np.ndarray],
        ts_ms: list[int],
        *,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        preprocess: bool | None = None,
    ) -> SltPrediction:
        extraction = extract_slt_features_from_frames(
            frames_bgr,
            ts_ms,
            sample_fps=float(self.sample_fps if sample_fps is None else sample_fps),
            max_frames=int(self.max_frames if max_frames is None else max_frames),
            preprocess=bool(self.preprocess if preprocess is None else preprocess),
            include_debug=False,
        )
        text, conf, emb = self.predict_from_sequence(extraction.features)
        toks = normalize_slt_text(text).split() if text else []
        return SltPrediction(
            text=text,
            confidence=conf,
            token_sequence=toks,
            embedding=emb,
            frames_total=extraction.frames_total,
            frames_used=extraction.frames_used,
        )
