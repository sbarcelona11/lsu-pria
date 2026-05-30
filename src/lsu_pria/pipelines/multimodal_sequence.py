from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from ..features import extract_multimodal_frame_features


@dataclass
class MultimodalSequencePipeline:
    model: object
    labels: list[str]
    window_size: int = 16
    min_frames: int = 8
    smoothing_alpha: float = 0.30

    _buf: Optional[np.ndarray] = None
    _ema: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path) -> "MultimodalSequencePipeline":
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            labels=payload["labels"],
            window_size=int(payload.get("window_size", 16)),
            min_frames=int(payload.get("min_frames", 8)),
            smoothing_alpha=float(payload.get("smoothing_alpha", 0.30)),
        )

    def reset(self) -> None:
        self._buf = None
        self._ema = None

    def push_frame_feature(self, feat: np.ndarray) -> Tuple[str, float]:
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if self._buf is None:
            self._buf = feat[None, :]
        else:
            self._buf = np.concatenate([self._buf, feat[None, :]], axis=0)
            if self._buf.shape[0] > self.window_size:
                self._buf = self._buf[-self.window_size :]
        if self._buf is None or self._buf.shape[0] < self.min_frames:
            return "no_hand", 0.0
        mean = self._buf.mean(axis=0)
        std = self._buf.std(axis=0)
        delta = self._buf[-1] - self._buf[0]
        x = np.concatenate([mean, std, delta], axis=0).astype(np.float32)
        proba = self.model.predict_proba(x[None, :])[0]
        if self._ema is None:
            self._ema = proba
        else:
            self._ema = (1 - self.smoothing_alpha) * self._ema + self.smoothing_alpha * proba
        idx = int(np.argmax(self._ema))
        return self.labels[idx], float(self._ema[idx])

    def predict_multimodal(
        self,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
        pose: np.ndarray | None,
        face: np.ndarray | None,
    ) -> Tuple[str, float]:
        if left_hand is None and right_hand is None:
            return "no_hand", 0.0
        feat = extract_multimodal_frame_features(left_hand, right_hand, pose, face)
        return self.push_frame_feature(feat)
