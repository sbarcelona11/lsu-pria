from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from ..features import extract_landmark_features
from ..hand import HandResult


def _aggregate_window(window_feats: np.ndarray) -> np.ndarray:
    """
    Turn a (T,F) window into a fixed vector.
    Features: mean, std, delta(last-first).
    """
    if window_feats.ndim != 2:
        raise ValueError("window_feats must be (T,F)")
    mean = window_feats.mean(axis=0)
    std = window_feats.std(axis=0)
    delta = window_feats[-1] - window_feats[0]
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)


@dataclass
class SequencePipeline:
    model: object
    labels: list[str]
    window_size: int = 16
    min_frames: int = 8
    smoothing_alpha: float = 0.30

    _buf: Optional[np.ndarray] = None  # (T,F) ring buffer as a simple growing window
    _ema: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path) -> "SequencePipeline":
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

    def _push(self, feat: np.ndarray) -> None:
        if self._buf is None:
            self._buf = feat[None, :].astype(np.float32)
            return
        self._buf = np.concatenate([self._buf, feat[None, :].astype(np.float32)], axis=0)
        if self._buf.shape[0] > self.window_size:
            self._buf = self._buf[-self.window_size :]

    def predict(
        self,
        frame_bgr: np.ndarray,
        hand: HandResult,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        if hand.landmarks is None:
            return "no_hand", 0.0

        feat = extract_landmark_features(hand.landmarks, handedness=hand.handedness).astype(np.float32)
        self._push(feat)
        if self._buf is None or self._buf.shape[0] < self.min_frames:
            return "no_hand", 0.0

        x = _aggregate_window(self._buf)
        proba = self.model.predict_proba(x[None, :])[0]
        if self._ema is None:
            self._ema = proba
        else:
            self._ema = (1 - self.smoothing_alpha) * self._ema + self.smoothing_alpha * proba
        idx = int(np.argmax(self._ema))
        return self.labels[idx], float(self._ema[idx])

