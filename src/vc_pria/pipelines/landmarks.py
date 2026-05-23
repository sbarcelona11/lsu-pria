from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from ..hand import HandResult
from ..features import extract_landmark_features


@dataclass
class LandmarksPipeline:
    model: object
    labels: list[str]
    smoothing_alpha: float = 0.35
    _ema: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path) -> "LandmarksPipeline":
        payload = joblib.load(path)
        return cls(model=payload["model"], labels=payload["labels"], smoothing_alpha=payload.get("smoothing_alpha", 0.35))

    def predict(
        self,
        frame_bgr: np.ndarray,
        hand: HandResult,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        if hand.landmarks is None:
            return "no_hand", 0.0
        feat = extract_landmark_features(hand.landmarks, handedness=hand.handedness)
        proba = self.model.predict_proba(feat[None, :])[0]
        if self._ema is None:
            self._ema = proba
        else:
            self._ema = (1 - self.smoothing_alpha) * self._ema + self.smoothing_alpha * proba
        idx = int(np.argmax(self._ema))
        return self.labels[idx], float(self._ema[idx])

