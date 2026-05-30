from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ..hand import HandResult


def _crop_roi(frame_bgr: np.ndarray, bbox: Optional[tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 2 or h <= 2:
        return None
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 <= 2 or y1 - y0 <= 2:
        return None
    return frame_bgr[y0:y1, x0:x1]


def _apply_mask_to_roi(roi_bgr: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
    if roi_mask is None:
        return roi_bgr
    if roi_mask.shape[:2] != roi_bgr.shape[:2]:
        roi_mask = cv2.resize(roi_mask, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(roi_bgr, roi_bgr, mask=roi_mask)


@dataclass
class CnnPipeline:
    model: torch.nn.Module
    labels: list[str]
    device: str = "cpu"
    smoothing_alpha: float = 0.25
    _ema: Optional[torch.Tensor] = None

    @classmethod
    def load(cls, path: Path) -> "CnnPipeline":
        checkpoint = torch.load(path, map_location="cpu")
        from ..train_cnn_model import build_model

        labels = checkpoint["labels"]
        model = build_model(num_classes=len(labels))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(model=model, labels=labels, device="cpu", smoothing_alpha=checkpoint.get("smoothing_alpha", 0.25))

    def predict(
        self,
        frame_bgr: np.ndarray,
        hand: HandResult,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        roi = _crop_roi(frame_bgr, hand.bbox)
        if roi is None:
            return "no_hand", 0.0

        roi_mask = None
        if skin_mask is not None and hand.bbox is not None:
            x, y, w, h = hand.bbox
            roi_mask = skin_mask[y : y + h, x : x + w]
            roi = _apply_mask_to_roi(roi, roi_mask)

        roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            proba = F.softmax(logits, dim=1)[0]
            if self._ema is None:
                self._ema = proba
            else:
                self._ema = (1 - self.smoothing_alpha) * self._ema + self.smoothing_alpha * proba
            idx = int(torch.argmax(self._ema).item())
            return self.labels[idx], float(self._ema[idx].item())
