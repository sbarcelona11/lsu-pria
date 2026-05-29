from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def maybe_denoise(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(frame_bgr, None, 3, 3, 7, 21)


@dataclass
class SkinMaskConfig:
    # YCrCb thresholds (common heuristic)
    y_min: int = 0
    y_max: int = 255
    cr_min: int = 135
    cr_max: int = 180
    cb_min: int = 85
    cb_max: int = 135
    morph_kernel: int = 5

    # HSV thresholds (optional alternative)
    h_min: int = 0
    h_max: int = 25
    s_min: int = 40
    s_max: int = 255
    v_min: int = 60
    v_max: int = 255


def skin_mask_ycrcb(frame_bgr: np.ndarray, cfg: SkinMaskConfig) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([cfg.y_min, cfg.cr_min, cfg.cb_min], dtype=np.uint8)
    upper = np.array([cfg.y_max, cfg.cr_max, cfg.cb_max], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def skin_mask_hsv(frame_bgr: np.ndarray, cfg: SkinMaskConfig) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper = np.array([cfg.h_max, cfg.s_max, cfg.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def compute_fps(prev_fps: float, dt: float, alpha: float = 0.2) -> float:
    if dt <= 1e-6:
        return prev_fps
    inst = 1.0 / dt
    if prev_fps <= 0:
        return inst
    return (1 - alpha) * prev_fps + alpha * inst


def draw_text_block(
    frame_bgr: np.ndarray,
    lines: Iterable[str],
    org: Tuple[int, int] = (10, 10),
    font_scale: float = 0.6,
    line_h: int = 20,
) -> None:
    x0, y0 = org
    for i, line in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(frame_bgr, line, (x0, y + 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (x0, y + 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
