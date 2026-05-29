from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


BBox = Tuple[int, int, int, int]  # x,y,w,h


@dataclass
class HandResult:
    landmarks: Optional[np.ndarray]  # (21,3) in normalized image coords
    handedness: Optional[str]
    bbox: Optional[BBox]
    score: float


class HandDetector:
    def __init__(self, max_num_hands: int = 1) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: np.ndarray) -> Optional[HandResult]:
        h, w = frame_rgb.shape[:2]
        out = self._hands.process(frame_rgb)
        if not out.multi_hand_landmarks:
            return None

        lm = out.multi_hand_landmarks[0]
        handed = None
        score = 0.0
        if out.multi_handedness:
            handed = out.multi_handedness[0].classification[0].label
            score = float(out.multi_handedness[0].classification[0].score)

        landmarks = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h
        x0, x1 = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(xs.max(), 0, w - 1))
        y0, y1 = int(np.clip(ys.min(), 0, h - 1)), int(np.clip(ys.max(), 0, h - 1))
        pad = int(0.12 * max(x1 - x0, y1 - y0))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
        bbox = (x0, y0, x1 - x0, y1 - y0)

        return HandResult(landmarks=landmarks, handedness=handed, bbox=bbox, score=score)

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        lm = self._mp_hands.HandLandmark
        proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for i in range(21):
            proto.landmark.add(x=float(landmarks[i, 0]), y=float(landmarks[i, 1]), z=float(landmarks[i, 2]))
        self._mp_draw.draw_landmarks(
            frame_bgr,
            proto,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self._mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )

