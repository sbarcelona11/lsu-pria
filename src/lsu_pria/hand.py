from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .mediapipe_compat import ensure_mediapipe_task_model, has_legacy_solutions


BBox = Tuple[int, int, int, int]  # x,y,w,h


@dataclass
class HandResult:
    landmarks: Optional[np.ndarray]  # (21,3) in normalized image coords
    handedness: Optional[str]
    bbox: Optional[BBox]
    score: float


class HandDetector:
    def __init__(self, max_num_hands: int = 1) -> None:
        self._legacy = has_legacy_solutions()
        self._max_num_hands = int(max_num_hands)
        if self._legacy:
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = ensure_mediapipe_task_model("hand_landmarker.task")
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self._max_num_hands,
            )
            self._hands = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> Optional[HandResult]:
        h, w = frame_rgb.shape[:2]
        if self._legacy:
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
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            out = self._hands.detect(image)
            hand_lms = getattr(out, "hand_landmarks", None) or []
            if not hand_lms:
                return None
            landmarks = np.array([[p.x, p.y, p.z] for p in hand_lms[0]], dtype=np.float32)
            handed = None
            score = 0.0
            handedness = getattr(out, "handedness", None) or []
            if handedness and handedness[0]:
                cat = handedness[0][0]
                handed = getattr(cat, "category_name", None) or getattr(cat, "display_name", None)
                score = float(getattr(cat, "score", 0.0) or 0.0)

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
        if self._legacy:
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
            return

        # Simple OpenCV drawing fallback (no MediaPipe proto utilities).
        h, w = frame_bgr.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in landmarks[:, :2]]
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (0, 17),
        ]
        for a, b in connections:
            cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 2, (255, 0, 0), -1)
