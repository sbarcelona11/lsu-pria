from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np


POSE_KEYPOINTS = [
    0,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
]

FACE_KEYPOINTS = [
    1,
    33,
    61,
    199,
    263,
    291,
]


@dataclass
class HolisticResult:
    left_hand: Optional[np.ndarray]
    right_hand: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    face: Optional[np.ndarray]

    def any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


class HolisticDetector:
    def __init__(self) -> None:
        self._mp = mp.solutions.holistic
        self._holistic = self._mp.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _to_array(landmarks, expected: int) -> Optional[np.ndarray]:
        if landmarks is None:
            return None
        pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
        if pts.shape != (expected, 3):
            return None
        return pts

    def detect(self, frame_rgb: np.ndarray) -> HolisticResult:
        out = self._holistic.process(frame_rgb)
        left_hand = self._to_array(out.left_hand_landmarks, 21)
        right_hand = self._to_array(out.right_hand_landmarks, 21)
        pose_full = self._to_array(out.pose_landmarks, 33)
        face_full = self._to_array(out.face_landmarks, 468)

        pose = pose_full[POSE_KEYPOINTS] if pose_full is not None else None
        face = face_full[FACE_KEYPOINTS] if face_full is not None else None
        return HolisticResult(left_hand=left_hand, right_hand=right_hand, pose=pose, face=face)


def multimodal_bbox(result: HolisticResult, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    pts: list[np.ndarray] = []
    if result.left_hand is not None:
        pts.append(result.left_hand[:, :2])
    if result.right_hand is not None:
        pts.append(result.right_hand[:, :2])
    if not pts:
        return None
    xy = np.concatenate(pts, axis=0).astype(np.float32)
    xs = np.clip(xy[:, 0] * float(width), 0, max(0, width - 1))
    ys = np.clip(xy[:, 1] * float(height), 0, max(0, height - 1))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = int(0.12 * max(1, x1 - x0, y1 - y0))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(max(0, width - 1), x1 + pad)
    y1 = min(max(0, height - 1), y1 + pad)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def multimodal_primary_hand_landmarks(result: HolisticResult) -> Optional[np.ndarray]:
    if result.right_hand is not None:
        return result.right_hand
    return result.left_hand
