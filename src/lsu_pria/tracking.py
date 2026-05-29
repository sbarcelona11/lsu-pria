from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .hand import BBox


def _create_tracker() -> Optional[cv2.Tracker]:
    # Prefer CSRT if available; fall back to KCF.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    return None


@dataclass
class RoiTracker:
    tracker: Optional[cv2.Tracker] = None
    bbox: Optional[BBox] = None
    status: str = "off"

    def reset(self) -> None:
        self.tracker = None
        self.bbox = None
        self.status = "off"

    def update_from_detection(self, frame_bgr: np.ndarray, bbox: BBox) -> None:
        tr = _create_tracker()
        if tr is None:
            self.status = "unavailable"
            return
        norm_bbox = tuple(int(v) for v in bbox)
        ok = tr.init(frame_bgr, norm_bbox)
        if ok:
            self.tracker = tr
            self.bbox = norm_bbox
            self.status = "tracking"
        else:
            self.status = "init_failed"

    def track(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        if self.tracker is None:
            self.status = "no_tracker"
            return None
        ok, box = self.tracker.update(frame_bgr)
        if not ok:
            self.status = "lost"
            return None
        x, y, w, h = [int(v) for v in box]
        self.bbox = (x, y, w, h)
        self.status = "tracking"
        return self.bbox
