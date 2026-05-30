from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .hand import HandDetector, HandResult
from .multimodal import HolisticDetector, multimodal_bbox, multimodal_primary_hand_landmarks
from .opencv_utils import (
    SkinMaskConfig,
    apply_clahe,
    compute_fps,
    draw_text_block,
    maybe_denoise,
    skin_mask_hsv,
    skin_mask_ycrcb,
)
from .pipelines.cnn import CnnPipeline
from .pipelines.landmarks import LandmarksPipeline
from .pipelines.multimodal_sequence import MultimodalSequencePipeline
from .pipelines.sequence import SequencePipeline
from .tracking import RoiTracker


@dataclass
class DemoToggles:
    preprocess: bool = True
    show_skin_mask: bool = False
    use_tracker: bool = True
    mask_space: str = "ycrcb"  # ycrcb | hsv


class DemoRunner:
    def __init__(self, pipeline: str, model_path: Path, toggles: DemoToggles | None = None) -> None:
        self.pipeline_name = pipeline
        self.toggles = toggles or DemoToggles()
        self.detector = HandDetector(max_num_hands=1)
        self.holistic = HolisticDetector()
        self.tracker = RoiTracker()
        self.skin_cfg = SkinMaskConfig()

        if pipeline == "landmarks":
            self.pipeline = LandmarksPipeline.load(model_path)
        elif pipeline == "cnn":
            self.pipeline = CnnPipeline.load(model_path)
        elif pipeline == "sequence":
            self.pipeline = SequencePipeline.load(model_path)
        elif pipeline == "multimodal":
            self.pipeline = MultimodalSequencePipeline.load(model_path)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        self._fps = 0.0
        self._last_pred: Optional[str] = None
        self._last_conf: float = 0.0
        self._lost_hand = False

    def handle_key(self, key: int) -> None:
        if key == ord("m"):
            self.toggles.show_skin_mask = not self.toggles.show_skin_mask
        elif key == ord("t"):
            self.toggles.use_tracker = not self.toggles.use_tracker
            if not self.toggles.use_tracker:
                self.tracker.reset()
        elif key == ord("p"):
            self.toggles.preprocess = not self.toggles.preprocess
        elif key == ord("k"):
            self.toggles.mask_space = "hsv" if self.toggles.mask_space == "ycrcb" else "ycrcb"
        elif key == ord("x"):
            # Reset temporal buffers (useful for sequence pipeline)
            if hasattr(self.pipeline, "reset"):
                try:
                    self.pipeline.reset()
                except Exception:
                    pass

    def process_frame(self, frame_bgr: np.ndarray, dt: float) -> np.ndarray:
        self._fps = compute_fps(self._fps, dt)

        work_bgr = frame_bgr.copy()
        if self.toggles.preprocess:
            work_bgr = apply_clahe(work_bgr)
            work_bgr = maybe_denoise(work_bgr)

        frame_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        result = None
        holistic_res = None
        if self.pipeline_name == "multimodal":
            holistic_res = self.holistic.detect(frame_rgb)
        else:
            result = self.detector.detect(frame_rgb)

        skin_mask = None
        if self.toggles.show_skin_mask:
            skin_mask = skin_mask_ycrcb(work_bgr, self.skin_cfg) if self.toggles.mask_space == "ycrcb" else skin_mask_hsv(work_bgr, self.skin_cfg)

        hand_used: Optional[HandResult] = None
        if self.pipeline_name == "multimodal":
            bbox_mm = multimodal_bbox(holistic_res, work_bgr.shape[1], work_bgr.shape[0]) if holistic_res is not None else None
            if holistic_res is not None and holistic_res.any_hand():
                self._lost_hand = False
                if self.toggles.use_tracker and bbox_mm is not None:
                    self.tracker.update_from_detection(work_bgr, bbox_mm)
            else:
                self._lost_hand = True
            if bbox_mm is not None:
                hand_used = HandResult(
                    landmarks=multimodal_primary_hand_landmarks(holistic_res) if holistic_res is not None else None,
                    handedness=None,
                    bbox=bbox_mm,
                    score=1.0,
                )
        elif result is not None:
            hand_used = result
            self._lost_hand = False
            if self.toggles.use_tracker and result.bbox is not None:
                self.tracker.update_from_detection(work_bgr, result.bbox)
        else:
            self._lost_hand = True
            if self.toggles.use_tracker:
                tracked = self.tracker.track(work_bgr)
                if tracked is not None:
                    hand_used = HandResult(
                        landmarks=None,
                        handedness=None,
                        bbox=tracked,
                        score=0.0,
                    )

        label = None
        conf = 0.0
        if self.pipeline_name == "multimodal" and holistic_res is not None:
            label, conf = self.pipeline.predict_multimodal(
                holistic_res.left_hand,
                holistic_res.right_hand,
                holistic_res.pose,
                holistic_res.face,
            )
            self._last_pred = label
            self._last_conf = conf
        elif hand_used is not None:
            label, conf = self.pipeline.predict(work_bgr, hand_used, skin_mask=skin_mask)
            self._last_pred = label
            self._last_conf = conf

        out = frame_bgr.copy()
        if result is not None and result.landmarks is not None:
            self.detector.draw_landmarks(out, result.landmarks)
        if self.pipeline_name == "multimodal" and holistic_res is not None:
            if holistic_res.left_hand is not None:
                self.detector.draw_landmarks(out, holistic_res.left_hand)
            if holistic_res.right_hand is not None:
                self.detector.draw_landmarks(out, holistic_res.right_hand)

        bbox = hand_used.bbox if hand_used is not None else None
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if skin_mask is not None:
            mask_small = cv2.resize(skin_mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            out[0 : mask_bgr.shape[0], 0 : mask_bgr.shape[1]] = mask_bgr

        lines = [
            f"pipeline: {self.pipeline_name}",
            f"fps: {self._fps:.1f}",
            f"preprocess(p): {self.toggles.preprocess}",
            f"skin mask(m): {self.toggles.show_skin_mask} ({self.toggles.mask_space}, k to switch)",
            f"tracker(t): {self.toggles.use_tracker} ({self.tracker.status})",
            f"hand: {'lost' if self._lost_hand else 'ok'}",
            "reset temporal(x): True" if self.pipeline_name in {"sequence", "multimodal"} else "reset temporal(x): -",
        ]
        if self._last_pred is not None:
            lines.append(f"pred: {self._last_pred} ({self._last_conf:.2f})")
        draw_text_block(out, lines, org=(10, 10))
        return out
