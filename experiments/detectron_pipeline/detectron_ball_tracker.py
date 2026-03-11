"""
Detectron2-based volleyball detector + simple centroid tracker.

Install dependencies:
  pip install detectron2
  pip install opencv-python
  pip install torch torchvision
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config_detectron import (
    BALL_CLASS_ID,
    CONFIDENCE_THRESHOLD,
    MAX_ASPECT_RATIO_DEVIATION,
    MAX_BALL_AREA_RATIO,
    MAX_MISSING_FRAMES,
    MAX_TRACKING_DISTANCE_PX,
    MIN_BALL_AREA,
    MODEL_CONFIG_PATH,
)


@dataclass
class BallDetection:
    x: int
    y: int
    confidence: float
    bbox: tuple[int, int, int, int]


class DetectronBallTracker:
    """Detects ball candidates per frame and links them with centroid tracking."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_tracking_distance_px: int = MAX_TRACKING_DISTANCE_PX,
        max_missing_frames: int = MAX_MISSING_FRAMES,
        device: Optional[str] = None,
    ) -> None:
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise ImportError(
                "Detectron2 import failed. Install/repair with:\n"
                "  python -m pip install --no-build-isolation "
                "\"git+https://github.com/facebookresearch/detectron2.git\"\n"
                "If you see 'No module named pkg_resources', run:\n"
                "  python -m pip install \"setuptools<81\"\n"
                f"Original import error: {exc}"
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_PATH))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG_PATH)
        if device:
            cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(cfg)
        self.confidence_threshold = confidence_threshold
        self.max_tracking_distance_px = float(max_tracking_distance_px)
        self.max_missing_frames = max_missing_frames

        self._last_centroid: Optional[np.ndarray] = None
        self._missing_frames = 0

    @property
    def missing_frames(self) -> int:
        return self._missing_frames

    def detect_ball(self, frame: np.ndarray) -> Optional[BallDetection]:
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        candidates = self._extract_candidates(instances, frame.shape[:2])
        selected = self._select_candidate(candidates)

        if selected is None:
            self._missing_frames += 1
            if self._missing_frames > self.max_missing_frames:
                self._last_centroid = None
            return None

        center = selected["center"]
        self._last_centroid = center
        self._missing_frames = 0

        x1, y1, x2, y2 = selected["bbox"]
        return BallDetection(
            x=int(round(center[0])),
            y=int(round(center[1])),
            confidence=float(selected["confidence"]),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
        )

    def _extract_candidates(
        self, instances, frame_shape: tuple[int, int]
    ) -> List[Dict[str, np.ndarray | float | int | tuple[int, int, int, int]]]:
        if len(instances) == 0:
            return []

        frame_h, frame_w = frame_shape
        frame_area = float(frame_h * frame_w)
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        sports_ball_candidates: List[Dict[str, np.ndarray | float | int | tuple[int, int, int, int]]] = []
        fallback_candidates: List[Dict[str, np.ndarray | float | int | tuple[int, int, int, int]]] = []

        for box, score, class_id in zip(boxes, scores, classes):
            if float(score) < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            if not self._passes_shape_filter(x1, y1, x2, y2, frame_area):
                continue

            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            candidate: Dict[str, np.ndarray | float | int | tuple[int, int, int, int]] = {
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "confidence": float(score),
                "class_id": int(class_id),
            }

            if int(class_id) == BALL_CLASS_ID:
                sports_ball_candidates.append(candidate)
            else:
                fallback_candidates.append(candidate)

        if sports_ball_candidates:
            return sports_ball_candidates
        return fallback_candidates

    def _passes_shape_filter(
        self, x1: int, y1: int, x2: int, y2: int, frame_area: float
    ) -> bool:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = float(width * height)
        if area < MIN_BALL_AREA:
            return False
        if area > frame_area * MAX_BALL_AREA_RATIO:
            return False

        aspect = width / float(height)
        if abs(aspect - 1.0) > MAX_ASPECT_RATIO_DEVIATION:
            return False
        return True

    def _select_candidate(
        self, candidates: List[Dict[str, np.ndarray | float | int | tuple[int, int, int, int]]]
    ) -> Optional[Dict[str, np.ndarray | float | int | tuple[int, int, int, int]]]:
        if not candidates:
            return None

        if self._last_centroid is None:
            return max(candidates, key=lambda item: float(item["confidence"]))

        allowed_distance = self.max_tracking_distance_px
        if self._missing_frames > 0:
            allowed_distance *= 1.35

        best_candidate = None
        best_score = -1.0
        for candidate in candidates:
            center = candidate["center"]
            distance = float(np.linalg.norm(center - self._last_centroid))
            confidence = float(candidate["confidence"])
            if distance > allowed_distance and self._missing_frames < self.max_missing_frames:
                continue

            proximity = max(0.0, 1.0 - (distance / max(allowed_distance, 1.0)))
            score = (0.60 * confidence) + (0.40 * proximity)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        if self._missing_frames >= self.max_missing_frames:
            return max(candidates, key=lambda item: float(item["confidence"]))
        return None
