"""
ball_tracking_core.py
---------------------
Shared volleyball ball decision engine.

The logic here is the ball pipeline from test_ball_detection.py: YOLO candidate
parsing, foreground scoring, temporal motion gating, speed/stationary rejection,
trajectory updates and missed-frame reset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


TRACK_OBSERVED = "observed"
TRACK_PREDICTED = "predicted"
TRACK_LOST = "lost"


@dataclass
class BallTrackingConfig:
    conf_threshold: float = 0.15
    resize_width: int = 1280
    pixels_per_meter: float = 50.0
    smoothing_window: int = 5
    max_speed_kmh: float = 150.0
    min_ball_bbox_w: int = 8
    min_ball_bbox_h: int = 8
    min_ball_bbox_area: int = 64
    valid_y_min_ratio: float = 0.18
    valid_y_max_ratio: float = 0.92
    valid_x_min_ratio: float = 0.03
    valid_x_max_ratio: float = 0.97
    min_movement_pixels: float = 5.0
    max_stationary_frames: int = 3
    min_valid_speed_kmh: float = 1.0
    use_min_speed_filter: bool = True
    trajectory_length: int = 40
    max_trajectory_segment_pixels: float = 120.0
    max_step_pixels: float = 120.0
    max_prediction_error_pixels: float = 90.0
    max_missed_frames: int = 5
    use_motion_gating: bool = True
    distance_weight: float = 1.0
    prediction_weight: float = 1.4
    confidence_weight: float = 0.5
    foreground_weight: float = 0.4
    low_foreground_penalty: float = 25.0
    use_background_subtraction: bool = True
    foreground_min_pixels: int = 6
    foreground_min_ratio: float = 0.08
    foreground_patch_radius: int = 10
    bg_history: int = 500
    bg_var_threshold: float = 16.0
    bg_detect_shadows: bool = False


DEFAULT_CONFIG = BallTrackingConfig()


@dataclass
class BallTrackResult:
    ball_track_state: str
    accepted_detection: Optional[Dict]
    selected_candidate: Optional[Dict]
    accepted_ball_center: Optional[Tuple[int, int]]
    predicted_ball_center: Optional[Tuple[int, int]]
    selection_reason: str
    foreground_reason: str
    candidate_stats: Dict
    displayed_speed_kmh: Optional[float]
    raw_speed_kmh: Optional[float]
    speed_px: Optional[float]
    speed_px_mean: Optional[float]
    missed_frames: int
    trajectory: List[Tuple[int, int]]
    ignored_jump: bool = False
    ignored_stationary: bool = False
    ignored_low_speed: bool = False
    detection_accepted: bool = False
    debug: Dict = field(default_factory=dict)


def resize_frame(frame, target_width: int):
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame

    scale = target_width / float(width)
    target_height = int(height * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def resize_frame_with_scale(frame, target_width: int):
    resized = resize_frame(frame, target_width)
    height, width = frame.shape[:2]
    resized_h, resized_w = resized.shape[:2]
    scale_x = width / float(resized_w)
    scale_y = height / float(resized_h)
    return resized, scale_x, scale_y


def class_name(class_id: int, names) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), "")).lower()
    if isinstance(names, list) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)]).lower()
    return ""


def is_ball_class(class_id: int, names) -> bool:
    name = class_name(class_id, names)
    if "ball" in name:
        return True
    if isinstance(names, dict) and len(names) == 1 and int(class_id) == 0:
        return True
    if isinstance(names, list) and len(names) == 1 and int(class_id) == 0:
        return True
    return False


def is_valid_ball_position(
    center: Tuple[int, int],
    frame_shape,
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> bool:
    h, w = frame_shape[:2]
    cx, cy = center

    min_x = int(w * cfg.valid_x_min_ratio)
    max_x = int(w * cfg.valid_x_max_ratio)
    min_y = int(h * cfg.valid_y_min_ratio)
    max_y = int(h * cfg.valid_y_max_ratio)

    return min_x <= cx <= max_x and min_y <= cy <= max_y


def is_valid_ball_bbox(
    bbox: Tuple[int, int, int, int],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> bool:
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area = bw * bh

    return bw >= cfg.min_ball_bbox_w and bh >= cfg.min_ball_bbox_h and area >= cfg.min_ball_bbox_area


def create_background_subtractor(cfg: BallTrackingConfig = DEFAULT_CONFIG):
    if not cfg.use_background_subtraction:
        return None
    return cv2.createBackgroundSubtractorMOG2(
        history=cfg.bg_history,
        varThreshold=cfg.bg_var_threshold,
        detectShadows=cfg.bg_detect_shadows,
    )


def build_foreground_mask(bg_subtractor, frame):
    fg_mask = bg_subtractor.apply(frame)

    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return fg_mask


def foreground_score_at_center(
    fg_mask,
    center: Tuple[int, int],
    radius: int,
) -> Tuple[int, float]:
    h, w = fg_mask.shape[:2]
    cx, cy = center

    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius + 1)
    y2 = min(h, cy + radius + 1)

    roi = fg_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0, 0.0

    active_pixels = int(np.count_nonzero(roi))
    active_ratio = active_pixels / float(roi.size)

    return active_pixels, active_ratio


def get_ball_candidates(
    result,
    frame_shape=None,
    fg_mask=None,
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> Tuple[List[dict], dict]:
    if result is None or result.boxes is None:
        return [], {
            "foreground_filter_active": bool(cfg.use_background_subtraction and fg_mask is not None),
        }

    candidates: List[dict] = []
    foreground_filter_active = bool(cfg.use_background_subtraction and fg_mask is not None)
    for box in result.boxes:
        class_id = int(box.cls.item())
        if not is_ball_class(class_id, result.names):
            continue

        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bbox = (x1, y1, x2, y2)
        center = (cx, cy)

        if not is_valid_ball_bbox(bbox, cfg):
            continue

        if frame_shape is not None and not is_valid_ball_position(center, frame_shape, cfg):
            continue

        fg_active_pixels = 0
        fg_active_ratio = 0.0
        low_foreground = False
        if foreground_filter_active:
            fg_active_pixels, fg_active_ratio = foreground_score_at_center(
                fg_mask,
                center,
                cfg.foreground_patch_radius,
            )
            low_foreground = (
                fg_active_pixels < cfg.foreground_min_pixels
                and fg_active_ratio < cfg.foreground_min_ratio
            )

        candidates.append(
            {
                "bbox": bbox,
                "center": center,
                "confidence": confidence,
                "fg_active_pixels": fg_active_pixels,
                "fg_active_ratio": fg_active_ratio,
                "low_foreground": low_foreground,
            }
        )

    return candidates, {
        "foreground_filter_active": foreground_filter_active,
    }


def get_ball_detection(result, frame_shape=None, fg_mask=None, cfg: BallTrackingConfig = DEFAULT_CONFIG) -> Optional[dict]:
    candidates, _stats = get_ball_candidates(result, frame_shape, fg_mask, cfg)
    if not candidates:
        return None
    return max(candidates, key=lambda det: float(det["confidence"]))


def calculate_speed(
    previous_center: Tuple[int, int],
    current_center: Tuple[int, int],
    fps: float,
    pixels_per_meter: float,
) -> Tuple[float, float, float, float]:
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance_pixels = math.hypot(dx, dy)
    velocity_pixels_per_second = distance_pixels * fps
    velocity_mps = velocity_pixels_per_second / pixels_per_meter
    velocity_kmh = velocity_mps * 3.6
    return distance_pixels, velocity_pixels_per_second, velocity_mps, velocity_kmh


def pixel_distance(
    p1: Optional[Tuple[int, int]],
    p2: Optional[Tuple[int, int]],
) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def predict_next_center(
    previous_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if previous_center is None or older_center is None:
        return None

    vx = previous_center[0] - older_center[0]
    vy = previous_center[1] - older_center[1]
    return (previous_center[0] + vx, previous_center[1] + vy)


def format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{value:.2f}"


def candidate_debug_string(candidate: dict) -> str:
    return (
        f"conf={float(candidate['confidence']):.2f} "
        f"step={format_optional_metric(candidate.get('step_distance'))} "
        f"pred={format_optional_metric(candidate.get('prediction_error'))} "
        f"fg_px={int(candidate.get('fg_active_pixels', 0))} "
        f"fg_rt={float(candidate.get('fg_active_ratio', 0.0)):.2f} "
        f"score={format_optional_metric(candidate.get('final_score'))}"
    )


def score_ball_candidate(
    candidate: dict,
    previous_center: Optional[Tuple[int, int]],
    predicted_center: Optional[Tuple[int, int]],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> float:
    step_distance = pixel_distance(previous_center, candidate["center"])
    prediction_error = pixel_distance(predicted_center, candidate["center"])

    confidence = float(candidate["confidence"])
    fg_active_pixels = int(candidate.get("fg_active_pixels", 0))
    fg_active_ratio = float(candidate.get("fg_active_ratio", 0.0))
    low_foreground = bool(candidate.get("low_foreground", False))

    step_cost = (step_distance or 0.0) * cfg.distance_weight
    prediction_cost = (prediction_error or 0.0) * cfg.prediction_weight
    confidence_bonus = confidence * 100.0 * cfg.confidence_weight
    foreground_bonus = (fg_active_pixels + (fg_active_ratio * 100.0)) * cfg.foreground_weight
    low_foreground_penalty = cfg.low_foreground_penalty if low_foreground else 0.0
    final_score = step_cost + prediction_cost + low_foreground_penalty - confidence_bonus - foreground_bonus

    candidate["step_distance"] = step_distance
    candidate["prediction_error"] = prediction_error
    candidate["final_score"] = final_score

    return final_score


def select_ball_candidate(
    candidates: List[dict],
    previous_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> Tuple[Optional[dict], str]:
    if not candidates:
        return None, "missed_frame_no_valid_candidate"

    if not cfg.use_motion_gating:
        return (
            min(
                candidates,
                key=lambda candidate: score_ball_candidate(candidate, None, None, cfg),
            ),
            "selected_by_confidence",
        )

    if previous_center is None:
        return (
            min(
                candidates,
                key=lambda candidate: score_ball_candidate(candidate, None, None, cfg),
            ),
            "selected_by_confidence",
        )

    if older_center is None:
        gated_candidates = [
            candidate
            for candidate in candidates
            if (pixel_distance(previous_center, candidate["center"]) or 0.0) <= cfg.max_step_pixels
        ]
        if not gated_candidates:
            return None, "rejected_step_distance"
        return (
            min(
                gated_candidates,
                key=lambda candidate: score_ball_candidate(candidate, previous_center, None, cfg),
            ),
            "selected_by_motion_gate",
        )

    predicted_center = predict_next_center(previous_center, older_center)
    gated_candidates: List[dict] = []
    rejected_step_distance = False
    rejected_prediction_error = False

    for candidate in candidates:
        step_distance = pixel_distance(previous_center, candidate["center"]) or 0.0
        if step_distance > cfg.max_step_pixels:
            rejected_step_distance = True
            continue

        prediction_error = pixel_distance(predicted_center, candidate["center"]) or 0.0
        if prediction_error > cfg.max_prediction_error_pixels:
            rejected_prediction_error = True
            continue

        gated_candidates.append(candidate)

    if not gated_candidates:
        if rejected_prediction_error:
            return None, "rejected_prediction_error"
        if rejected_step_distance:
            return None, "rejected_step_distance"
        return None, "missed_frame_no_valid_candidate"

    return (
        min(
            gated_candidates,
            key=lambda candidate: score_ball_candidate(candidate, previous_center, predicted_center, cfg),
        ),
        "selected_by_motion_gate",
    )


def draw_trajectory(
    frame,
    points: Deque[Tuple[int, int]] | List[Tuple[int, int]],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> None:
    if len(points) < 2:
        return

    for index in range(1, len(points)):
        segment_distance = math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
        if segment_distance > cfg.max_trajectory_segment_pixels:
            continue
        cv2.line(frame, points[index - 1], points[index], (0, 255, 255), 2, cv2.LINE_AA)


def scale_point(point: Optional[Tuple[int, int]], scale_x: float, scale_y: float) -> Optional[Tuple[float, float]]:
    if point is None:
        return None
    return float(point[0]) * float(scale_x), float(point[1]) * float(scale_y)


def scale_detection(detection: Optional[Dict], scale_x: float, scale_y: float) -> Optional[Dict]:
    if detection is None:
        return None
    scaled = dict(detection)
    x1, y1, x2, y2 = detection["bbox"]
    cx, cy = detection["center"]
    scaled["bbox"] = (
        float(x1) * float(scale_x),
        float(y1) * float(scale_y),
        float(x2) * float(scale_x),
        float(y2) * float(scale_y),
    )
    scaled["center"] = (float(cx) * float(scale_x), float(cy) * float(scale_y))
    return scaled


def scale_points(points: List[Tuple[int, int]], scale_x: float, scale_y: float) -> List[Tuple[int, int]]:
    return [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points]


class BallTrackerCore:
    def __init__(self, cfg: BallTrackingConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.bg_subtractor = create_background_subtractor(cfg)
        self.previous_center: Optional[Tuple[int, int]] = None
        self.older_center: Optional[Tuple[int, int]] = None
        self.trajectory: Deque[Tuple[int, int]] = deque(maxlen=cfg.trajectory_length)
        self.speed_history_kmh: Deque[float] = deque(maxlen=cfg.smoothing_window)
        self.stationary_counter = 0
        self.last_accepted_center: Optional[Tuple[int, int]] = None
        self.missed_frames = 0
        self.last_result: Optional[BallTrackResult] = None

    def reset(self, center: Optional[Tuple[int, int]] = None) -> None:
        self.previous_center = center
        self.older_center = None
        self.trajectory.clear()
        if center is not None:
            self.trajectory.append((int(center[0]), int(center[1])))
        self.speed_history_kmh.clear()
        self.stationary_counter = 0
        self.last_accepted_center = center
        self.missed_frames = 0
        self.last_result = None

    def build_foreground_mask(self, frame):
        if self.bg_subtractor is None:
            return None
        return build_foreground_mask(self.bg_subtractor, frame)

    def trajectory_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        points = list(self.trajectory)
        if last_n is not None:
            return points[-max(1, int(last_n)) :]
        return points

    def _clear_track_after_misses(self) -> None:
        self.previous_center = None
        self.older_center = None
        self.speed_history_kmh.clear()
        self.trajectory.clear()
        self.stationary_counter = 0
        self.last_accepted_center = None

    def _register_miss(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > self.cfg.max_missed_frames:
            self._clear_track_after_misses()

    def update_from_yolo_result(
        self,
        result,
        frame_shape,
        fg_mask,
        fps: float,
        pixels_per_meter: Optional[float] = None,
        context_evaluator: Optional[Callable[[Dict], Dict]] = None,
    ) -> BallTrackResult:
        cfg = self.cfg
        ppm = cfg.pixels_per_meter if pixels_per_meter is None else float(pixels_per_meter)
        predicted_before = predict_next_center(self.previous_center, self.older_center)
        ball_candidates, candidate_stats = get_ball_candidates(result, frame_shape, fg_mask, cfg)
        detection, selection_reason = select_ball_candidate(
            ball_candidates,
            self.previous_center,
            self.older_center,
            cfg,
        )
        context_decision: Optional[Dict] = None
        context_rejected_candidate: Optional[Dict] = None
        if detection is not None and context_evaluator is not None:
            context_decision = context_evaluator(detection)
            if context_decision:
                detection["game_context"] = dict(context_decision)
                candidate_stats["game_context"] = dict(context_decision)
                if bool(context_decision.get("reject", False)):
                    context_rejected_candidate = dict(detection)
                    reason = str(context_decision.get("reason") or "context_rejected")
                    selection_reason = f"{selection_reason}|game_context_rejected:{reason}"
                    detection = None
        foreground_reason = ""
        if detection is not None and not candidate_stats["foreground_filter_active"]:
            foreground_reason = "selected_without_foreground_filter"
        elif detection is not None:
            foreground_reason = "selected_with_foreground"

        selected_candidate = dict(detection) if detection is not None else context_rejected_candidate
        accepted_detection: Optional[Dict] = None
        accepted_center: Optional[Tuple[int, int]] = None
        displayed_speed_kmh: Optional[float] = None
        raw_speed_kmh: Optional[float] = None
        speed_px: Optional[float] = None
        ignored_jump = False
        ignored_stationary = False
        ignored_low_speed = False
        detection_accepted = False

        if detection is None:
            self._register_miss()
        else:
            current_center = detection["center"]

            if self.previous_center is not None:
                movement_pixels = pixel_distance(self.previous_center, current_center)
                _dist_px, _speed_pxps, _speed_mps, raw_speed_kmh = calculate_speed(
                    self.previous_center,
                    current_center,
                    fps,
                    ppm,
                )
                speed_px = movement_pixels

                if raw_speed_kmh <= cfg.max_speed_kmh:
                    if movement_pixels is not None and movement_pixels < cfg.min_movement_pixels:
                        self.stationary_counter += 1
                    else:
                        self.stationary_counter = 0

                    if self.stationary_counter >= cfg.max_stationary_frames:
                        ignored_stationary = True
                    elif cfg.use_min_speed_filter and raw_speed_kmh < cfg.min_valid_speed_kmh:
                        ignored_low_speed = True
                    else:
                        self.speed_history_kmh.append(raw_speed_kmh)
                        self.older_center = self.previous_center
                        self.previous_center = current_center
                        self.last_accepted_center = current_center
                        self.missed_frames = 0
                        self.trajectory.append(current_center)
                        detection_accepted = True
                        if movement_pixels is None or movement_pixels >= cfg.min_movement_pixels:
                            self.stationary_counter = 0
                else:
                    ignored_jump = True
            else:
                self.older_center = self.previous_center
                self.previous_center = current_center
                self.last_accepted_center = current_center
                self.stationary_counter = 0
                self.missed_frames = 0
                self.trajectory.append(current_center)
                detection_accepted = True

            if not detection_accepted:
                self._register_miss()
            else:
                accepted_detection = dict(detection)
                accepted_center = current_center

            if self.speed_history_kmh:
                displayed_speed_kmh = sum(self.speed_history_kmh) / len(self.speed_history_kmh)

        predicted_after = predict_next_center(self.previous_center, self.older_center)
        if detection_accepted:
            ball_track_state = TRACK_OBSERVED
            predicted_center = predicted_after
        elif self.previous_center is not None:
            ball_track_state = TRACK_PREDICTED
            predicted_center = predicted_after if predicted_after is not None else self.previous_center
        else:
            ball_track_state = TRACK_LOST
            predicted_center = None

        if selected_candidate is not None:
            selected_candidate["track_state"] = ball_track_state
            if context_decision is not None:
                selected_candidate["game_context"] = dict(context_decision)
        if accepted_detection is not None:
            accepted_detection["track_state"] = TRACK_OBSERVED
            accepted_detection["accepted_ball_center"] = accepted_center
            accepted_detection["predicted_ball_center"] = predicted_center
            accepted_detection["speed_kmh"] = displayed_speed_kmh
            accepted_detection["speed_px"] = speed_px
            accepted_detection["speed_px_mean"] = 0.0
            if len(self.trajectory) >= 2:
                speeds = [
                    pixel_distance(self.trajectory[i - 1], self.trajectory[i]) or 0.0
                    for i in range(1, len(self.trajectory))
                ]
                accepted_detection["speed_px_mean"] = float(sum(speeds) / len(speeds)) if speeds else 0.0

        result_obj = BallTrackResult(
            ball_track_state=ball_track_state,
            accepted_detection=accepted_detection,
            selected_candidate=selected_candidate,
            accepted_ball_center=accepted_center,
            predicted_ball_center=predicted_center,
            selection_reason=selection_reason,
            foreground_reason=foreground_reason,
            candidate_stats=candidate_stats,
            displayed_speed_kmh=displayed_speed_kmh,
            raw_speed_kmh=raw_speed_kmh,
            speed_px=speed_px,
            speed_px_mean=accepted_detection.get("speed_px_mean") if accepted_detection is not None else None,
            missed_frames=self.missed_frames,
            trajectory=self.trajectory_points(),
            ignored_jump=ignored_jump,
            ignored_stationary=ignored_stationary,
            ignored_low_speed=ignored_low_speed,
            detection_accepted=detection_accepted,
            debug={
                "predicted_before": predicted_before,
                "predicted_after": predicted_after,
                "candidate_debug": candidate_debug_string(selected_candidate) if selected_candidate is not None else None,
                "game_context": dict(context_decision) if context_decision is not None else None,
            },
        )
        self.last_result = result_obj
        return result_obj
