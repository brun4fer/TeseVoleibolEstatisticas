"""
Standalone test script for YOLOv8 volleyball ball detection.

Run:
    python test_ball_detection.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import math
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


MODEL_PATH = Path("runs/detect/train/weights/best.pt")
VIDEO_PATH = Path(
    r"C:/Users/Utilizador/Desktop/Mestrado/Tese/VideosJogos/VideoAcademica.mp4"
)

START_TIME = "00:31:45"
END_TIME = "00:46:17"

CONF_THRESHOLD = 0.15
RESIZE_WIDTH = 1280
PIXELS_PER_METER = 50.0  # Ajustar ao video/resolucao para valores mais realistas.
SMOOTHING_WINDOW = 5
MAX_SPEED_KMH = 150.0
MIN_BALL_BBOX_W = 8
MIN_BALL_BBOX_H = 8
MIN_BALL_BBOX_AREA = 64
VALID_Y_MIN_RATIO = 0.18
VALID_Y_MAX_RATIO = 0.92
VALID_X_MIN_RATIO = 0.03
VALID_X_MAX_RATIO = 0.97
MIN_MOVEMENT_PIXELS = 5
MAX_STATIONARY_FRAMES = 3
MIN_VALID_SPEED_KMH = 1.0
USE_MIN_SPEED_FILTER = True
TRAJECTORY_LENGTH = 40
MAX_TRAJECTORY_SEGMENT_PIXELS = 120
MAX_STEP_PIXELS = 120
MAX_PREDICTION_ERROR_PIXELS = 90
MAX_MISSED_FRAMES = 5
USE_MOTION_GATING = True
DISTANCE_WEIGHT = 1.0
PREDICTION_WEIGHT = 1.4
CONFIDENCE_WEIGHT = 0.5
FOREGROUND_WEIGHT = 0.4
LOW_FOREGROUND_PENALTY = 25.0
USE_BACKGROUND_SUBTRACTION = True
FOREGROUND_MIN_PIXELS = 6
FOREGROUND_MIN_RATIO = 0.08
FOREGROUND_PATCH_RADIUS = 10
BG_HISTORY = 500
BG_VAR_THRESHOLD = 16
BG_DETECT_SHADOWS = False
SHOW_FOREGROUND_DEBUG = False
SHOW_FOREGROUND_MASK = False
WINDOW_NAME = "Ball Detection"


def time_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS, MM:SS or raw seconds to total seconds."""
    value = timestamp.strip()
    parts = value.split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)

    if len(parts) == 2:
        minutes, seconds = parts
        return (int(minutes) * 60) + float(seconds)

    if len(parts) == 1:
        return float(parts[0])

    raise ValueError(f"Timestamp invalido: {timestamp}")


def seconds_to_frame(seconds: float, fps: float) -> int:
    return int(seconds * fps)


def select_device() -> Tuple[str, int | str]:
    if torch.cuda.is_available():
        return "cuda:0", 0
    return "cpu", "cpu"


def resize_frame(frame, target_width: int):
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame

    scale = target_width / float(width)
    target_height = int(height * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


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


def is_valid_ball_position(center: Tuple[int, int], frame_shape) -> bool:
    h, w = frame_shape[:2]
    cx, cy = center

    min_x = int(w * VALID_X_MIN_RATIO)
    max_x = int(w * VALID_X_MAX_RATIO)
    min_y = int(h * VALID_Y_MIN_RATIO)
    max_y = int(h * VALID_Y_MAX_RATIO)

    return min_x <= cx <= max_x and min_y <= cy <= max_y


def is_valid_ball_bbox(bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area = bw * bh

    return (
        bw >= MIN_BALL_BBOX_W
        and bh >= MIN_BALL_BBOX_H
        and area >= MIN_BALL_BBOX_AREA
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


def get_ball_candidates(result, frame_shape=None, fg_mask=None) -> Tuple[List[dict], dict]:
    if result is None or result.boxes is None:
        return [], {
            "foreground_filter_active": bool(USE_BACKGROUND_SUBTRACTION and fg_mask is not None),
        }

    candidates: List[dict] = []
    foreground_filter_active = bool(USE_BACKGROUND_SUBTRACTION and fg_mask is not None)
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

        # valid ball bbox filter
        if not is_valid_ball_bbox(bbox):
            continue

        # valid ball position filter
        if frame_shape is not None and not is_valid_ball_position(center, frame_shape):
            continue

        fg_active_pixels = 0
        fg_active_ratio = 0.0
        low_foreground = False
        if foreground_filter_active:
            fg_active_pixels, fg_active_ratio = foreground_score_at_center(
                fg_mask,
                center,
                FOREGROUND_PATCH_RADIUS,
            )
            low_foreground = (
                fg_active_pixels < FOREGROUND_MIN_PIXELS
                and fg_active_ratio < FOREGROUND_MIN_RATIO
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


def get_ball_detection(result, frame_shape=None, fg_mask=None) -> Optional[dict]:
    candidates, _stats = get_ball_candidates(result, frame_shape, fg_mask)
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
) -> float:
    step_distance = pixel_distance(previous_center, candidate["center"])
    prediction_error = pixel_distance(predicted_center, candidate["center"])

    confidence = float(candidate["confidence"])
    fg_active_pixels = int(candidate.get("fg_active_pixels", 0))
    fg_active_ratio = float(candidate.get("fg_active_ratio", 0.0))
    low_foreground = bool(candidate.get("low_foreground", False))

    step_cost = (step_distance or 0.0) * DISTANCE_WEIGHT
    prediction_cost = (prediction_error or 0.0) * PREDICTION_WEIGHT
    confidence_bonus = confidence * 100.0 * CONFIDENCE_WEIGHT
    foreground_bonus = (fg_active_pixels + (fg_active_ratio * 100.0)) * FOREGROUND_WEIGHT
    low_foreground_penalty = LOW_FOREGROUND_PENALTY if low_foreground else 0.0
    final_score = step_cost + prediction_cost + low_foreground_penalty - confidence_bonus - foreground_bonus

    candidate["step_distance"] = step_distance
    candidate["prediction_error"] = prediction_error
    candidate["final_score"] = final_score

    return final_score


def select_ball_candidate(
    candidates: List[dict],
    previous_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
) -> Tuple[Optional[dict], str]:
    if not candidates:
        return None, "missed_frame_no_valid_candidate"

    if not USE_MOTION_GATING:
        return (
            min(
                candidates,
                key=lambda candidate: score_ball_candidate(candidate, None, None),
            ),
            "selected_by_confidence",
        )

    if previous_center is None:
        return (
            min(
                candidates,
                key=lambda candidate: score_ball_candidate(candidate, None, None),
            ),
            "selected_by_confidence",
        )

    if older_center is None:
        gated_candidates = [
            candidate
            for candidate in candidates
            if (pixel_distance(previous_center, candidate["center"]) or 0.0) <= MAX_STEP_PIXELS
        ]
        if not gated_candidates:
            return None, "rejected_step_distance"
        return (
            min(
                gated_candidates,
                key=lambda candidate: score_ball_candidate(candidate, previous_center, None),
            ),
            "selected_by_motion_gate",
        )

    predicted_center = predict_next_center(previous_center, older_center)
    gated_candidates: List[dict] = []
    rejected_step_distance = False
    rejected_prediction_error = False

    for candidate in candidates:
        step_distance = pixel_distance(previous_center, candidate["center"]) or 0.0
        if step_distance > MAX_STEP_PIXELS:
            rejected_step_distance = True
            continue

        prediction_error = pixel_distance(predicted_center, candidate["center"]) or 0.0
        if prediction_error > MAX_PREDICTION_ERROR_PIXELS:
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
            key=lambda candidate: score_ball_candidate(candidate, previous_center, predicted_center),
        ),
        "selected_by_motion_gate",
    )


def draw_trajectory(frame, points: Deque[Tuple[int, int]]) -> None:
    if len(points) < 2:
        return

    for index in range(1, len(points)):
        segment_distance = math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
        # skip unrealistic trajectory segment
        if segment_distance > MAX_TRAJECTORY_SEGMENT_PIXELS:
            continue
        cv2.line(frame, points[index - 1], points[index], (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")
    if PIXELS_PER_METER <= 0:
        raise ValueError("PIXELS_PER_METER deve ser maior que zero.")

    device_label, yolo_device = select_device()
    model = YOLO(str(MODEL_PATH))
    bg_subtractor = None
    if USE_BACKGROUND_SUBTRACTION:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=BG_DETECT_SHADOWS,
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {VIDEO_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("FPS invalido ou indisponivel no video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_seconds = time_to_seconds(START_TIME)
    end_seconds = time_to_seconds(END_TIME)
    start_frame = max(0, seconds_to_frame(start_seconds, fps))
    end_frame = seconds_to_frame(end_seconds, fps)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    if end_frame < start_frame:
        cap.release()
        raise ValueError(
            f"Janela invalida: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    previous_center: Optional[Tuple[int, int]] = None
    older_center: Optional[Tuple[int, int]] = None
    trajectory: Deque[Tuple[int, int]] = deque(maxlen=TRAJECTORY_LENGTH)
    speed_history_kmh: Deque[float] = deque(maxlen=SMOOTHING_WINDOW)
    stationary_counter = 0
    last_accepted_center: Optional[Tuple[int, int]] = None
    missed_frames = 0
    frame_index = start_frame

    print(
        f"Modelo: {MODEL_PATH} | Device: {device_label} | FPS: {fps:.2f} | "
        f"Frames: {start_frame}-{end_frame}",
        flush=True,
    )

    try:
        while frame_index <= end_frame:
            ok, frame = cap.read()
            if not ok:
                print("Leitura do video interrompida antes do fim.", flush=True)
                break

            frame = resize_frame(frame, RESIZE_WIDTH)
            fg_mask = build_foreground_mask(bg_subtractor, frame) if bg_subtractor is not None else None
            infer_size = max(frame.shape[0], frame.shape[1])
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=infer_size,
                device=yolo_device,
                verbose=False,
            )

            annotated = frame.copy()
            result = results[0] if results else None
            ball_candidates, candidate_stats = get_ball_candidates(result, frame.shape, fg_mask)
            detection, selection_reason = select_ball_candidate(
                ball_candidates,
                previous_center,
                older_center,
            )
            foreground_reason = ""
            if detection is not None and not candidate_stats["foreground_filter_active"]:
                foreground_reason = "selected_without_foreground_filter"
            elif detection is not None:
                foreground_reason = "selected_with_foreground"

            displayed_speed_kmh: Optional[float] = None

            if detection is None:
                missed_frames += 1
                if missed_frames > MAX_MISSED_FRAMES:
                    previous_center = None
                    older_center = None
                    speed_history_kmh.clear()
                    trajectory.clear()
                    stationary_counter = 0
                    last_accepted_center = None
                cv2.putText(
                    annotated,
                    "Ball: not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                print(
                    f"frame={frame_index} ball=None speed=-- reason={selection_reason} missed={missed_frames}",
                    flush=True,
                )
            else:
                x1, y1, x2, y2 = detection["bbox"]
                cx, cy = detection["center"]
                confidence = detection["confidence"]
                fg_active_pixels = int(detection.get("fg_active_pixels", 0))
                fg_active_ratio = float(detection.get("fg_active_ratio", 0.0))
                final_score = float(detection.get("final_score", 0.0))
                current_center = (cx, cy)
                raw_speed_kmh: Optional[float] = None
                ignored_jump = False
                ignored_stationary = False
                ignored_low_speed = False
                detection_accepted = False

                if previous_center is not None:
                    movement_pixels = pixel_distance(previous_center, current_center)
                    _dist_px, _speed_pxps, _speed_mps, raw_speed_kmh = calculate_speed(
                        previous_center,
                        current_center,
                        fps,
                        PIXELS_PER_METER,
                    )

                    if raw_speed_kmh <= MAX_SPEED_KMH:
                        # stationary false detection filter
                        if movement_pixels is not None and movement_pixels < MIN_MOVEMENT_PIXELS:
                            stationary_counter += 1
                        else:
                            stationary_counter = 0

                        # minimum speed filter
                        if stationary_counter >= MAX_STATIONARY_FRAMES:
                            ignored_stationary = True
                        elif USE_MIN_SPEED_FILTER and raw_speed_kmh < MIN_VALID_SPEED_KMH:
                            ignored_low_speed = True
                        else:
                            speed_history_kmh.append(raw_speed_kmh)
                            older_center = previous_center
                            previous_center = current_center
                            last_accepted_center = current_center
                            missed_frames = 0
                            trajectory.append(current_center)
                            detection_accepted = True
                            if movement_pixels is None or movement_pixels >= MIN_MOVEMENT_PIXELS:
                                stationary_counter = 0
                    else:
                        ignored_jump = True
                else:
                    older_center = previous_center
                    previous_center = current_center
                    last_accepted_center = current_center
                    stationary_counter = 0
                    missed_frames = 0
                    trajectory.append(current_center)
                    detection_accepted = True

                if not detection_accepted:
                    missed_frames += 1
                    if missed_frames > MAX_MISSED_FRAMES:
                        previous_center = None
                        older_center = None
                        speed_history_kmh.clear()
                        trajectory.clear()
                        stationary_counter = 0
                        last_accepted_center = None

                if speed_history_kmh:
                    displayed_speed_kmh = sum(speed_history_kmh) / len(speed_history_kmh)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, current_center, 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"ball {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if SHOW_FOREGROUND_DEBUG:
                    cv2.putText(
                        annotated,
                        f"fg_px={fg_active_pixels} fg_rt={fg_active_ratio:.2f} score={final_score:.2f}",
                        (x1, min(frame.shape[0] - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                fg_suffix = f" fg={foreground_reason}" if foreground_reason else ""
                candidate_debug = candidate_debug_string(detection)
                if ignored_jump:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_OUTLIER "
                        f"(raw={raw_speed_kmh:.2f} km/h) reason={selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={missed_frames}",
                        flush=True,
                    )
                elif ignored_stationary:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_STATIONARY "
                        f"(count={stationary_counter}) reason={selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={missed_frames}",
                        flush=True,
                    )
                elif ignored_low_speed:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_MIN_SPEED "
                        f"(raw={raw_speed_kmh:.2f} km/h) reason={selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={missed_frames}",
                        flush=True,
                    )
                else:
                    speed_text = f"{displayed_speed_kmh:.2f} km/h" if displayed_speed_kmh is not None else "--"
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed={speed_text} reason={selection_reason}{fg_suffix} "
                        f"{candidate_debug}",
                        flush=True,
                    )

            draw_trajectory(annotated, trajectory)
            speed_overlay = (
                f"Speed: {displayed_speed_kmh:.2f} km/h"
                if displayed_speed_kmh is not None
                else "Speed: --"
            )

            cv2.putText(
                annotated,
                speed_overlay,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Frame: {frame_index}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Device: {device_label}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if SHOW_FOREGROUND_MASK and fg_mask is not None:
                cv2.imshow("Foreground Mask", fg_mask)
            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
