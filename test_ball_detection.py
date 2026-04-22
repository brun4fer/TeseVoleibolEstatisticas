"""
Standalone test script for YOLOv8 volleyball ball detection.

Run:
    python test_ball_detection.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

from ball_tracking_core import (
    BallTrackerCore,
    BallTrackingConfig,
    build_foreground_mask,
    calculate_speed,
    candidate_debug_string,
    class_name,
    draw_trajectory,
    foreground_score_at_center,
    format_optional_metric,
    get_ball_candidates,
    get_ball_detection,
    is_ball_class,
    is_valid_ball_bbox,
    is_valid_ball_position,
    pixel_distance,
    predict_next_center,
    resize_frame,
    score_ball_candidate,
    select_ball_candidate,
)


MODEL_PATH = Path("runs/detect/train5/weights/best.pt")
VIDEO_PATH = Path(
    r"C:/Users/Utilizador/Desktop/Mestrado/Tese/VideosJogos/VideoAcademica.mp4"
)

START_TIME = "00:36:57"
END_TIME = "00:46:17"

BALL_CONFIG = BallTrackingConfig()
CONF_THRESHOLD = BALL_CONFIG.conf_threshold
RESIZE_WIDTH = BALL_CONFIG.resize_width
PIXELS_PER_METER = BALL_CONFIG.pixels_per_meter
SMOOTHING_WINDOW = BALL_CONFIG.smoothing_window
MAX_SPEED_KMH = BALL_CONFIG.max_speed_kmh
MIN_BALL_BBOX_W = BALL_CONFIG.min_ball_bbox_w
MIN_BALL_BBOX_H = BALL_CONFIG.min_ball_bbox_h
MIN_BALL_BBOX_AREA = BALL_CONFIG.min_ball_bbox_area
VALID_Y_MIN_RATIO = BALL_CONFIG.valid_y_min_ratio
VALID_Y_MAX_RATIO = BALL_CONFIG.valid_y_max_ratio
VALID_X_MIN_RATIO = BALL_CONFIG.valid_x_min_ratio
VALID_X_MAX_RATIO = BALL_CONFIG.valid_x_max_ratio
MIN_MOVEMENT_PIXELS = BALL_CONFIG.min_movement_pixels
MAX_STATIONARY_FRAMES = BALL_CONFIG.max_stationary_frames
MIN_VALID_SPEED_KMH = BALL_CONFIG.min_valid_speed_kmh
USE_MIN_SPEED_FILTER = BALL_CONFIG.use_min_speed_filter
TRAJECTORY_LENGTH = BALL_CONFIG.trajectory_length
MAX_TRAJECTORY_SEGMENT_PIXELS = BALL_CONFIG.max_trajectory_segment_pixels
MAX_STEP_PIXELS = BALL_CONFIG.max_step_pixels
MAX_PREDICTION_ERROR_PIXELS = BALL_CONFIG.max_prediction_error_pixels
MAX_MISSED_FRAMES = BALL_CONFIG.max_missed_frames
USE_MOTION_GATING = BALL_CONFIG.use_motion_gating
DISTANCE_WEIGHT = BALL_CONFIG.distance_weight
PREDICTION_WEIGHT = BALL_CONFIG.prediction_weight
CONFIDENCE_WEIGHT = BALL_CONFIG.confidence_weight
FOREGROUND_WEIGHT = BALL_CONFIG.foreground_weight
LOW_FOREGROUND_PENALTY = BALL_CONFIG.low_foreground_penalty
USE_BACKGROUND_SUBTRACTION = BALL_CONFIG.use_background_subtraction
FOREGROUND_MIN_PIXELS = BALL_CONFIG.foreground_min_pixels
FOREGROUND_MIN_RATIO = BALL_CONFIG.foreground_min_ratio
FOREGROUND_PATCH_RADIUS = BALL_CONFIG.foreground_patch_radius
BG_HISTORY = BALL_CONFIG.bg_history
BG_VAR_THRESHOLD = BALL_CONFIG.bg_var_threshold
BG_DETECT_SHADOWS = BALL_CONFIG.bg_detect_shadows
SHOW_FOREGROUND_DEBUG = False
SHOW_FOREGROUND_MASK = False
WINDOW_NAME = "Ball Detection"


def make_runtime_config() -> BallTrackingConfig:
    return BallTrackingConfig(
        conf_threshold=CONF_THRESHOLD,
        resize_width=RESIZE_WIDTH,
        pixels_per_meter=PIXELS_PER_METER,
        smoothing_window=SMOOTHING_WINDOW,
        max_speed_kmh=MAX_SPEED_KMH,
        min_ball_bbox_w=MIN_BALL_BBOX_W,
        min_ball_bbox_h=MIN_BALL_BBOX_H,
        min_ball_bbox_area=MIN_BALL_BBOX_AREA,
        valid_y_min_ratio=VALID_Y_MIN_RATIO,
        valid_y_max_ratio=VALID_Y_MAX_RATIO,
        valid_x_min_ratio=VALID_X_MIN_RATIO,
        valid_x_max_ratio=VALID_X_MAX_RATIO,
        min_movement_pixels=MIN_MOVEMENT_PIXELS,
        max_stationary_frames=MAX_STATIONARY_FRAMES,
        min_valid_speed_kmh=MIN_VALID_SPEED_KMH,
        use_min_speed_filter=USE_MIN_SPEED_FILTER,
        trajectory_length=TRAJECTORY_LENGTH,
        max_trajectory_segment_pixels=MAX_TRAJECTORY_SEGMENT_PIXELS,
        max_step_pixels=MAX_STEP_PIXELS,
        max_prediction_error_pixels=MAX_PREDICTION_ERROR_PIXELS,
        max_missed_frames=MAX_MISSED_FRAMES,
        use_motion_gating=USE_MOTION_GATING,
        distance_weight=DISTANCE_WEIGHT,
        prediction_weight=PREDICTION_WEIGHT,
        confidence_weight=CONFIDENCE_WEIGHT,
        foreground_weight=FOREGROUND_WEIGHT,
        low_foreground_penalty=LOW_FOREGROUND_PENALTY,
        use_background_subtraction=USE_BACKGROUND_SUBTRACTION,
        foreground_min_pixels=FOREGROUND_MIN_PIXELS,
        foreground_min_ratio=FOREGROUND_MIN_RATIO,
        foreground_patch_radius=FOREGROUND_PATCH_RADIUS,
        bg_history=BG_HISTORY,
        bg_var_threshold=BG_VAR_THRESHOLD,
        bg_detect_shadows=BG_DETECT_SHADOWS,
    )


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


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")
    if PIXELS_PER_METER <= 0:
        raise ValueError("PIXELS_PER_METER deve ser maior que zero.")

    device_label, yolo_device = select_device()
    model = YOLO(str(MODEL_PATH))
    runtime_config = make_runtime_config()
    ball_tracker = BallTrackerCore(runtime_config)

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
            fg_mask = ball_tracker.build_foreground_mask(frame)
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
            ball_result = ball_tracker.update_from_yolo_result(
                result,
                frame.shape,
                fg_mask,
                fps=fps,
                pixels_per_meter=PIXELS_PER_METER,
            )
            detection = ball_result.selected_candidate
            displayed_speed_kmh: Optional[float] = ball_result.displayed_speed_kmh

            if detection is None:
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
                    f"frame={frame_index} ball=None speed=-- "
                    f"reason={ball_result.selection_reason} missed={ball_result.missed_frames}",
                    flush=True,
                )
            else:
                x1, y1, x2, y2 = detection["bbox"]
                cx, cy = detection["center"]
                confidence = detection["confidence"]
                fg_active_pixels = int(detection.get("fg_active_pixels", 0))
                fg_active_ratio = float(detection.get("fg_active_ratio", 0.0))
                final_score = float(detection.get("final_score", 0.0))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
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

                fg_suffix = f" fg={ball_result.foreground_reason}" if ball_result.foreground_reason else ""
                candidate_debug = candidate_debug_string(detection)
                if ball_result.ignored_jump:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_OUTLIER "
                        f"(raw={ball_result.raw_speed_kmh:.2f} km/h) reason={ball_result.selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={ball_result.missed_frames}",
                        flush=True,
                    )
                elif ball_result.ignored_stationary:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_STATIONARY "
                        f"(count={ball_tracker.stationary_counter}) reason={ball_result.selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={ball_result.missed_frames}",
                        flush=True,
                    )
                elif ball_result.ignored_low_speed:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_MIN_SPEED "
                        f"(raw={ball_result.raw_speed_kmh:.2f} km/h) reason={ball_result.selection_reason}{fg_suffix} "
                        f"{candidate_debug} missed={ball_result.missed_frames}",
                        flush=True,
                    )
                else:
                    speed_text = f"{displayed_speed_kmh:.2f} km/h" if displayed_speed_kmh is not None else "--"
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed={speed_text} "
                        f"reason={ball_result.selection_reason}{fg_suffix} {candidate_debug}",
                        flush=True,
                    )

            draw_trajectory(annotated, ball_tracker.trajectory, runtime_config)
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
