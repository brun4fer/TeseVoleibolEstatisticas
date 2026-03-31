"""
Standalone test script for YOLOv8 volleyball ball detection.

Run:
    python test_ball_detection.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import math
from typing import Deque, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = Path("runs/detect/train/weights/best.pt")
VIDEO_PATH = Path(
    r"C:/Users/Utilizador/Desktop/Mestrado/Tese/VideosJogos/VideoAcademica.mp4"
)

START_TIME = "00:29:30"
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


def get_ball_detection(result, frame_shape) -> Optional[dict]:
    if result is None or result.boxes is None:
        return None

    best_detection: Optional[dict] = None
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
        if not is_valid_ball_position(center, frame_shape):
            continue

        detection = {
            "bbox": bbox,
            "center": center,
            "confidence": confidence,
        }

        if best_detection is None or confidence > best_detection["confidence"]:
            best_detection = detection

    return best_detection


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
    trajectory: Deque[Tuple[int, int]] = deque(maxlen=TRAJECTORY_LENGTH)
    speed_history_kmh: Deque[float] = deque(maxlen=SMOOTHING_WINDOW)
    stationary_counter = 0
    last_accepted_center: Optional[Tuple[int, int]] = None
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
            infer_size = max(frame.shape[0], frame.shape[1])
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=infer_size,
                device=yolo_device,
                verbose=False,
            )

            annotated = frame.copy()
            detection = get_ball_detection(results[0] if results else None, frame.shape)
            displayed_speed_kmh: Optional[float] = None

            if detection is None:
                previous_center = None
                speed_history_kmh.clear()
                # reset trajectory when ball is lost
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
                print(f"frame={frame_index} ball=None speed=--", flush=True)
            else:
                x1, y1, x2, y2 = detection["bbox"]
                cx, cy = detection["center"]
                confidence = detection["confidence"]
                current_center = (cx, cy)
                raw_speed_kmh: Optional[float] = None
                ignored_jump = False
                ignored_stationary = False
                ignored_low_speed = False

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
                            previous_center = current_center
                            last_accepted_center = current_center
                            trajectory.append(current_center)
                            if movement_pixels is None or movement_pixels >= MIN_MOVEMENT_PIXELS:
                                stationary_counter = 0
                    else:
                        ignored_jump = True
                        previous_center = None
                        speed_history_kmh.clear()
                        # reset trajectory after invalid jump
                        trajectory.clear()
                        stationary_counter = 0
                        last_accepted_center = None
                else:
                    previous_center = current_center
                    last_accepted_center = current_center
                    stationary_counter = 0
                    trajectory.append(current_center)

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

                if ignored_jump:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_OUTLIER "
                        f"(raw={raw_speed_kmh:.2f} km/h)",
                        flush=True,
                    )
                elif ignored_stationary:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_STATIONARY "
                        f"(count={stationary_counter})",
                        flush=True,
                    )
                elif ignored_low_speed:
                    print(
                        f"frame={frame_index} ball=({cx}, {cy}) speed=IGNORED_MIN_SPEED "
                        f"(raw={raw_speed_kmh:.2f} km/h)",
                        flush=True,
                    )
                else:
                    speed_text = f"{displayed_speed_kmh:.2f} km/h" if displayed_speed_kmh is not None else "--"
                    print(f"frame={frame_index} ball=({cx}, {cy}) speed={speed_text}", flush=True)

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
