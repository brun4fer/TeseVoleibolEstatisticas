"""
Run the standalone Detectron2 ball-tracking experiment.

Install dependencies:
  pip install detectron2
  pip install opencv-python
  pip install torch torchvision
"""

from __future__ import annotations

import csv
import time

import cv2

from config_detectron import (
    CONFIDENCE_THRESHOLD,
    DEBUG_DIR,
    DEBUG_FRAME_INTERVAL,
    DEBUG_MAX_IMAGES,
    END_TIME,
    OUTPUTS_DIR,
    PROGRESS_EVERY_FRAMES,
    START_TIME,
    VIDEO_PATH,
)
from detectron_ball_tracker import BallDetection, DetectronBallTracker


def timestamp_to_seconds(timestamp: str) -> int:
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp '{timestamp}'. Expected HH:MM:SS.")
    hours, minutes, seconds = [int(part) for part in parts]
    return (hours * 3600) + (minutes * 60) + seconds


def timestamp_to_frame(timestamp: str, fps: float) -> int:
    return int(round(timestamp_to_seconds(timestamp) * fps))


def draw_detection(frame, detection: BallDetection | None) -> None:
    if detection is None:
        return

    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 40), 2)
    cv2.circle(frame, (detection.x, detection.y), 8, (0, 0, 255), -1)
    label = f"ball {detection.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS is invalid or unavailable.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = timestamp_to_frame(START_TIME, fps)
    end_frame = timestamp_to_frame(END_TIME, fps)
    end_frame = min(end_frame, max(total_frames - 1, 0))
    if start_frame >= end_frame:
        cap.release()
        raise ValueError(
            f"Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    annotated_video_path = OUTPUTS_DIR / "detectron_segment_annotated.mp4"
    csv_path = OUTPUTS_DIR / "detectron_ball_coordinates.csv"

    video_writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {annotated_video_path}")

    tracker = DetectronBallTracker(confidence_threshold=CONFIDENCE_THRESHOLD)

    frame_idx = start_frame
    processed_count = 0
    saved_debug = 0
    max_to_process = (end_frame - start_frame) + 1
    start_clock = time.perf_counter()

    print(
        f"Detectron2 experiment started | fps={fps:.2f} | "
        f"frame_range={start_frame}-{end_frame} ({max_to_process} frames).",
        flush=True,
    )

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame", "x", "y", "confidence"])

            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    print("Stopped early: unable to read more frames from the video.")
                    break

                detection = tracker.detect_ball(frame)

                if detection is None:
                    writer.writerow([frame_idx, "", "", ""])
                else:
                    writer.writerow(
                        [
                            frame_idx,
                            detection.x,
                            detection.y,
                            f"{detection.confidence:.6f}",
                        ]
                    )

                annotated = frame.copy()
                draw_detection(annotated, detection)
                video_writer.write(annotated)

                should_save_debug = processed_count % DEBUG_FRAME_INTERVAL == 0
                if (
                    detection is None
                    and processed_count % max(1, DEBUG_FRAME_INTERVAL // 3) == 0
                ):
                    should_save_debug = True

                if should_save_debug and saved_debug < DEBUG_MAX_IMAGES:
                    debug_image_path = DEBUG_DIR / f"frame_{frame_idx:08d}.jpg"
                    cv2.imwrite(str(debug_image_path), annotated)
                    saved_debug += 1

                processed_count += 1
                if processed_count % PROGRESS_EVERY_FRAMES == 0:
                    elapsed = max(time.perf_counter() - start_clock, 1e-6)
                    speed_fps = processed_count / elapsed
                    remaining_frames = max_to_process - processed_count
                    eta_seconds = remaining_frames / max(speed_fps, 1e-6)
                    print(
                        f"Processed {processed_count}/{max_to_process} frames "
                        f"(global frame {frame_idx}) | "
                        f"speed={speed_fps:.2f} fps | eta={eta_seconds/60:.1f} min",
                        flush=True,
                    )

                frame_idx += 1
    finally:
        cap.release()
        video_writer.release()

    total_elapsed = time.perf_counter() - start_clock
    avg_speed = processed_count / max(total_elapsed, 1e-6)
    print("Detectron2 experiment completed.", flush=True)
    print(f"Frames processed: {processed_count}", flush=True)
    print(f"Average speed: {avg_speed:.2f} fps", flush=True)
    print(f"Annotated video: {annotated_video_path}", flush=True)
    print(f"CSV coordinates: {csv_path}", flush=True)
    print(f"Debug images dir: {DEBUG_DIR}", flush=True)


if __name__ == "__main__":
    main()
