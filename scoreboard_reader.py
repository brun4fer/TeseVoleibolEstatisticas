"""
Standalone volleyball scoreboard reader using template matching.

Usage:
    python scoreboard_reader.py --video "video.mp4" --start 00:29:02 --end 00:46:17
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


DIGIT_NAMES = ("sets_top", "points_top", "sets_bottom", "points_bottom")
CANVAS_W = 40
CANVAS_H = 60


def parse_hhmmss_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format '{ts}'. Expected HH:MM:SS")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return float(hh * 3600 + mm * 60 + ss)


def split_scoreboard_regions(roi_shape: Tuple[int, int, int]) -> Dict[str, Tuple[int, int, int, int]]:
    h, w = roi_shape[:2]
    h2 = h // 2
    w2 = w // 2
    return {
        "sets_top": (0, 0, w2, h2),
        "points_top": (w2, 0, w - w2, h2),
        "sets_bottom": (0, h2, w2, h - h2),
        "points_bottom": (w2, h2, w - w2, h - h2),
    }


def preprocess_cell(cell) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)
    th = cv2.medianBlur(th, 3)
    return th


def crop_and_normalize_digit(th: np.ndarray, invert: bool = False) -> np.ndarray:
    coords = cv2.findNonZero(th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = th[y : y + h, x : x + w]
    else:
        digit = th

    if digit is None or digit.size == 0:
        return np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    h, w = digit.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    if invert:
        digit = cv2.bitwise_not(digit)

    # Keep aspect ratio, then center on fixed canvas.
    scale = min(CANVAS_W / float(w), CANVAS_H / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit_resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_CUBIC,
    )

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    x_offset = (CANVAS_W - new_w) // 2
    y_offset = (CANVAS_H - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = digit_resized
    return canvas


def load_templates(template_dir: Path) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for d in range(10):
        path = template_dir / f"{d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing template file: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read template image: {path}")
        th = preprocess_cell(img)
        templates[d] = crop_and_normalize_digit(th, invert=False)
    return templates


def match_digit(digit: np.ndarray, templates: Dict[int, np.ndarray]) -> Tuple[Optional[int], float]:
    best_digit: Optional[int] = None
    best_score = -1.0
    for d, tmpl in templates.items():
        digit = cv2.GaussianBlur(digit,(3,3),0)
        score = float(cv2.matchTemplate(digit, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best_score = score
            best_digit = d
    return best_digit, best_score


class Stabilizer:
    def __init__(self, history_len: int = 10) -> None:
        self.buffers: Dict[str, Deque[int]] = {k: deque(maxlen=history_len) for k in DIGIT_NAMES}

    def update(self, detections: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
        out: Dict[str, Optional[int]] = {}
        for k in DIGIT_NAMES:
            v = detections.get(k)
            if v is not None:
                self.buffers[k].append(int(v))
            if self.buffers[k]:
                out[k] = Counter(self.buffers[k]).most_common(1)[0][0]
            else:
                out[k] = None
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust volleyball scoreboard reader with template matching.")
    p.add_argument("--video", type=str, required=True, help="Input video path.")
    p.add_argument("--start", type=str, default="00:00:00", help="Start timestamp (HH:MM:SS).")
    p.add_argument("--end", type=str, default=None, help="End timestamp (HH:MM:SS).")
    p.add_argument("--templates", type=str, default="templates", help="Template folder with 0.png ... 9.png")
    p.add_argument("--history", type=int, default=10, help="Stabilization history length.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    templates = load_templates(Path(args.templates))
    stabilizer = Stabilizer(history_len=max(1, int(args.history)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS from video.")

    start_seconds = parse_hhmmss_to_seconds(args.start)
    end_seconds = parse_hhmmss_to_seconds(args.end) if args.end else None
    if end_seconds is not None and end_seconds < start_seconds:
        cap.release()
        raise RuntimeError(f"--end ({args.end}) must be >= --start ({args.start}).")

    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps) if end_seconds is not None else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read video frame")

    cv2.namedWindow("Select Scoreboard ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    roi_box = cv2.selectROI("Select Scoreboard ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Scoreboard ROI")
    x, y, w, h = map(int, roi_box)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("ROI selection cancelled.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    last_printed: Optional[Tuple[int, int, int, int]] = None

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        view = frame.copy()
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            frame_idx += 1
            continue

        regions = split_scoreboard_regions(roi.shape)
        detections: Dict[str, Optional[int]] = {}

        for name, (rx, ry, rw, rh) in regions.items():
            cell = roi[ry : ry + rh, rx : rx + rw]
            th = preprocess_cell(cell)
            digit = crop_and_normalize_digit(th, invert=True)
            best_digit, best_score = match_digit(digit, templates)
            detections[name] = best_digit

            print(f"{name} -> digit {best_digit if best_digit is not None else '?'} score {best_score:.2f}")
            cv2.imshow(f"digit_{name}", digit)

            gx1, gy1 = x + rx, y + ry
            gx2, gy2 = gx1 + rw, gy1 + rh
            cv2.rectangle(view, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
            cv2.putText(
                view,
                name,
                (gx1, max(15, gy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        stable = stabilizer.update(detections)
        if all(stable[k] is not None for k in DIGIT_NAMES):
            sets_x = int(stable["sets_top"])
            sets_y = int(stable["sets_bottom"])
            pts_x = int(stable["points_top"])
            pts_y = int(stable["points_bottom"])
            score_tuple = (sets_x, sets_y, pts_x, pts_y)
            if score_tuple != last_printed:
                print(f"SETS: {sets_x}-{sets_y}")
                print(f"POINTS: {pts_x}-{pts_y}")
                last_printed = score_tuple

        cv2.imshow("Scoreboard Reader", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
