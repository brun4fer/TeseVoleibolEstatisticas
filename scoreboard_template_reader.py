from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


TARGET_SIZE = (40, 40)
DEFAULT_SET_RATIO = 0.30
DEFAULT_BLOCK_SIZE = 21
DEFAULT_THRESHOLD_C = 10


@dataclass(frozen=True)
class RegionSpec:
    name: str
    invert: bool
    margins: Tuple[float, float, float, float]


REGION_SPECS = (
    RegionSpec("sets_a", invert=False, margins=(0.05, 0.05, 0.00, 0.08)),
    RegionSpec("points_a", invert=True, margins=(0.14, 0.04, 0.00, 0.08)),
    RegionSpec("sets_b", invert=False, margins=(0.05, 0.05, 0.08, 0.00)),
    RegionSpec("points_b", invert=True, margins=(0.14, 0.04, 0.08, 0.00)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a volleyball scoreboard using template matching."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--templates-dir",
        default="templates",
        help="Directory containing digit templates 0.png to 9.png.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start timestamp in HH:MM:SS, MM:SS, or seconds.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End timestamp in HH:MM:SS, MM:SS, or seconds.",
    )
    parser.add_argument(
        "--set-ratio",
        type=float,
        default=DEFAULT_SET_RATIO,
        help="Fraction of the ROI width reserved for the left sets column.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Odd block size used by adaptive thresholding.",
    )
    parser.add_argument(
        "--threshold-c",
        type=int,
        default=DEFAULT_THRESHOLD_C,
        help="Constant subtracted by adaptive thresholding.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame. Default: 1.",
    )
    parser.add_argument(
        "--roi-search",
        type=int,
        default=8,
        help="Pixel radius used to refine the selected ROI position.",
    )
    parser.add_argument(
        "--roi-size-search",
        type=int,
        default=4,
        help="Pixel radius used to refine the selected ROI size.",
    )
    return parser.parse_args()


def parse_timestamp(value: str | None) -> float | None:
    if value is None:
        return None

    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timestamp: {value}")

    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc

    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours = 0.0
        minutes, seconds = numbers
    else:
        hours = 0.0
        minutes = 0.0
        seconds = numbers[0]

    if minutes < 0 or seconds < 0 or hours < 0:
        raise ValueError(f"Timestamp must be non-negative: {value}")
    if len(numbers) > 1 and minutes >= 60:
        raise ValueError(f"Minutes must be below 60: {value}")
    if len(numbers) > 1 and seconds >= 60:
        raise ValueError(f"Seconds must be below 60: {value}")

    return (hours * 3600.0) + (minutes * 60.0) + seconds


def seconds_to_frame(seconds: float | None, fps: float, total_frames: int, default: int) -> int:
    if seconds is None:
        return default
    frame_idx = int(seconds * fps)
    return max(0, min(frame_idx, max(total_frames - 1, 0)))


def ensure_odd(value: int) -> int:
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def normalize_binary(binary: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    points = cv2.findNonZero(binary)
    if points is None:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points)
    digit = binary[y : y + h, x : x + w]

    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    max_w = max(target_size[0] - 6, 1)
    max_h = max(target_size[1] - 6, 1)
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))

    digit_resized = cv2.resize(
        digit,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )

    x_offset = (target_size[0] - resized_w) // 2
    y_offset = (target_size[1] - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = digit_resized
    return canvas


def crop_inner_region(image: np.ndarray, margins: Tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    left_ratio, right_ratio, top_ratio, bottom_ratio = margins

    left = int(round(width * left_ratio))
    right = width - int(round(width * right_ratio))
    top = int(round(height * top_ratio))
    bottom = height - int(round(height * bottom_ratio))

    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))

    return image[top:bottom, left:right]


def extract_primary_component(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape[:2]
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    best_label = None
    best_score = float("-inf")
    min_area = max(4, int(binary.size * 0.01))

    for label in range(1, component_count):
        x, y, box_w, box_h, area = stats[label]
        if area < min_area:
            continue

        touches_left = x <= 0
        touches_top = y <= 0
        touches_right = x + box_w >= width
        touches_bottom = y + box_h >= height
        aspect_h = box_h / max(box_w, 1)
        aspect_w = box_w / max(box_h, 1)

        if (touches_left or touches_right) and aspect_h >= 4.0 and area <= binary.size * 0.30:
            continue
        if (touches_top or touches_bottom) and aspect_w >= 4.0 and area <= binary.size * 0.30:
            continue

        center_x, center_y = centroids[label]
        center_score = 1.0 - (
            (
                abs(center_x - (width / 2.0)) / ((width / 2.0) + 1e-6)
                + abs(center_y - (height / 2.0)) / ((height / 2.0) + 1e-6)
            )
            / 2.0
        )
        score = area * (0.7 + (0.3 * center_score))
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return binary

    selected = np.zeros_like(binary)
    selected[labels == best_label] = 255
    return selected


def preprocess_digit_image(
    image: np.ndarray,
    invert: bool,
    target_size: Tuple[int, int],
    block_size: int,
    threshold_c: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        threshold_c,
    )

    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = extract_primary_component(binary)
    return normalize_binary(binary, target_size)


def preprocess_template_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None
    elif image.ndim == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = image[:, :, 3]
    else:
        gray = image.copy()
        alpha = None

    if alpha is not None and np.any(alpha > 0):
        _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    return normalize_binary(binary, target_size)


def load_templates(
    templates_dir: Path,
    target_size: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for digit in range(10):
        template_path = templates_dir / f"{digit}.png"
        template_image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template_image is None:
            raise FileNotFoundError(f"Missing template: {template_path}")
        templates[digit] = preprocess_template_image(
            template_image,
            target_size,
        )
    return templates


def split_scoreboard_roi(roi: np.ndarray, set_ratio: float) -> Dict[str, np.ndarray]:
    height, width = roi.shape[:2]
    middle_y = height // 2
    split_x = int(round(width * set_ratio))
    split_x = max(1, min(split_x, width - 1))

    return {
        "sets_a": roi[:middle_y, :split_x],
        "points_a": roi[:middle_y, split_x:],
        "sets_b": roi[middle_y:, :split_x],
        "points_b": roi[middle_y:, split_x:],
    }


def read_scoreboard_roi(
    roi: np.ndarray,
    templates: Dict[int, np.ndarray],
    set_ratio: float,
    block_size: int,
    threshold_c: int,
) -> tuple[Dict[str, int], Dict[str, tuple[np.ndarray, np.ndarray, float]], float]:
    region_images = split_scoreboard_roi(roi, set_ratio)
    results: Dict[str, int] = {}
    debug_info: Dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    total_score = 0.0

    for spec in REGION_SPECS:
        crop = region_images[spec.name]
        cropped_digit = crop_inner_region(crop, spec.margins)
        processed = preprocess_digit_image(
            cropped_digit,
            spec.invert,
            TARGET_SIZE,
            block_size,
            threshold_c,
        )
        digit, score = match_digit(processed, templates)
        results[spec.name] = digit
        debug_info[spec.name] = (cropped_digit, processed, score)
        total_score += score

    return results, debug_info, total_score


def match_digit(processed: np.ndarray, templates: Dict[int, np.ndarray]) -> tuple[int, float]:
    processed_std = float(np.std(processed))
    if processed_std < 1e-6:
        return 0, -1.0

    best_digit = 0
    best_score = float("-inf")

    for digit, template in templates.items():
        template_std = float(np.std(template))
        if template_std < 1e-6:
            continue

        score = float(
            cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        )
        if score > best_score:
            best_digit = digit
            best_score = score

    if best_score == float("-inf"):
        return 0, -1.0

    return best_digit, best_score


def refine_scoreboard_roi(
    frame: np.ndarray,
    initial_roi: tuple[int, int, int, int],
    templates: Dict[int, np.ndarray],
    set_ratio: float,
    block_size: int,
    threshold_c: int,
    search_radius: int,
    size_radius: int,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    score_cache: Dict[tuple[int, int, int, int], float | None] = {}

    def score_rect(rect: tuple[int, int, int, int]) -> float | None:
        if rect in score_cache:
            return score_cache[rect]

        x, y, w, h = rect
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            score_cache[rect] = None
            return None
        if w < 8 or h < 8:
            score_cache[rect] = None
            return None

        roi = frame[y : y + h, x : x + w]
        _results, _debug_info, total_score = read_scoreboard_roi(
            roi,
            templates,
            set_ratio,
            block_size,
            threshold_c,
        )
        score_cache[rect] = total_score
        return total_score

    def search_position(
        rect: tuple[int, int, int, int],
        radius: int,
    ) -> tuple[int, int, int, int]:
        base_x, base_y, base_w, base_h = rect
        best_rect = rect
        best_score = float("-inf")

        for delta_x in range(-radius, radius + 1):
            for delta_y in range(-radius, radius + 1):
                candidate_rect = (
                    base_x + delta_x,
                    base_y + delta_y,
                    base_w,
                    base_h,
                )
                total_score = score_rect(candidate_rect)
                if total_score is None:
                    continue

                candidate_score = total_score - (0.0025 * (abs(delta_x) + abs(delta_y)))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_rect = candidate_rect

        return best_rect

    def search_size(
        rect: tuple[int, int, int, int],
        radius: int,
    ) -> tuple[int, int, int, int]:
        base_x, base_y, base_w, base_h = rect
        center_x = base_x + (base_w / 2.0)
        center_y = base_y + (base_h / 2.0)
        best_rect = rect
        best_score = float("-inf")

        for delta_w in range(-radius, radius + 1):
            candidate_w = base_w + delta_w
            if candidate_w < 8:
                continue

            for delta_h in range(-radius, radius + 1):
                candidate_h = base_h + delta_h
                if candidate_h < 8:
                    continue

                candidate_x = int(round(center_x - (candidate_w / 2.0)))
                candidate_y = int(round(center_y - (candidate_h / 2.0)))
                candidate_rect = (
                    candidate_x,
                    candidate_y,
                    candidate_w,
                    candidate_h,
                )
                total_score = score_rect(candidate_rect)
                if total_score is None:
                    continue

                candidate_score = total_score - (0.0025 * (abs(delta_w) + abs(delta_h)))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_rect = candidate_rect

        return best_rect

    candidates = [initial_roi]

    position_first = initial_roi
    if search_radius > 0:
        position_first = search_position(position_first, search_radius)
    if size_radius > 0:
        position_first = search_size(position_first, size_radius)
    if search_radius > 0:
        position_first = search_position(position_first, max(2, search_radius // 2))
    candidates.append(position_first)

    size_first = initial_roi
    if size_radius > 0:
        size_first = search_size(size_first, size_radius)
    if search_radius > 0:
        size_first = search_position(size_first, search_radius)
    if size_radius > 0:
        size_first = search_size(size_first, size_radius)
    if search_radius > 0:
        size_first = search_position(size_first, max(2, search_radius // 2))
    candidates.append(size_first)

    best_roi = initial_roi
    best_score = float("-inf")
    for candidate in candidates:
        total_score = score_rect(candidate)
        if total_score is None:
            continue
        if total_score > best_score:
            best_score = total_score
            best_roi = candidate

    return best_roi


def build_debug_panel(label: str, crop: np.ndarray, processed: np.ndarray, digit: int, score: float) -> np.ndarray:
    crop_view = cv2.resize(crop, (160, 120), interpolation=cv2.INTER_CUBIC)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed_view = cv2.resize(processed_bgr, (160, 120), interpolation=cv2.INTER_NEAREST)
    panel = cv2.hconcat([crop_view, processed_view])
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"digit={digit} score={score:.3f}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_roi_overlay(roi: np.ndarray, set_ratio: float, results: Dict[str, int]) -> np.ndarray:
    overlay = roi.copy()
    height, width = roi.shape[:2]
    middle_y = height // 2
    split_x = int(round(width * set_ratio))
    split_x = max(1, min(split_x, width - 1))

    cv2.line(overlay, (split_x, 0), (split_x, height), (0, 255, 0), 1)
    cv2.line(overlay, (0, middle_y), (width, middle_y), (0, 255, 0), 1)

    cv2.putText(
        overlay,
        f"SETS: {results['sets_a']}-{results['sets_b']}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"POINTS: {results['points_a']}-{results['points_b']}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def select_scoreboard_roi(cap: cv2.VideoCapture, start_frame: int) -> tuple[int, int, int, int]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the frame used for ROI selection.")

    selection = cv2.selectROI("Select Scoreboard ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Scoreboard ROI")

    x, y, w, h = [int(v) for v in selection]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection was cancelled or empty.")
    return x, y, w, h


def main() -> None:
    args = parse_args()

    if not 0.0 < args.set_ratio < 1.0:
        raise ValueError("--set-ratio must be between 0 and 1.")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    if args.roi_search < 0:
        raise ValueError("--roi-search must be 0 or greater.")
    if args.roi_size_search < 0:
        raise ValueError("--roi-size-search must be 0 or greater.")

    block_size = ensure_odd(args.block_size)
    threshold_c = args.threshold_c

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_absolute():
        templates_dir = Path(__file__).resolve().parent / templates_dir
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    templates = load_templates(templates_dir, TARGET_SIZE)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError("Could not read video FPS.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_seconds = parse_timestamp(args.start)
        end_seconds = parse_timestamp(args.end)

        start_frame = seconds_to_frame(start_seconds, fps, total_frames, 0)
        end_frame = seconds_to_frame(
            end_seconds,
            fps,
            total_frames,
            max(total_frames - 1, 0),
        )

        if end_frame < start_frame:
            raise ValueError("--end must be after --start.")

        roi_x, roi_y, roi_w, roi_h = select_scoreboard_roi(cap, start_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, refine_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read the frame used for ROI refinement.")

        refined_roi = refine_scoreboard_roi(
            refine_frame,
            (roi_x, roi_y, roi_w, roi_h),
            templates,
            args.set_ratio,
            block_size,
            threshold_c,
            args.roi_search,
            args.roi_size_search,
        )
        roi_x, roi_y, roi_w, roi_h = refined_roi
        print(
            f"Refined ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}",
            flush=True,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_index = start_frame
        while frame_index <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_index - start_frame) % args.frame_step != 0:
                frame_index += 1
                continue

            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if roi.size == 0:
                raise RuntimeError("Selected ROI is outside the frame bounds.")

            results, debug_info, _total_score = read_scoreboard_roi(
                roi,
                templates,
                args.set_ratio,
                block_size,
                threshold_c,
            )

            for spec in REGION_SPECS:
                cropped_digit, processed, score = debug_info[spec.name]
                debug_panel = build_debug_panel(
                    spec.name,
                    cropped_digit,
                    processed,
                    results[spec.name],
                    score,
                )
                cv2.imshow(f"DEBUG_{spec.name}", debug_panel)

            roi_debug = draw_roi_overlay(roi, args.set_ratio, results)
            cv2.imshow("ROI", roi_debug)

            print(f"SETS: {results['sets_a']}-{results['sets_b']}", flush=True)
            print(f"POINTS: {results['points_a']}-{results['points_b']}", flush=True)
            print(flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
