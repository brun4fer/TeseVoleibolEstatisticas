from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


TARGET_SIZE = (40, 40)
DEFAULT_SET_RATIO = 0.30
DEFAULT_BLOCK_SIZE = 21
DEFAULT_THRESHOLD_C = 10
DEFAULT_ROI_SEARCH = 18
DEFAULT_ROI_SIZE_SEARCH = 10
DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_NMS_IOU_THRESHOLD = 0.30
DEFAULT_X_GROUP_DISTANCE = 10


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

ALLOWED_DIGITS_BY_REGION = {
    "sets_a": tuple(range(6)),
    "points_a": tuple(range(10)),
    "sets_b": tuple(range(6)),
    "points_b": tuple(range(10)),
}

MAX_DIGITS_BY_REGION = {
    "sets_a": 1,
    "points_a": 2,
    "sets_b": 1,
    "points_b": 2,
}

MATCH_THRESHOLD_BY_REGION = {
    "sets_a": 0.48,
    "points_a": 0.42,
    "sets_b": 0.48,
    "points_b": 0.42,
}


@dataclass(frozen=True)
class MatchCandidate:
    digit: int
    score: float
    x: int
    y: int
    w: int
    h: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a volleyball scoreboard using template matching."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--templates-dir",
        default="digit_templates",
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
        default=DEFAULT_ROI_SEARCH,
        help="Pixel radius used to refine the selected ROI position.",
    )
    parser.add_argument(
        "--roi-size-search",
        type=int,
        default=DEFAULT_ROI_SIZE_SEARCH,
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
    """
    Backward-compatible single-digit preprocessing.
    Kept for callers that still expect a centered 40x40 glyph.
    """
    binary = preprocess_region_for_matching(
        image=image,
        invert=invert,
        block_size=block_size,
        threshold_c=threshold_c,
    )
    binary = extract_primary_component(binary)
    return normalize_binary(binary, target_size)


def preprocess_region_for_matching(
    image: np.ndarray,
    invert: bool,
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
    return _remove_small_components(binary)


def _remove_small_components(binary: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if component_count <= 1:
        return binary

    min_area = max(3, int(binary.size * 0.008))
    cleaned = np.zeros_like(binary)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned if np.count_nonzero(cleaned) > 0 else binary


def normalize_strip_for_matching(
    binary: np.ndarray,
    target_height: int,
    min_width: int,
) -> np.ndarray:
    points = cv2.findNonZero(binary)
    canvas_width = max(int(min_width), TARGET_SIZE[0])
    if points is None:
        return np.zeros((target_height, canvas_width), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points)
    strip = binary[y : y + h, x : x + w]
    if strip.size == 0:
        return np.zeros((target_height, canvas_width), dtype=np.uint8)

    scale = float(target_height) / float(max(h, 1))
    resized_w = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(strip, (resized_w, target_height), interpolation=interpolation)

    padded_width = max(canvas_width, resized_w + 8)
    canvas = np.zeros((target_height, padded_width), dtype=np.uint8)
    x_offset = max(0, (padded_width - resized_w) // 2)
    canvas[:, x_offset : x_offset + resized_w] = resized
    return canvas


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


def _bbox_iou(a: MatchCandidate, b: MatchCandidate) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = a.w * a.h
    area_b = b.w * b.h
    union = max(1, area_a + area_b - inter_area)
    return float(inter_area) / float(union)


def _nms_and_group_by_x(
    candidates: list[MatchCandidate],
    iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    x_group_distance: int = DEFAULT_X_GROUP_DISTANCE,
) -> list[MatchCandidate]:
    if not candidates:
        return []

    kept: list[MatchCandidate] = []
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    for candidate in ordered:
        center_x = candidate.x + (candidate.w / 2.0)
        should_keep = True
        for chosen in kept:
            chosen_center_x = chosen.x + (chosen.w / 2.0)
            if _bbox_iou(candidate, chosen) >= iou_threshold:
                should_keep = False
                break
            if abs(center_x - chosen_center_x) <= float(x_group_distance):
                should_keep = False
                break
        if should_keep:
            kept.append(candidate)
    return kept


def match_digits(
    processed: np.ndarray,
    templates: Dict[int, np.ndarray],
    allowed_digits: Tuple[int, ...] | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    x_group_distance: int = DEFAULT_X_GROUP_DISTANCE,
    max_digits: int = 2,
) -> tuple[int, float]:
    processed_std = float(np.std(processed))
    if processed_std < 1e-6:
        return 0, -1.0

    digit_pool = allowed_digits if allowed_digits is not None else tuple(sorted(templates.keys()))
    candidates: list[MatchCandidate] = []
    fallback_best: MatchCandidate | None = None

    for digit in digit_pool:
        template = templates[digit]
        template_std = float(np.std(template))
        if template_std < 1e-6:
            continue
        t_h, t_w = template.shape[:2]
        p_h, p_w = processed.shape[:2]
        if p_h < t_h or p_w < t_w:
            continue

        response = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
        _min_score, max_score, _min_loc, max_loc = cv2.minMaxLoc(response)
        current_best = MatchCandidate(
            digit=digit,
            score=float(max_score),
            x=int(max_loc[0]),
            y=int(max_loc[1]),
            w=int(t_w),
            h=int(t_h),
        )
        if fallback_best is None or current_best.score > fallback_best.score:
            fallback_best = current_best

        y_coords, x_coords = np.where(response >= float(match_threshold))
        for y, x in zip(y_coords.tolist(), x_coords.tolist()):
            score = float(response[y, x])
            candidates.append(
                MatchCandidate(
                    digit=int(digit),
                    score=score,
                    x=int(x),
                    y=int(y),
                    w=int(t_w),
                    h=int(t_h),
                )
            )

    if not candidates and fallback_best is not None:
        candidates = [fallback_best]
    if not candidates:
        return 0, -1.0

    filtered = _nms_and_group_by_x(
        candidates=candidates,
        iou_threshold=float(nms_iou_threshold),
        x_group_distance=int(x_group_distance),
    )
    if not filtered and fallback_best is not None:
        filtered = [fallback_best]
    if not filtered:
        return 0, -1.0

    if max_digits > 0 and len(filtered) > max_digits:
        filtered = sorted(filtered, key=lambda c: c.score, reverse=True)[:max_digits]
    filtered = sorted(filtered, key=lambda c: c.x)

    digits_as_text = "".join(str(candidate.digit) for candidate in filtered)
    value = int(digits_as_text) if digits_as_text else 0
    score = float(np.mean([candidate.score for candidate in filtered]))
    return value, score


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
        processed_binary = preprocess_region_for_matching(
            cropped_digit,
            spec.invert,
            block_size,
            threshold_c,
        )
        max_digits = int(MAX_DIGITS_BY_REGION.get(spec.name, 1))
        min_width = TARGET_SIZE[0] * max(1, max_digits)
        processed = normalize_strip_for_matching(
            processed_binary,
            target_height=TARGET_SIZE[1],
            min_width=min_width,
        )
        digit, score = match_digits(
            processed,
            templates,
            allowed_digits=ALLOWED_DIGITS_BY_REGION.get(spec.name),
            match_threshold=float(MATCH_THRESHOLD_BY_REGION.get(spec.name, DEFAULT_MATCH_THRESHOLD)),
            max_digits=max_digits,
        )
        results[spec.name] = digit
        debug_info[spec.name] = (cropped_digit, processed, score)
        total_score += score

    return results, debug_info, total_score


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
    initial_w = int(initial_roi[2])
    initial_h = int(initial_roi[3])
    search_radius = max(search_radius, int(round(max(initial_w, initial_h) * 0.20)))
    size_radius = max(size_radius, int(round(max(initial_w, initial_h) * 0.10)))

    def iter_steps(radius: int) -> Tuple[int, ...]:
        steps = []
        for step in (max(1, radius // 3), max(1, radius // 6), 1):
            if step not in steps:
                steps.append(step)
        return tuple(steps)

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

    def search_position_once(
        rect: tuple[int, int, int, int],
        radius: int,
        step: int,
    ) -> tuple[int, int, int, int]:
        base_x, base_y, base_w, base_h = rect
        best_rect = rect
        best_score = float("-inf")

        for delta_x in range(-radius, radius + 1, step):
            for delta_y in range(-radius, radius + 1, step):
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

    def search_position(
        rect: tuple[int, int, int, int],
        radius: int,
    ) -> tuple[int, int, int, int]:
        best_rect = rect
        current_radius = radius
        for step in iter_steps(radius):
            best_rect = search_position_once(best_rect, current_radius, step)
            current_radius = max(step, current_radius // 2)
        return best_rect

    def search_size_once(
        rect: tuple[int, int, int, int],
        radius: int,
        step: int,
    ) -> tuple[int, int, int, int]:
        base_x, base_y, base_w, base_h = rect
        center_x = base_x + (base_w / 2.0)
        center_y = base_y + (base_h / 2.0)
        best_rect = rect
        best_score = float("-inf")

        for delta_w in range(-radius, radius + 1, step):
            candidate_w = base_w + delta_w
            if candidate_w < 8:
                continue

            for delta_h in range(-radius, radius + 1, step):
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

    def search_size(
        rect: tuple[int, int, int, int],
        radius: int,
    ) -> tuple[int, int, int, int]:
        best_rect = rect
        current_radius = radius
        for step in iter_steps(radius):
            best_rect = search_size_once(best_rect, current_radius, step)
            current_radius = max(step, current_radius // 2)
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


class ScoreboardReader:
    def __init__(
        self,
        templates_dir: str | Path,
        set_ratio: float = DEFAULT_SET_RATIO,
        block_size: int = DEFAULT_BLOCK_SIZE,
        threshold_c: int = DEFAULT_THRESHOLD_C,
        roi_search: int = DEFAULT_ROI_SEARCH,
        roi_size_search: int = DEFAULT_ROI_SIZE_SEARCH,
    ):
        if not 0.0 < set_ratio < 1.0:
            raise ValueError("set_ratio must be between 0 and 1.")
        if roi_search < 0:
            raise ValueError("roi_search must be 0 or greater.")
        if roi_size_search < 0:
            raise ValueError("roi_size_search must be 0 or greater.")

        resolved_templates_dir = Path(templates_dir)
        if not resolved_templates_dir.is_absolute():
            resolved_templates_dir = Path(__file__).resolve().parent / resolved_templates_dir
        if not resolved_templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {resolved_templates_dir}")

        self.templates_dir = resolved_templates_dir
        self.set_ratio = float(set_ratio)
        self.block_size = ensure_odd(block_size)
        self.threshold_c = int(threshold_c)
        self.roi_search = int(roi_search)
        self.roi_size_search = int(roi_size_search)
        self.templates = load_templates(self.templates_dir, TARGET_SIZE)
        self.roi: Optional[tuple[int, int, int, int]] = None
        self.last_results: Optional[Dict[str, int]] = None
        self.last_debug_info: Dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self.last_total_score: float = 0.0

    def set_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        if self.roi is not None:
            return self.roi

        selection = cv2.selectROI(
            "Select Scoreboard ROI",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Select Scoreboard ROI")

        x, y, w, h = [int(v) for v in selection]
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI selection was cancelled or empty.")

        self.roi = refine_scoreboard_roi(
            frame,
            (x, y, w, h),
            self.templates,
            self.set_ratio,
            self.block_size,
            self.threshold_c,
            self.roi_search,
            self.roi_size_search,
        )
        return self.roi

    def read(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        if self.roi is None:
            raise RuntimeError("Scoreboard ROI has not been selected yet.")

        x, y, w, h = self.roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            raise RuntimeError("Selected ROI is outside the frame bounds.")

        results, debug_info, total_score = read_scoreboard_roi(
            roi,
            self.templates,
            self.set_ratio,
            self.block_size,
            self.threshold_c,
        )
        self.last_results = results
        self.last_debug_info = debug_info
        self.last_total_score = total_score
        return (
            results["sets_a"],
            results["points_a"],
            results["sets_b"],
            results["points_b"],
        )


def main() -> None:
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = ScoreboardReader(
        templates_dir=args.templates_dir,
        set_ratio=args.set_ratio,
        block_size=args.block_size,
        threshold_c=args.threshold_c,
        roi_search=args.roi_search,
        roi_size_search=args.roi_size_search,
    )

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

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read the frame used for ROI selection.")

        roi_x, roi_y, roi_w, roi_h = reader.set_roi(first_frame)
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

            sets_a, points_a, sets_b, points_b = reader.read(frame)
            results = reader.last_results
            debug_info = reader.last_debug_info

            if results is None:
                raise RuntimeError("Reader returned no scoreboard data.")

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

            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            roi_debug = draw_roi_overlay(roi, reader.set_ratio, results)
            cv2.imshow("ROI", roi_debug)

            print(f"SETS: {sets_a}-{sets_b}", flush=True)
            print(f"POINTS: {points_a}-{points_b}", flush=True)
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
