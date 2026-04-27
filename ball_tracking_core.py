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
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

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
    # Candidatos que permanecem na mesma posição (±grid_px) por este nº de frames
    # consecutivos são considerados falsos positivos estáticos e removidos.
    static_fp_max_frames: int = 30
    static_fp_grid_px: int = 30

    # --- Fix 1: Threshold adaptativo de confiança YOLO ---
    # O threshold sobe quando o tracker está estável (poucos falsos positivos)
    # e desce em períodos sem deteção (motion blur, oclusão, rede).
    adaptive_conf_high: float = 0.35
    adaptive_conf_mid: float = 0.20
    adaptive_conf_low: float = 0.15
    adaptive_conf_mid_after_misses: int = 5
    adaptive_conf_low_after_misses: int = 15

    # --- Fix 2: Zona de busca expandida após gap ---
    # Após N frames sem deteção, expandimos as gates de step/prediction-error para
    # evitar rejeitar a bola como "speed outlier" quando reaparece longe.
    gap_search_expand_2x_after: int = 5
    gap_search_expand_4x_after: int = 15

    # --- Fix 3: Blacklist permanente de posições estáticas (FP persistente) ---
    # Janela de N últimas deteções aceites; se ≥ min_hits estiverem dentro de
    # radius_px da nova posição, classifica-se como FP estático persistente.
    # Qualquer candidato futuro a < reject_radius_px é descartado para sempre.
    position_blacklist_window: int = 60
    position_blacklist_radius_px: float = 20.0
    position_blacklist_min_hits: int = 40
    position_blacklist_reject_radius_px: float = 25.0

    # --- Fix 5: Tolerância especial na zona da rede ---
    # Na zona da rede a bola muda de direção abruptamente em blocos/spikes.
    # Relaxamos as gates de step/prediction-error nesta região.
    net_zone_relax_factor: float = 1.5

    # --- Sistema de scoring contínuo (substitui rejeição binária) ----------
    # Cada candidato recebe um score em [0, 1+] composto por 5 componentes.
    # A soma dos 5 pesos = 1.0 (ou 1.2 com boost na zona da rede).
    score_weight_yolo: float = 0.25       # confiança do YOLO
    score_weight_speed: float = 0.25      # consistência de velocidade
    score_weight_motion: float = 0.20     # foreground/MOG2
    score_weight_direction: float = 0.15  # consistência direcional (Kalman)
    score_weight_static: float = 0.15     # penalidade gradual por staticidade

    # Thresholds de decisão.
    score_accept_threshold: float = 0.55          # >= : aceitar sempre
    score_recovery_threshold: float = 0.35        # entre os dois: só em recovery
    score_recovery_gap_frames: int = 10           # gap mínimo p/ recovery threshold
    # Recovery total (zerar peso de speed e redistribuir): a partir deste gap.
    score_full_recovery_gap_frames: int = 15

    # Boost na zona da rede (signed_distance_to_net < score_net_zone_radius_m).
    score_net_zone_boost: float = 1.20
    score_net_zone_radius_m: float = 1.5

    # Janela e raio para a penalidade gradual de staticidade.
    # Raio pequeno (~8 px) para distinguir um FP fixo (que repete o MESMO
    # pixel) de uma bola num rally lento (que varre uma área de ±25 px sem
    # estar parada). Antes (25 px) penalizava bolas legítimas e perdia
    # frames críticos no momento da reversão.
    score_static_window: int = 60
    score_static_radius_px: float = 8.0
    score_static_max_hits: int = 30   # >= esta contagem ⇒ penalidade = 1.0

    # Conversão fg_ratio/fg_pixels → motion_score em [0, 1].
    score_motion_ratio_scale: float = 5.0
    score_motion_pixels_scale: float = 50.0


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
    context_penalty = float((candidate.get("game_context") or {}).get("penalty", 0.0))
    final_score = step_cost + prediction_cost + low_foreground_penalty + context_penalty - confidence_bonus - foreground_bonus

    candidate["step_distance"] = step_distance
    candidate["prediction_error"] = prediction_error
    candidate["context_penalty"] = context_penalty
    candidate["final_score"] = final_score

    return final_score


def select_ball_candidate(
    candidates: List[dict],
    previous_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
    candidate_evaluator: Optional[Callable[[Dict], Dict]] = None,
    search_multiplier: float = 1.0,
) -> Tuple[Optional[dict], str, Optional[dict]]:
    # search_multiplier expande as gates de step/prediction-error:
    #   1.0 → comportamento normal
    #   >1.0 → tolerância maior (Fix 2 após gap, Fix 5 na zona da rede)
    if not candidates:
        return None, "missed_frame_no_valid_candidate", None
    effective_max_step = float(cfg.max_step_pixels) * float(search_multiplier)
    effective_max_pred_err = float(cfg.max_prediction_error_pixels) * float(search_multiplier)

    candidate_pool: List[dict] = []
    best_context_rejected: Optional[dict] = None
    for candidate in candidates:
        if candidate_evaluator is not None:
            context = candidate_evaluator(candidate)
            candidate["game_context"] = dict(context or {})
            if bool((context or {}).get("reject", False)):
                if best_context_rejected is None or float(candidate.get("confidence", 0.0)) > float(best_context_rejected.get("confidence", 0.0)):
                    best_context_rejected = dict(candidate)
                continue
        candidate_pool.append(candidate)

    if not candidate_pool:
        return None, "rejected_by_game_context", best_context_rejected

    if not cfg.use_motion_gating:
        return (
            min(
                candidate_pool,
                key=lambda candidate: score_ball_candidate(candidate, None, None, cfg),
            ),
            "selected_by_confidence",
            best_context_rejected,
        )

    if previous_center is None:
        return (
            min(
                candidate_pool,
                key=lambda candidate: score_ball_candidate(candidate, None, None, cfg),
            ),
            "selected_by_confidence",
            best_context_rejected,
        )

    if older_center is None:
        gated_candidates = [
            candidate
            for candidate in candidate_pool
            if (pixel_distance(previous_center, candidate["center"]) or 0.0) <= effective_max_step
        ]
        if not gated_candidates:
            return None, "rejected_step_distance", best_context_rejected
        return (
            min(
                gated_candidates,
                key=lambda candidate: score_ball_candidate(candidate, previous_center, None, cfg),
            ),
            "selected_by_motion_gate",
            best_context_rejected,
        )

    predicted_center = predict_next_center(previous_center, older_center)
    gated_candidates: List[dict] = []
    rejected_step_distance = False
    rejected_prediction_error = False

    for candidate in candidate_pool:
        step_distance = pixel_distance(previous_center, candidate["center"]) or 0.0
        if step_distance > effective_max_step:
            rejected_step_distance = True
            print(
                f"[STEP-REJECT] Candidato rejeitado: {step_distance:.0f}px "
                f"> max {effective_max_step:.0f}px (mult={search_multiplier:.2f}) | "
                f"{previous_center} → {candidate['center']}"
            )
            continue

        prediction_error = pixel_distance(predicted_center, candidate["center"]) or 0.0
        if prediction_error > effective_max_pred_err:
            rejected_prediction_error = True
            continue

        gated_candidates.append(candidate)

    if not gated_candidates:
        if rejected_prediction_error:
            return None, "rejected_prediction_error", best_context_rejected
        if rejected_step_distance:
            return None, "rejected_step_distance", best_context_rejected
        return None, "missed_frame_no_valid_candidate", best_context_rejected

    return (
        min(
            gated_candidates,
            key=lambda candidate: score_ball_candidate(candidate, previous_center, predicted_center, cfg),
        ),
        "selected_by_motion_gate",
        best_context_rejected,
    )


def calc_static_penalty(
    candidate: Dict,
    accepted_position_history: Deque[Tuple[float, float]],
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
) -> float:
    """Penalidade gradual em [0, 1] por staticidade da posição.

    Substitui a blacklist binária: quanto mais vezes a posição apareceu na
    janela recente, mais penalidade. Saturamos a 1.0 ao atingir
    score_static_max_hits ocorrências.
    """
    if not accepted_position_history:
        return 0.0
    cx, cy = float(candidate["center"][0]), float(candidate["center"][1])
    radius = float(cfg.score_static_radius_px)
    window = int(cfg.score_static_window)
    history_slice = list(accepted_position_history)[-window:]
    nearby = sum(
        1
        for (px, py) in history_slice
        if abs(px - cx) < radius and abs(py - cy) < radius
    )
    max_hits = max(1, int(cfg.score_static_max_hits))
    return min(1.0, nearby / float(max_hits))


def score_candidate(
    candidate: Dict,
    *,
    last_accepted_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
    accepted_position_history: Deque[Tuple[float, float]],
    gap_frames: int,
    fps: float,
    pixels_per_meter: float,
    max_ball_speed_ms: float,
    foreground_filter_active: bool,
    in_net_zone: bool,
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
    pixel_to_court_m: Optional[Callable[[Tuple[int, int]], Tuple[float, float]]] = None,
) -> Tuple[float, Dict[str, float], bool]:
    """Calcula o score contínuo [0, ~1.2] de um candidato.

    Retorna (score, breakdown_dict, recovery_mode_used).

    Componentes (pesos por defeito):
      - yolo_conf  (0.25) → confiança directa do YOLO
      - speed      (0.25) → 1 - speed/max_speed (degrada gradualmente)
      - motion     (0.20) → activação de foreground (MOG2)
      - direction  (0.15) → 1 - prediction_error/max_pred_err
      - static     (0.15) → 1 - penalidade gradual de staticidade

    Em recovery total (gap >= score_full_recovery_gap_frames), o peso de
    speed é zerado e redistribuído metade-metade entre yolo_conf e motion.
    Na zona da rede, o score final é multiplicado por score_net_zone_boost.
    """
    weights: Dict[str, float] = {}

    full_recovery = int(gap_frames) >= int(cfg.score_full_recovery_gap_frames)

    w_yolo = float(cfg.score_weight_yolo)
    w_speed = float(cfg.score_weight_speed)
    w_motion = float(cfg.score_weight_motion)
    w_direction = float(cfg.score_weight_direction)
    w_static = float(cfg.score_weight_static)

    if full_recovery:
        # Speed deixa de pesar; redistribuímos para yolo + motion.
        half = w_speed / 2.0
        w_yolo += half
        w_motion += half
        w_speed = 0.0

    # 1. Confiança YOLO ----------------------------------------------------
    conf = float(candidate.get("confidence", 0.0))
    weights["yolo_conf"] = conf * w_yolo

    # 2. Consistência de velocidade ---------------------------------------
    # Usa a HOMOGRAFIA quando disponível (pixel_to_court_m) → distância em
    # metros reais, independente da perspectiva. Longe da câmara, 1 px ≈ muitos
    # metros; perto, 1 px ≈ poucos cm. Sem o callback, fallback para o
    # pixels_per_meter fixo (errado por perspectiva, mas previsível).
    speed_score = 1.0  # neutro quando ainda não há referência
    speed_value_mps: Optional[float] = None
    if w_speed > 0.0 and last_accepted_center is not None and fps > 0:
        if pixel_to_court_m is not None:
            try:
                last_xy = pixel_to_court_m(last_accepted_center)
                curr_xy = pixel_to_court_m(candidate["center"])
                court_dist_m = math.hypot(curr_xy[0] - last_xy[0], curr_xy[1] - last_xy[1])
                # Tempo entre observações = frames decorridos (gap+1) / fps.
                dt_s = max((max(0, int(gap_frames)) + 1) / float(fps), 1.0 / 240.0)
                speed_value_mps = court_dist_m / dt_s
            except Exception:
                speed_value_mps = None
        if speed_value_mps is None and pixels_per_meter > 0:
            step_px = pixel_distance(last_accepted_center, candidate["center"]) or 0.0
            speed_value_mps = (step_px * float(fps)) / float(pixels_per_meter)
        if speed_value_mps is not None:
            # Margem cresce 10% por frame de gap (recuperar bola distante).
            max_speed = float(max_ball_speed_ms) * (1.0 + max(0, int(gap_frames)) * 0.1)
            if max_speed > 0:
                speed_score = max(0.0, 1.0 - (speed_value_mps / max_speed))
    weights["speed"] = speed_score * w_speed

    # 3. Movimento (foreground MOG2) --------------------------------------
    if not foreground_filter_active:
        # Sem máscara → componente neutra para não penalizar injustamente.
        motion_score = 1.0
    else:
        fg_ratio = float(candidate.get("fg_active_ratio", 0.0))
        fg_pixels = int(candidate.get("fg_active_pixels", 0))
        motion_score = min(
            1.0,
            fg_ratio * float(cfg.score_motion_ratio_scale)
            + fg_pixels / max(float(cfg.score_motion_pixels_scale), 1e-6),
        )
    weights["motion"] = motion_score * w_motion

    # 4. Consistência direcional (Kalman/predição linear) -----------------
    direction_score = 1.0
    pred_err_value: Optional[float] = None
    if last_accepted_center is not None and older_center is not None:
        predicted = predict_next_center(last_accepted_center, older_center)
        if predicted is not None:
            pred_err_value = pixel_distance(predicted, candidate["center"]) or 0.0
            max_err = float(cfg.max_prediction_error_pixels) * (
                1.0 + max(0, int(gap_frames)) * 0.1
            )
            if max_err > 0:
                direction_score = max(0.0, 1.0 - (pred_err_value / max_err))
    weights["direction"] = direction_score * w_direction

    # 5. Penalidade gradual de staticidade --------------------------------
    static_penalty = calc_static_penalty(candidate, accepted_position_history, cfg)
    weights["static"] = (1.0 - static_penalty) * w_static

    score = sum(weights.values())

    # Boost na zona da rede --------------------------------------------------
    if in_net_zone:
        score *= float(cfg.score_net_zone_boost)
        weights["net_zone_boost"] = float(cfg.score_net_zone_boost)

    # Anota debug no candidato (sobrescreve cost-based final_score sem o usar).
    candidate["score_breakdown"] = dict(weights)
    candidate["score"] = float(score)
    candidate["score_speed_mps"] = speed_value_mps
    candidate["score_pred_err"] = pred_err_value
    candidate["score_static_penalty"] = static_penalty
    candidate["score_full_recovery"] = bool(full_recovery)
    candidate["score_in_net_zone"] = bool(in_net_zone)

    return float(score), weights, bool(full_recovery)


def select_ball_candidate_by_score(
    candidates: List[Dict],
    *,
    last_accepted_center: Optional[Tuple[int, int]],
    older_center: Optional[Tuple[int, int]],
    accepted_position_history: Deque[Tuple[float, float]],
    gap_frames: int,
    fps: float,
    pixels_per_meter: float,
    max_ball_speed_ms: float,
    foreground_filter_active: bool,
    cfg: BallTrackingConfig = DEFAULT_CONFIG,
    candidate_evaluator: Optional[Callable[[Dict], Dict]] = None,
    net_zone_evaluator: Optional[Callable[[Tuple[int, int]], bool]] = None,
    pixel_to_court_m: Optional[Callable[[Tuple[int, int]], Tuple[float, float]]] = None,
) -> Tuple[Optional[Dict], str, Optional[Dict], float, bool]:
    """Seleciona o candidato com maior score que passe o threshold.

    Retorna: (escolhido, reason, melhor_rejeitado_por_contexto, score_top, recovery_used).

    Threshold:
      score >= score_accept_threshold (0.55)              → aceitar sempre
      score_recovery_threshold (0.35) <= score < 0.55     → aceitar SE gap >= score_recovery_gap_frames
                                                            (marca interpolated=True)
      score < 0.35                                         → rejeitar
    """
    if not candidates:
        return None, "missed_frame_no_valid_candidate", None, 0.0, False

    candidate_pool: List[Dict] = []
    best_context_rejected: Optional[Dict] = None
    for candidate in candidates:
        if candidate_evaluator is not None:
            context = candidate_evaluator(candidate)
            candidate["game_context"] = dict(context or {})
            if bool((context or {}).get("reject", False)):
                if best_context_rejected is None or float(candidate.get("confidence", 0.0)) > float(
                    best_context_rejected.get("confidence", 0.0)
                ):
                    best_context_rejected = dict(candidate)
                continue
        candidate_pool.append(candidate)

    if not candidate_pool:
        return None, "rejected_by_game_context", best_context_rejected, 0.0, False

    scored: List[Tuple[float, bool, Dict]] = []
    for candidate in candidate_pool:
        in_net = bool(net_zone_evaluator(candidate["center"])) if net_zone_evaluator else False
        score, breakdown, full_recovery = score_candidate(
            candidate,
            last_accepted_center=last_accepted_center,
            older_center=older_center,
            accepted_position_history=accepted_position_history,
            gap_frames=gap_frames,
            fps=fps,
            pixels_per_meter=pixels_per_meter,
            max_ball_speed_ms=max_ball_speed_ms,
            foreground_filter_active=foreground_filter_active,
            in_net_zone=in_net,
            cfg=cfg,
            pixel_to_court_m=pixel_to_court_m,
        )
        scored.append((score, full_recovery, candidate))
        print(
            f"[SCORE] center={candidate['center']} score={score:.3f} "
            f"yolo={breakdown.get('yolo_conf', 0):.2f} speed={breakdown.get('speed', 0):.2f} "
            f"motion={breakdown.get('motion', 0):.2f} dir={breakdown.get('direction', 0):.2f} "
            f"static={breakdown.get('static', 0):.2f} net={'Y' if in_net else 'N'} "
            f"gap={gap_frames} fullrec={full_recovery}"
        )

    scored.sort(key=lambda s: s[0], reverse=True)
    top_score, top_full_recovery, top_candidate = scored[0]

    accept_th = float(cfg.score_accept_threshold)
    recover_th = float(cfg.score_recovery_threshold)
    recover_gap = int(cfg.score_recovery_gap_frames)

    if top_score >= accept_th:
        print(f"[SCORE-ACCEPT] center={top_candidate['center']} score={top_score:.3f} >= {accept_th:.2f}")
        return top_candidate, "selected_by_score", best_context_rejected, top_score, top_full_recovery

    if top_score >= recover_th and int(gap_frames) >= recover_gap:
        # Recovery: aceita com confiança degradada → marca interpolated=True
        # para que classificadores de spike/block/ace o ignorem.
        top_candidate["interpolated"] = True
        print(
            f"[RECOVERY-MODE] Ativado após {gap_frames} frames, "
            f"threshold relaxado: aceite center={top_candidate['center']} "
            f"score={top_score:.3f} (>= {recover_th:.2f})"
        )
        return top_candidate, "selected_by_score_recovery", best_context_rejected, top_score, True

    print(
        f"[SCORE-REJECT] center={top_candidate['center']} score={top_score:.3f} "
        f"< {recover_th:.2f} (gap={gap_frames})"
    )
    return None, "rejected_by_score", best_context_rejected, top_score, False


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
        # Streak de frames em que cada posição quantizada apareceu consecutivamente.
        self._static_candidate_counts: Dict[Tuple[int, int], int] = {}
        # Posições confirmadas como FP estático — rejeitadas permanentemente.
        self._static_fp_blacklist: Set[Tuple[int, int]] = set()

        # Fix 3: histórico das últimas N posições aceites e blacklist persistente
        # baseada em proximidade real (±radius_px), não em quantização em grelha.
        self._accepted_position_history: Deque[Tuple[float, float]] = deque(
            maxlen=int(cfg.position_blacklist_window)
        )
        self._position_blacklist: List[Tuple[float, float]] = []

        # Fix 1: cache do último threshold de confiança imprimido para evitar log spam.
        self._last_logged_conf: Optional[float] = None

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
        # Streak de FP estático reseta por rally; blacklist persiste (o objeto não se move).
        self._static_candidate_counts.clear()
        # Histórico de posições aceites também reseta — a blacklist *permanece*
        # (uma posição confirmada como FP estático nesta sessão continua a sê-lo).
        self._accepted_position_history.clear()
        self._last_logged_conf = None

    # --- Fix 1: Threshold adaptativo de confiança YOLO ---------------------
    def current_conf_threshold(self) -> float:
        """Retorna o threshold YOLO efetivo em função do estado do tracker.

        Estável → high (0.35). 5+ misses → mid (0.20). 15+ misses → low (0.15).
        """
        cfg = self.cfg
        misses = int(self.missed_frames)
        if misses >= int(cfg.adaptive_conf_low_after_misses):
            new_conf = float(cfg.adaptive_conf_low)
        elif misses >= int(cfg.adaptive_conf_mid_after_misses):
            new_conf = float(cfg.adaptive_conf_mid)
        else:
            new_conf = float(cfg.adaptive_conf_high)
        if self._last_logged_conf is None or abs(self._last_logged_conf - new_conf) > 1e-6:
            if misses >= int(cfg.adaptive_conf_mid_after_misses):
                print(
                    f"[ADAPTIVE-CONF] Threshold baixado para {new_conf:.2f} "
                    f"após {misses} frames sem deteção"
                )
            elif self._last_logged_conf is not None:
                print(f"[ADAPTIVE-CONF] Threshold restaurado para {new_conf:.2f} (bola reencontrada)")
            self._last_logged_conf = new_conf
        return new_conf

    # --- Fix 2: Multiplicador de busca em função do gap --------------------
    def current_search_multiplier(self) -> float:
        """Expande as gates de step/prediction-error após gaps longos."""
        cfg = self.cfg
        misses = int(self.missed_frames)
        if misses >= int(cfg.gap_search_expand_4x_after):
            return 4.0
        if misses >= int(cfg.gap_search_expand_2x_after):
            return 2.0
        return 1.0

    # --- Fix 3: Blacklist por proximidade real (±radius_px) ----------------
    def _candidate_in_blacklist(self, center: Tuple[int, int]) -> bool:
        if not self._position_blacklist:
            return False
        r = float(self.cfg.position_blacklist_reject_radius_px)
        cx, cy = float(center[0]), float(center[1])
        for bx, by in self._position_blacklist:
            if math.hypot(cx - bx, cy - by) <= r:
                return True
        return False

    def _filter_blacklisted_candidates(self, candidates: List[Dict]) -> List[Dict]:
        if not self._position_blacklist:
            return candidates
        return [c for c in candidates if not self._candidate_in_blacklist(c["center"])]

    def _register_accepted_position(self, center: Tuple[int, int]) -> None:
        """Adiciona a deteção aceite ao histórico — fonte da static_penalty.

        A blacklist binária foi substituída pela penalidade gradual em
        `score_candidate`/`calc_static_penalty`. Mantemos apenas o histórico
        de posições aceites (deque com janela `position_blacklist_window`),
        que é exatamente o sinal usado pela static_penalty.
        """
        cx, cy = float(center[0]), float(center[1])
        self._accepted_position_history.append((cx, cy))

    def build_foreground_mask(self, frame):
        if self.bg_subtractor is None:
            return None
        return build_foreground_mask(self.bg_subtractor, frame)

    def trajectory_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        points = list(self.trajectory)
        if last_n is not None:
            return points[-max(1, int(last_n)) :]
        return points

    def _quantize_pos(self, center: Tuple[int, int]) -> Tuple[int, int]:
        g = max(1, int(self.cfg.static_fp_grid_px))
        return (round(center[0] / g) * g, round(center[1] / g) * g)

    def _filter_and_update_static_fp(self, candidates: List[Dict]) -> List[Dict]:
        """Remove candidatos estáticos (FP persistente fora do campo).

        Um candidato é classificado como FP estático se a sua posição quantizada
        (grid de static_fp_grid_px) aparece em static_fp_max_frames frames
        consecutivos. A posição é adicionada ao blacklist e rejeitada para sempre.
        """
        max_frames = int(self.cfg.static_fp_max_frames)
        seen: Set[Tuple[int, int]] = {self._quantize_pos(c["center"]) for c in candidates}

        for qpos in seen:
            if qpos in self._static_fp_blacklist:
                continue
            count = self._static_candidate_counts.get(qpos, 0) + 1
            self._static_candidate_counts[qpos] = count
            if count >= max_frames:
                self._static_fp_blacklist.add(qpos)
                print(f"[STATIC-FP] Candidato rejeitado em {qpos} após {count} frames estáticos")

        # Resetar streak de posições que não apareceram neste frame.
        for qpos in [k for k in list(self._static_candidate_counts) if k not in seen]:
            del self._static_candidate_counts[qpos]

        return [c for c in candidates if self._quantize_pos(c["center"]) not in self._static_fp_blacklist]

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
        previous_in_net_zone: bool = False,
        net_zone_evaluator: Optional[Callable[[Tuple[int, int]], bool]] = None,
        max_ball_speed_ms: Optional[float] = None,
        pixel_to_court_m: Optional[Callable[[Tuple[int, int]], Tuple[float, float]]] = None,
    ) -> BallTrackResult:
        # `previous_in_net_zone` permanece para compatibilidade. O sistema de
        # scoring usa `net_zone_evaluator(center)` por candidato (boost ×1.2).
        cfg = self.cfg
        ppm = cfg.pixels_per_meter if pixels_per_meter is None else float(pixels_per_meter)
        max_speed_ms = float(max_ball_speed_ms) if max_ball_speed_ms is not None else 35.0
        predicted_before = predict_next_center(self.previous_center, self.older_center)
        ball_candidates, candidate_stats = get_ball_candidates(result, frame_shape, fg_mask, cfg)
        # Os filtros binários antigos (static_fp grid, position blacklist) foram
        # substituídos pela penalidade gradual `static_penalty` no scoring.
        # O método `_filter_and_update_static_fp` é mantido em código mas já
        # não é chamado no caminho principal.

        gap_frames = int(self.missed_frames)
        foreground_filter_active = bool(candidate_stats.get("foreground_filter_active", False))

        detection, selection_reason, context_rejected_candidate, top_score, recovery_used = (
            select_ball_candidate_by_score(
                ball_candidates,
                last_accepted_center=self.previous_center,
                older_center=self.older_center,
                accepted_position_history=self._accepted_position_history,
                gap_frames=gap_frames,
                fps=float(fps) if fps else 30.0,
                pixels_per_meter=float(ppm),
                max_ball_speed_ms=max_speed_ms,
                foreground_filter_active=foreground_filter_active,
                cfg=cfg,
                candidate_evaluator=context_evaluator,
                net_zone_evaluator=net_zone_evaluator,
                pixel_to_court_m=pixel_to_court_m,
            )
        )
        context_decision: Optional[Dict] = None
        if detection is not None:
            context_decision = dict(detection.get("game_context") or {})
            if context_decision:
                candidate_stats["game_context"] = dict(context_decision)
        elif context_rejected_candidate is not None:
            context_decision = dict(context_rejected_candidate.get("game_context") or {})
            if context_decision:
                candidate_stats["game_context"] = dict(context_decision)
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
            # O selector baseado em score já decidiu: confiamos na sua decisão.
            # Velocidade/staticidade são componentes contínuas do score, não
            # gates binárias adicionais (manter gates duplicaria a rejeição que
            # o score já incorporou de forma graduada).
            current_center = detection["center"]
            speed_px = pixel_distance(self.previous_center, current_center)
            if self.previous_center is not None and fps:
                # Velocidade em metros reais via homografia (corrige perspectiva).
                # Fallback para `calculate_speed` (pixels_per_meter fixo) só se
                # o callback não estiver disponível.
                raw_speed_kmh = None
                if pixel_to_court_m is not None:
                    try:
                        prev_xy = pixel_to_court_m(self.previous_center)
                        curr_xy = pixel_to_court_m(current_center)
                        court_dist_m = math.hypot(curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1])
                        dt_s = max((max(0, gap_frames) + 1) / float(fps), 1.0 / 240.0)
                        raw_speed_kmh = (court_dist_m / dt_s) * 3.6
                    except Exception:
                        raw_speed_kmh = None
                if raw_speed_kmh is None and ppm:
                    _dist_px, _speed_pxps, _speed_mps, raw_speed_kmh = calculate_speed(
                        self.previous_center,
                        current_center,
                        fps,
                        ppm,
                    )
                if raw_speed_kmh is not None:
                    self.speed_history_kmh.append(raw_speed_kmh)

            self.older_center = self.previous_center
            self.previous_center = current_center
            self.last_accepted_center = current_center
            self.stationary_counter = 0
            self.missed_frames = 0
            self.trajectory.append(current_center)
            # Mantém o histórico de posições aceites para a static_penalty.
            self._register_accepted_position(current_center)
            detection_accepted = True
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
            # Propaga score/recovery: interpolated=True faz com que classifi-
            # cadores de spike/block/ace ignorem este ponto se quiserem.
            accepted_detection["score"] = float(top_score)
            accepted_detection["score_breakdown"] = dict(detection.get("score_breakdown") or {})
            accepted_detection["recovery_used"] = bool(recovery_used)
            if recovery_used or bool(detection.get("interpolated", False)):
                accepted_detection["interpolated"] = True
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
                "score_top": float(top_score),
                "score_breakdown": dict((selected_candidate or {}).get("score_breakdown") or {}),
                "recovery_used": bool(recovery_used),
            },
        )
        self.last_result = result_obj
        return result_obj
