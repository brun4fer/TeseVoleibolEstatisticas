"""
tracker.py
-------------
Detection and tracking for volleyball:
- YOLO (ultralytics) + ByteTrack for player IDs.
- Kalman filter for ball interpolation on missed frames.
- Pixel -> court conversion via homography.
- Geometric net event detector for high precision blocks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ball_tracking_core import (
    BallTrackerCore,
    BallTrackingConfig,
    TRACK_LOST,
    TRACK_OBSERVED,
    TRACK_PREDICTED,
    resize_frame_with_scale,
    scale_detection,
    scale_point,
    scale_points,
)
from config import config
from court_geometry import CourtGeometry
from volleyball_rules import VolleyballGameIntelligence, VolleyballRulesConfig

# Relevant COCO classes
CLASS_PERSON = 0
CLASS_SPORTS_BALL = 32


@dataclass
class BallState:
    pixel: Tuple[float, float]
    court: Tuple[float, float]
    speed_px: float
    visible: bool
    vx: float
    vy: float
    track_state: str = TRACK_LOST
    accepted_ball_center: Optional[Tuple[float, float]] = None
    predicted_ball_center: Optional[Tuple[float, float]] = None
    ball_side: Optional[str] = None
    possession_side: Optional[str] = None
    possession_team: Optional[str] = None
    ball_quality: str = "lost"
    ball_accepted_for_stats: bool = False
    game_context: Optional[Dict] = None


class BallKalman:
    """Kalman filter for (x, y) position with constant velocity (vx, vy)."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.measurement_noise
        self.initialized = False

    def reset(self):
        self.initialized = False

    def update(self, meas: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if meas is not None:
            x, y = meas
            if not self.initialized:
                self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
                self.initialized = True
            est = self.kf.correct(np.array([[x], [y]], np.float32))
            return float(est[0]), float(est[1])
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])


class VolleyballTracker:
    def __init__(self, H: np.ndarray, net_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path(getattr(config, "ball_yolo_model", "runs/detect/train5/weights/best.pt"))
        if not model_path.exists():
            candidate = Path("runs/detect/train5/weights/best.pt")
            model_path = candidate if candidate.exists() else Path(config.yolo_model)
        if not model_path.exists():
            model_path = Path("best.pt")
        self.model = YOLO(str(model_path))
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
        self.yolo_device = 0 if self.device.startswith("cuda") else "cpu"

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.net_line = net_line
        self.geometry = CourtGeometry(
            H,
            net_line,
            court_margin_m=float(getattr(config, "court_margin_m", 0.35)),
            neutral_tolerance_px=float(getattr(config, "game_net_neutral_px", getattr(config, "net_buffer_px", 15.0))),
            net_zone_tolerance_px=float(getattr(config, "net_band_height_px", 25.0)),
        )
        self.ball_kf = BallKalman()
        self.trail: Deque[Tuple[int, int]] = deque(maxlen=config.max_trail)
        self.ball_drawer: List[Tuple[float, float, float]] = []
        self.ball_drawer_maxlen: int = 200
        self.ball_drawer_jump_px: float = 700.0
        self.ball_drawer_jump_after_occlusion_px: float = 700.0
        self.ball_drawer_occlusion_frames: int = 5
        # Fix 4: interpolação de trajetória para gaps curtos. Aplicamos para
        # qualquer região do campo (não apenas zona da rede) quando o gap é
        # curto e temos posição confirmada antes E depois do mesmo.
        self.ball_drawer_infill_min_missing_frames: int = 2
        self.ball_drawer_infill_max_missing_frames: int = 8
        self.ball_drawer_infill_net_dist_px: float = float(config.net_band_height_px) + 40.0
        # Sidecar set com as chaves (x, y, t) de pontos interpolados — usado
        # por consumidores que precisem distinguir pontos reais vs. interpolados
        # (ex: classificação de spike/block deve ignorar interpolated=True).
        self._interpolated_drawer_keys: set = set()
        self.ball_drawer_hold_s: float = 2.0
        self.ball_drawer_hold_until_ts: float = -1e9
        self.ball_drawer_clear_after_ts: Optional[float] = None
        self.ball_drawer_outlier_px: float = 220.0
        self.ball_drawer_outlier_bridge_px: float = 180.0
        self.ball_conf_high: float = float(getattr(config, "ball_conf_high", 0.35))
        self.ball_conf_low: float = float(getattr(config, "ball_conf_low", 0.15))
        self.kalman_gate_px: float = float(getattr(config, "kalman_gate_px", 60.0))
        self.ball_min_area_px: float = float(getattr(config, "ball_min_area_px", 10.0))
        self.ball_max_area_px: float = float(getattr(config, "ball_max_area_px", 2000.0))
        self.ball_core = BallTrackerCore(
            BallTrackingConfig(
                conf_threshold=float(getattr(config, "ball_core_conf_threshold", 0.15)),
                resize_width=int(getattr(config, "ball_core_resize_width", 1280)),
                pixels_per_meter=float(getattr(config, "ball_pixels_per_meter", 50.0)),
                trajectory_length=int(getattr(config, "ball_debug_trajectory_length", 40)),
                max_trajectory_segment_pixels=float(getattr(config, "ball_debug_max_segment_px", 120.0)),
                static_fp_max_frames=int(getattr(config, "static_fp_max_frames", 30)),
                static_fp_grid_px=int(getattr(config, "static_fp_grid_px", 30)),
            )
        )
        self.game_intelligence = VolleyballGameIntelligence(
            self.geometry,
            VolleyballRulesConfig.from_config(config),
        )
        self.last_game_context: Dict = self.game_intelligence.context_snapshot()
        self.ball_frame_scale_x: float = 1.0
        self.ball_frame_scale_y: float = 1.0
        self.last_ball_core_result = None
        self.net_buffer_px: float = float(getattr(config, "net_buffer_px", 15.0))
        self.side_confirm_frames: int = 5
        self.current_ball_side: Optional[str] = None
        self.side_streak_frames: int = 0
        self.possession_confirm_frames: int = 3
        self.current_possession: Optional[str] = None
        self.campo_posse_atual: Optional[str] = None
        self.posse_atual: Optional[str] = None
        self.attacking_side: Optional[str] = None
        self.tocou_rede: bool = False
        self.last_ball_detected = False
        self.frames_since_ball = 0
        self.ball_id = 1
        self.ball_missing = 0
        self.ball_last_det: Optional[Dict] = None
        self.current_ball_det: Optional[Dict] = None
        self.field_roi_polygon: Optional[np.ndarray] = self._build_field_roi_polygon()
        self.field_roi_top_y: Optional[float] = float(np.min(self.field_roi_polygon[:, 1])) if self.field_roi_polygon is not None else None
        self.ceiling_margin_px: float = 150.0

        # High-precision block/net detector parameters.
        self.block_height_margin_m = float(getattr(config, "block_height_margin_m", 0.50))
        self.block_height_margin_px = self._meters_to_pixels_near_net(self.block_height_margin_m)
        self.block_player_proximity_px = float(getattr(config, "block_player_proximity_px", 110.0))
        self.block_min_vx_px = float(getattr(config, "block_min_vx_px", 1.5))
        self.block_occlusion_max_frames = int(getattr(config, "block_occlusion_max_frames", 12))
        self.block_event_ttl_s = float(getattr(config, "block_event_ttl_s", 3.0))
        self.block_event_cooldown_frames = int(getattr(config, "block_event_cooldown_frames", 6))

        self.recent_net_events: Deque[Dict] = deque(maxlen=180)
        self.pending_net_occlusion: Optional[Dict] = None
        self.last_visible_ball: Optional[Dict] = None
        self.prev_ball_visible = False
        self.last_net_event_frame = -10_000

    def _class_name(self, cls: int, res) -> str:
        names = getattr(res, "names", None)
        if isinstance(names, dict):
            return str(names.get(int(cls), "")).lower()
        if isinstance(names, list) and 0 <= int(cls) < len(names):
            return str(names[int(cls)]).lower()
        return ""

    def _is_ball_detection(self, cls: int, res) -> bool:
        # Supports COCO (sports ball=32) and custom single-class ball models (often class 0).
        name = self._class_name(cls, res)
        if "ball" in name:
            return True
        names = getattr(res, "names", None)
        if isinstance(names, dict) and len(names) == 1 and int(cls) == 0:
            return True
        if isinstance(names, list) and len(names) == 1 and int(cls) == 0:
            return True
        if int(cls) == 0 and "person" not in name:
            return True
        return int(cls) == CLASS_SPORTS_BALL

    def _is_person_detection(self, cls: int, res) -> bool:
        name = self._class_name(cls, res)
        if "person" in name:
            return True
        return int(cls) == CLASS_PERSON and "ball" not in name

    def detect(
        self,
        frame,
        timestamp_s: Optional[float] = None,
        fps: Optional[float] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict:
        """Run the same ball decision pipeline used by test_ball_detection.py."""
        ball_frame, scale_x, scale_y = resize_frame_with_scale(frame, self.ball_core.cfg.resize_width)
        self.ball_frame_scale_x = float(scale_x)
        self.ball_frame_scale_y = float(scale_y)

        fg_mask = self.ball_core.build_foreground_mask(ball_frame)
        infer_size = max(ball_frame.shape[0], ball_frame.shape[1])
        # Fix 1: usa o threshold adaptativo do core. Quando o tracker está
        # estável o YOLO devolve só candidatos fortes; em frames sem deteção
        # baixa progressivamente para recuperar a bola após oclusão/blur.
        adaptive_conf = self.ball_core.current_conf_threshold()
        results = self.model.predict(
            source=ball_frame,
            conf=adaptive_conf,
            imgsz=infer_size,
            device=self.yolo_device,
            verbose=False,
        )
        res = results[0] if results else None
        fps_value = float(fps) if fps is not None and fps > 0 else 30.0
        context_evaluator = None
        if getattr(self.game_intelligence.cfg, "enabled", True) and getattr(self.game_intelligence.cfg, "validate_ball_candidates", True):
            def context_evaluator(candidate: Dict) -> Dict:
                scaled_candidate = scale_detection(candidate, scale_x, scale_y)
                center = scaled_candidate.get("center") if scaled_candidate is not None else None
                decision = self.game_intelligence.evaluate_candidate(center, timestamp_s, frame_idx)
                return decision.to_dict()

        # Boost de zona da rede para o scoring (avalia cada candidato em
        # coords full-frame via signed_distance_to_net).
        net_zone_radius_m = float(self.ball_core.cfg.score_net_zone_radius_m)
        ppm_near_net = self._meters_to_pixels_near_net(1.0) or 50.0
        net_zone_radius_px = float(net_zone_radius_m) * float(ppm_near_net)

        def net_zone_evaluator(center_resized: Tuple[int, int]) -> bool:
            full_pt = scale_point(center_resized, scale_x, scale_y)
            if full_pt is None:
                return False
            try:
                return abs(self._signed_side(full_pt)) < net_zone_radius_px
            except Exception:
                return False

        # Conversor pixel→court(metros) via homografia. Usado pelo scoring e
        # pela velocidade de display para calcular distância REAL em metros
        # (não em pixels), corrigindo automaticamente a perspectiva — longe
        # da câmara 1 px = mais metros, perto da câmara 1 px = menos metros.
        def pixel_to_court_m(center_resized: Tuple[int, int]) -> Tuple[float, float]:
            full_pt = scale_point(center_resized, scale_x, scale_y)
            return self.geometry.pixel_to_court(full_pt)

        max_ball_speed_ms = float(getattr(config, "max_ball_speed_ms", 35.0))

        core_result = self.ball_core.update_from_yolo_result(
            res,
            ball_frame.shape,
            fg_mask,
            fps=fps_value,
            pixels_per_meter=self.ball_core.cfg.pixels_per_meter,
            context_evaluator=context_evaluator,
            net_zone_evaluator=net_zone_evaluator,
            max_ball_speed_ms=max_ball_speed_ms,
            pixel_to_court_m=pixel_to_court_m,
        )
        self.last_ball_core_result = core_result

        players: List[Dict] = []
        if res is not None and getattr(res, "boxes", None) is not None:
            for box in res.boxes:
                cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                if not self._is_person_detection(cls, res):
                    continue
                conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                if conf < 0.40:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tid = int(box.id.item()) if box.id is not None else -1
                players.append(
                    {
                        "id": tid,
                        "bbox": (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                        "conf": conf,
                    }
                )

        accepted_det = scale_detection(core_result.accepted_detection, scale_x, scale_y)
        predicted_center = scale_point(core_result.predicted_ball_center, scale_x, scale_y)
        if accepted_det is not None:
            game_context_decision = dict(core_result.debug.get("game_context") or {})
            accepted_det["conf"] = float(accepted_det.get("confidence", 0.0))
            accepted_det["visible"] = True
            accepted_det["track_state"] = TRACK_OBSERVED
            accepted_det["accepted_ball_center"] = accepted_det["center"]
            accepted_det["predicted_ball_center"] = predicted_center
            accepted_det["selection_reason"] = core_result.selection_reason
            accepted_det["foreground_reason"] = core_result.foreground_reason
            accepted_det["game_context_decision"] = game_context_decision
            accepted_det["ball_core_debug"] = dict(core_result.debug)
            now_t = float(timestamp_s) if timestamp_s is not None else float(time.time())
            self._infill_ball_drawer_gap_if_needed(float(accepted_det["center"][0]), float(accepted_det["center"][1]), now_t)
            self._append_ball_drawer(float(accepted_det["center"][0]), float(accepted_det["center"][1]), now_t)
            self.ball_missing = 0
            self.ball_last_det = dict(accepted_det)
            self.current_ball_det = dict(accepted_det)
            ball_det = accepted_det
        else:
            self.current_ball_det = None
            ball_det = {
                "bbox": None,
                "center": predicted_center,
                "conf": 0.0,
                "confidence": 0.0,
                "visible": False,
                "track_state": core_result.ball_track_state,
                "accepted_ball_center": None,
                "predicted_ball_center": predicted_center,
                "selection_reason": core_result.selection_reason,
                "foreground_reason": core_result.foreground_reason,
                "game_context_decision": dict(core_result.debug.get("game_context") or {}),
                "ball_core_debug": dict(core_result.debug),
            }
            if core_result.ball_track_state == TRACK_LOST:
                ball_det = None

        if getattr(config, "BALL_DEBUG_LOG", False):
            print(self._ball_core_log_line(core_result, timestamp_s=timestamp_s))

        return {"players": players, "ball_det": ball_det}

    def _ball_core_log_line(self, core_result, timestamp_s: Optional[float] = None) -> str:
        selected = core_result.selected_candidate
        if selected is None:
            center_txt = "None"
            conf_txt = "--"
            score_txt = "--"
            pred_txt = "--"
            fg_txt = "--"
        else:
            center_txt = str(selected.get("center"))
            conf_txt = f"{float(selected.get('confidence', 0.0)):.2f}"
            score = selected.get("final_score")
            score_txt = f"{float(score):.2f}" if score is not None else "--"
            pred = selected.get("prediction_error")
            pred_txt = f"{float(pred):.2f}" if pred is not None else "--"
            fg_txt = f"{int(selected.get('fg_active_pixels', 0))}/{float(selected.get('fg_active_ratio', 0.0)):.2f}"
        game_decision = core_result.debug.get("game_context") if core_result is not None else None
        game_reason = ""
        if game_decision:
            game_reason = (
                f" game={game_decision.get('quality')} "
                f"penalty={float(game_decision.get('penalty', 0.0)):.1f} "
                f"greason={game_decision.get('reason')}"
            )
        ts_txt = "--" if timestamp_s is None else f"{float(timestamp_s):.2f}"
        return (
            f"[BALL-CORE] ts={ts_txt} state={core_result.ball_track_state} "
            f"center={center_txt} conf={conf_txt} score={score_txt} pred_err={pred_txt} "
            f"fg={fg_txt} reason={core_result.selection_reason} missed={core_result.missed_frames}{game_reason}"
        )

    def update_ball(
        self,
        ball_det: Optional[Dict],
        timestamp_s: Optional[float] = None,
        frame_idx: Optional[int] = None,
    ) -> BallState:
        ball_visible = bool(ball_det is not None and bool(ball_det.get("visible", False)))
        meas = ball_det["center"] if ball_visible else None
        track_state = str(ball_det.get("track_state", TRACK_OBSERVED if ball_visible else TRACK_LOST)) if ball_det is not None else TRACK_LOST
        accepted_ball_center = ball_det.get("accepted_ball_center") if ball_det is not None else None
        predicted_ball_center = ball_det.get("predicted_ball_center") if ball_det is not None else None
        was_visible = self.last_ball_detected

        if meas is not None:
            self.ball_kf.update(meas)
            x, y = float(meas[0]), float(meas[1])
            self.frames_since_ball = 0
            self.last_ball_detected = True
        else:
            if track_state == TRACK_PREDICTED and predicted_ball_center is not None:
                kx, ky = float(predicted_ball_center[0]), float(predicted_ball_center[1])
            elif self.ball_kf.initialized:
                kx, ky = self.ball_kf.update(None)
            else:
                kx, ky = (self.trail[-1] if self.trail else (0.0, 0.0))
            v_avg = self._avg_velocity()
            if v_avg is not None and self.trail:
                x = self.trail[-1][0] + v_avg[0]
                y = self.trail[-1][1] + v_avg[1]
                x = 0.5 * x + 0.5 * kx
                y = 0.5 * y + 0.5 * ky
            else:
                x, y = kx, ky
            self.frames_since_ball += 1
            self.last_ball_detected = False
            if timestamp_s is not None and was_visible:
                # Keep last detections available during short occlusions near block.
                self.ball_drawer_hold_until_ts = float(timestamp_s) + self.ball_drawer_hold_s

        self.trail.append((int(x), int(y)))
        if abs(float(x)) > 1e-6 or abs(float(y)) > 1e-6:
            # Dynamic possession must be tracked on every frame, including short occlusions.
            self.update_possession((float(x), float(y)))
        speed_px = self._instant_speed()
        court_xy = self.geometry.pixel_to_court((x, y))
        vx, vy = self._last_velocity()
        state = BallState(
            pixel=(x, y),
            court=court_xy,
            speed_px=speed_px,
            visible=ball_visible,
            vx=vx,
            vy=vy,
            track_state=track_state,
            accepted_ball_center=accepted_ball_center,
            predicted_ball_center=predicted_ball_center,
        )
        if getattr(self.game_intelligence.cfg, "enabled", True):
            context = self.game_intelligence.update_ball(
                state,
                timestamp_s=float(timestamp_s) if timestamp_s is not None else 0.0,
                frame_idx=int(frame_idx) if frame_idx is not None else len(self.trail),
            )
            context_dict = context.to_dict()
            self.last_game_context = context_dict
            state.ball_side = context.ball_side
            state.possession_side = context.possession_side
            state.possession_team = context.possession_team
            state.ball_quality = context.ball_quality
            state.ball_accepted_for_stats = context.ball_accepted_for_stats
            state.game_context = context_dict
        else:
            state.ball_quality = "trusted" if ball_visible else "lost"
            state.ball_accepted_for_stats = bool(ball_visible)
        return state

    def _append_ball_drawer(self, x: float, y: float, t: float) -> None:
        if abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6:
            return
        if not self._inside_field_roi((x, y)):
            return
        if self.ball_drawer:
            last_x, last_y, _last_t = self.ball_drawer[-1]
            dist = float(np.hypot(x - last_x, y - last_y))
            max_jump = self.ball_drawer_jump_after_occlusion_px if self.frames_since_ball > self.ball_drawer_occlusion_frames else self.ball_drawer_jump_px
            if dist > max_jump:
                print(f"[CLEANUP] Deteção descartada: Salto de {dist:.1f}px (Impossível).")
                return
        self.ball_drawer.append((float(x), float(y), float(t)))
        self._prune_isolated_outlier()
        if len(self.ball_drawer) > self.ball_drawer_maxlen:
            overflow = len(self.ball_drawer) - self.ball_drawer_maxlen
            for stale in self.ball_drawer[:overflow]:
                self._interpolated_drawer_keys.discard(stale)
            del self.ball_drawer[:overflow]
        print(f"A adicionar à gaveta: {x}, {y}. Total agora: {len(self.ball_drawer)}")

    def _kalman_predicted_center(self) -> Optional[Tuple[float, float]]:
        if self.ball_kf.initialized:
            try:
                state = np.asarray(self.ball_kf.kf.statePost, dtype=np.float32).reshape(-1, 1)
                trans = np.asarray(self.ball_kf.kf.transitionMatrix, dtype=np.float32)
                pred = trans @ state
                return float(pred[0]), float(pred[1])
            except Exception:
                pass
        if self.ball_last_det is not None:
            c = self.ball_last_det.get("center")
            if c is not None:
                return float(c[0]), float(c[1])
        if self.trail:
            return float(self.trail[-1][0]), float(self.trail[-1][1])
        return None

    def _infill_ball_drawer_gap_if_needed(self, x: float, y: float, t: float) -> None:
        # Fix 4: preenche gaps curtos (≤ 8 frames) por interpolação linear
        # entre a última posição conhecida e a posição reencontrada. Os pontos
        # interpolados ficam marcados via sidecar set para que consumidores
        # de event-classification (spike/block/ace) os possam ignorar.
        if not self.ball_drawer:
            return
        missing = int(self.frames_since_ball)
        if missing < self.ball_drawer_infill_min_missing_frames or missing > self.ball_drawer_infill_max_missing_frames:
            return

        last_x, last_y, last_t = self.ball_drawer[-1]

        added = 0
        for i in range(1, missing):
            alpha = i / float(missing)
            ix = float(last_x + (float(x) - float(last_x)) * alpha)
            iy = float(last_y + (float(y) - float(last_y)) * alpha)
            if abs(ix) < 1e-6 and abs(iy) < 1e-6:
                continue
            if not self._inside_field_roi((ix, iy)):
                continue
            if float(t) > float(last_t):
                it = float(last_t + (float(t) - float(last_t)) * alpha)
            else:
                it = float(last_t + (1e-3 * i))
            entry = (ix, iy, it)
            self.ball_drawer.append(entry)
            self._interpolated_drawer_keys.add(entry)
            added += 1

        if added > 0:
            self._prune_isolated_outlier()
            if len(self.ball_drawer) > self.ball_drawer_maxlen:
                overflow = len(self.ball_drawer) - self.ball_drawer_maxlen
                # Limpa chaves do sidecar referentes a entradas removidas.
                for stale in self.ball_drawer[:overflow]:
                    self._interpolated_drawer_keys.discard(stale)
                del self.ball_drawer[:overflow]
            print(
                f"[INTERP] Preenchidos {added} frames por interpolação entre "
                f"({last_x:.0f},{last_y:.0f}) e ({x:.0f},{y:.0f})"
            )

    def is_interpolated_drawer_point(self, point: Tuple[float, float, float]) -> bool:
        """True se o ponto da gaveta foi inserido por interpolação (Fix 4).

        Consumers que classifiquem spike/block/ace devem filtrar estes pontos.
        """
        return point in self._interpolated_drawer_keys

    def drawer_snapshot(self) -> List[Tuple[float, float, float]]:
        return self._ordered_ball_drawer()

    def drawer_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        ordered = self._ordered_ball_drawer()
        # Require at least 3 ordered points to infer a stable direction.
        if len(ordered) < 3:
            return []
        pts = [(int(p[0]), int(p[1])) for p in ordered]
        if last_n is not None:
            target_n = max(3, int(last_n))
            if len(pts) <= target_n:
                return pts
            return pts[-target_n:]
        return pts

    def accepted_ball_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        points = self.ball_core.trajectory_points(last_n=last_n)
        return scale_points(points, self.ball_frame_scale_x, self.ball_frame_scale_y)

    def ball_debug_snapshot(self) -> Dict:
        core_result = self.last_ball_core_result
        if core_result is None:
            quality = {
                "tracking": False,
                "lost": True,
                "missed_frames": 0,
                "max_missed_frames": self.ball_core.cfg.max_missed_frames,
                "reason": None,
                "track_state": TRACK_LOST,
            }
        else:
            quality = {
                "tracking": core_result.ball_track_state == TRACK_OBSERVED,
                "lost": core_result.ball_track_state == TRACK_LOST,
                "missed_frames": core_result.missed_frames,
                "max_missed_frames": self.ball_core.cfg.max_missed_frames,
                "reason": core_result.selection_reason,
                "track_state": core_result.ball_track_state,
                "foreground_reason": core_result.foreground_reason,
                "candidate_debug": core_result.debug.get("candidate_debug"),
            }
        return {
            "current_det": dict(self.current_ball_det) if self.current_ball_det is not None else None,
            "trajectory": self.accepted_ball_points(),
            "quality": quality,
            "game_context": self.game_context_snapshot(),
            "max_segment_px": float(self.ball_core.cfg.max_trajectory_segment_pixels * max(self.ball_frame_scale_x, self.ball_frame_scale_y)),
        }

    def game_context_snapshot(self) -> Dict:
        return dict(self.last_game_context) if self.last_game_context is not None else self.game_intelligence.context_snapshot()

    def note_score_change(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
        timestamp_s: float,
        frame_idx: int,
    ) -> None:
        self.game_intelligence.confirm_score_change(
            prev_score=prev_score,
            new_score=new_score,
            timestamp_s=float(timestamp_s),
            frame_idx=int(frame_idx),
        )
        self.last_game_context = self.game_intelligence.context_snapshot()

    def clear_ball_drawer(self) -> None:
        self.ball_drawer.clear()
        # Fix 4: invalida sidecar de pontos interpolados ao iniciar novo rally.
        self._interpolated_drawer_keys.clear()
        self.ball_drawer_hold_until_ts = -1e9
        self.ball_drawer_clear_after_ts = None
        self.tocou_rede = False
        self.current_ball_det = None
        self.ball_core.reset()
        self.game_intelligence.reset_for_new_rally()
        self.last_game_context = self.game_intelligence.context_snapshot()

    def reset_drawer_for_service(self, x: float, y: float, t: float) -> None:
        self.clear_ball_drawer()
        self.current_ball_side = None
        self.side_streak_frames = 0
        # Keep global possession across service resets.
        self.attacking_side = None
        self.pending_net_occlusion = None
        if abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6:
            return
        self.ball_drawer.append((float(x), float(y), float(t)))
        core_x = int(round(float(x) / max(self.ball_frame_scale_x, 1e-6)))
        core_y = int(round(float(y) / max(self.ball_frame_scale_y, 1e-6)))
        self.ball_core.reset((core_x, core_y))
        side = self._side_from_pixel((float(x), float(y)))
        self.current_ball_side = side
        self.side_streak_frames = 1
        print(f"A adicionar à gaveta: {x}, {y}. Total agora: {len(self.ball_drawer)}")

    def defer_clear_ball_drawer(self, clear_at_ts: float) -> None:
        # Disabled for rally persistence: drawer reset must happen only on service/OCR.
        self.ball_drawer_clear_after_ts = None

    def _maybe_clear_ball_drawer(self, timestamp_s: float) -> None:
        # Disabled for rally persistence: keep full drawer during the rally.
        return

    def _inside_field_roi(self, pixel_pt: Tuple[float, float]) -> bool:
        x, y = float(pixel_pt[0]), float(pixel_pt[1])
        if self.field_roi_polygon is not None:
            inside_polygon = cv2.pointPolygonTest(self.field_roi_polygon.astype(np.float32), (x, y), False) >= 0
            if not inside_polygon and self.field_roi_top_y is not None:
                x_min = float(np.min(self.field_roi_polygon[:, 0]))
                x_max = float(np.max(self.field_roi_polygon[:, 0]))
                inside_polygon = x_min <= x <= x_max and (self.field_roi_top_y - self.ceiling_margin_px) <= y <= self.field_roi_top_y
            if not inside_polygon:
                return False
            if self.field_roi_top_y is not None and y < (self.field_roi_top_y - self.ceiling_margin_px):
                return False
            return True
        court_pt = self.geometry.pixel_to_court((x, y))
        return self._court_contains(court_pt)

    def _build_field_roi_polygon(self) -> Optional[np.ndarray]:
        return self.geometry.get_court_zones()["court"].copy()

    def _prune_isolated_outlier(self) -> None:
        if len(self.ball_drawer) < 3:
            return
        p0 = self.ball_drawer[-3]
        p1 = self.ball_drawer[-2]
        p2 = self.ball_drawer[-1]
        d01 = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        d12 = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        d02 = float(np.hypot(p2[0] - p0[0], p2[1] - p0[1]))
        if d01 > self.ball_drawer_outlier_px and d12 > self.ball_drawer_outlier_px and d02 < self.ball_drawer_outlier_bridge_px:
            self._interpolated_drawer_keys.discard(self.ball_drawer[-2])
            del self.ball_drawer[-2]

    def _ordered_ball_drawer(self) -> List[Tuple[float, float, float]]:
        if len(self.ball_drawer) < 2:
            return list(self.ball_drawer)
        return sorted(self.ball_drawer, key=lambda p: float(p[2]))

    def detect_net_event(
        self,
        ball_state: BallState,
        players: List[Dict],
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[Dict]:
        """Detect BLOCK/BALL_ON_NET from geometric net logic."""
        self._prune_old_events(timestamp_s)

        event = None
        if ball_state.visible:
            event = self._detect_direct_net_event(players, frame_idx, timestamp_s)
            if event is None:
                event = self._resolve_occlusion_if_needed(ball_state, players, frame_idx, timestamp_s)
            else:
                self.pending_net_occlusion = None
            self.last_visible_ball = {
                "pt": (float(ball_state.pixel[0]), float(ball_state.pixel[1])),
                "vx": float(ball_state.vx),
                "frame_idx": frame_idx,
                "ts": timestamp_s,
            }
        else:
            self._open_occlusion_if_needed(frame_idx, timestamp_s)
            if self.pending_net_occlusion and frame_idx - self.pending_net_occlusion["start_frame"] > self.block_occlusion_max_frames:
                self.pending_net_occlusion = None

        self.prev_ball_visible = ball_state.visible
        if event is not None:
            self.tocou_rede = True
            self.recent_net_events.append(event)
            self.last_net_event_frame = frame_idx
        return event

    def get_recent_net_event(self, timestamp_s: float, max_age_s: Optional[float] = None) -> Optional[Dict]:
        max_age = self.block_event_ttl_s if max_age_s is None else max_age_s
        for ev in reversed(self.recent_net_events):
            if timestamp_s - ev["timestamp_s"] <= max_age:
                return ev
        return None

    def _ball_fallback_det(self) -> Optional[Dict]:
        self.ball_missing += 1
        if self.ball_missing <= config.ball_max_age_frames and self.ball_last_det is not None:
            fallback = dict(self.ball_last_det)
            fallback["visible"] = False
            return fallback
        self.ball_last_det = None
        return None

    def _instant_speed(self) -> float:
        if len(self.trail) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
        return float(np.hypot(x2 - x1, y2 - y1))

    def _avg_velocity(self) -> Optional[Tuple[float, float]]:
        if len(self.trail) < 3:
            return None
        v1 = np.array(self.trail[-1]) - np.array(self.trail[-2])
        v2 = np.array(self.trail[-2]) - np.array(self.trail[-3])
        v_mean = (v1 + v2) / 2.0
        return float(v_mean[0]), float(v_mean[1])

    def _last_velocity(self) -> Tuple[float, float]:
        if len(self.trail) < 2:
            return 0.0, 0.0
        v = np.array(self.trail[-1]) - np.array(self.trail[-2])
        return float(v[0]), float(v[1])

    def acceleration(self) -> float:
        if len(self.trail) < 3:
            return 0.0
        v1 = np.array(self.trail[-2]) - np.array(self.trail[-3])
        v2 = np.array(self.trail[-1]) - np.array(self.trail[-2])
        return float(np.linalg.norm(v2 - v1))

    def horizontal_inversion(self) -> float:
        """Return cosine between consecutive vectors (negative means inversion)."""
        if len(self.trail) < 3:
            return 1.0
        v1 = np.array(self.trail[-2]) - np.array(self.trail[-3])
        v2 = np.array(self.trail[-1]) - np.array(self.trail[-2])
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 1.0
        return float(np.dot(v1, v2) / norm)

    def trail_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        pts = self.drawer_points()
        if not pts:
            pts = list(self.trail)
        if last_n is not None:
            return pts[-last_n:]
        return pts

    def _court_contains(self, court_pt: Tuple[float, float]) -> bool:
        return self.geometry.is_inside_court(court_pt, point_space="court")

    def _side_from_pixel(self, pixel_pt: Tuple[float, float]) -> Optional[str]:
        side = self.geometry.get_side_of_net(pixel_pt, neutral_tolerance_px=1.0)
        return self.current_ball_side if side is None else side

    def update_possession(self, pixel_pt: Tuple[float, float]) -> Optional[str]:
        if self._dist_to_net_line(pixel_pt) < self.net_buffer_px:
            # Neutral zone near net: keep current confirmed possession as attack reference.
            return self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else self.attacking_side

        side = self._side_from_pixel(pixel_pt)
        if side is None:
            return self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else self.attacking_side

        previous_side = self.current_ball_side
        if self.current_ball_side is None:
            self.current_ball_side = side
            self.side_streak_frames = 1
        elif side == self.current_ball_side:
            self.side_streak_frames += 1
        else:
            # Attacker is the side with possession immediately before field change.
            pre_change_attacker = self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else self.current_ball_side
            if pre_change_attacker in ("CampoA", "CampoB"):
                self.attacking_side = pre_change_attacker
            # Do not reset drawer on net crossing; keep last 50 frames of full rally.
            self.current_ball_side = side
            self.side_streak_frames = 1

        if self.side_streak_frames > self.possession_confirm_frames:
            self.current_possession = side
            self.campo_posse_atual = side
            self.posse_atual = side
        elif self.current_possession is None:
            self.current_possession = side
            self.campo_posse_atual = side
            self.posse_atual = side

        # Attack reference must follow the current possession, not the service side.
        if self.ball_near_net(pixel_pt):
            net_attacker = self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else previous_side
            if net_attacker in ("CampoA", "CampoB"):
                self.attacking_side = net_attacker
        elif self.posse_atual in ("CampoA", "CampoB"):
            self.attacking_side = self.posse_atual
        return self.attacking_side

    def _signed_side(self, pixel_pt: Tuple[float, float]) -> float:
        return float(self.geometry.signed_distance_to_net(pixel_pt))

    def _project_point_to_net(self, pixel_pt: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        return self.geometry.project_point_to_net(pixel_pt)

    def _dist_to_net_line(self, pixel_pt: Tuple[float, float]) -> float:
        return float(self.geometry.distance_to_net(pixel_pt))

    def ball_near_net(self, pixel_pt: Tuple[float, float]) -> bool:
        return bool(self.geometry.is_in_net_zone(pixel_pt, tolerance_px=float(config.net_band_height_px)))

    def ball_on_net_line(self, pixel_pt: Tuple[float, float]) -> bool:
        return bool(self.geometry.is_in_net_zone(pixel_pt, tolerance_px=float(config.net_line_tolerance_px)))

    def crossed_net_line(self) -> bool:
        if len(self.trail) < 2:
            return False
        return bool(
            self.geometry.did_cross_net(
                self.trail[-2],
                self.trail[-1],
                neutral_tolerance_px=float(config.net_line_tolerance_px),
            )
        )

    def _segment_intersects_net(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance_px: float) -> bool:
        return bool(self.geometry.did_cross_net(p1, p2, neutral_tolerance_px=float(tolerance_px)))

    def _segment_intersection_point(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))
        return None

    def _impact_point_on_net(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        intersection = self._segment_intersection_point(p1, p2, self.net_line[0], self.net_line[1])
        if intersection is not None:
            return intersection
        proj1, d1 = self._project_point_to_net(p1)
        proj2, d2 = self._project_point_to_net(p2)
        return proj1 if d1 <= d2 else proj2

    def _height_filter_ok(self, impact_px: Tuple[float, float]) -> bool:
        net_proj, _ = self._project_point_to_net(impact_px)
        # y grows downward on image. Positive offset means "above" top of net.
        offset_above_net = net_proj[1] - impact_px[1]
        low = -float(config.net_line_tolerance_px)
        high = float(self.block_height_margin_px + config.net_line_tolerance_px)
        return low <= offset_above_net <= high

    def _is_vx_inversion(self, vx_before: float, vx_after: float) -> bool:
        if abs(vx_before) < self.block_min_vx_px or abs(vx_after) < self.block_min_vx_px:
            return False
        return vx_before * vx_after < 0

    def _teams_from_incoming_vx(self, incoming_vx: float) -> Tuple[str, str]:
        attacking = "TeamA" if incoming_vx > 0 else "TeamB"
        defending = "TeamB" if attacking == "TeamA" else "TeamA"
        return attacking, defending

    def _defender_side(self, pre_impact_pt: Tuple[float, float]) -> int:
        side = self._signed_side(pre_impact_pt)
        if abs(side) < 1e-6:
            return 0
        return -1 if side > 0 else 1

    def _player_center(self, player: Dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = player["bbox"]
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def _has_defender_near(self, impact_px: Tuple[float, float], players: List[Dict], defender_side: int) -> bool:
        impact = np.array(impact_px, dtype=np.float32)
        for p in players:
            center = self._player_center(p)
            dist = float(np.linalg.norm(np.array(center, dtype=np.float32) - impact))
            if dist > self.block_player_proximity_px:
                continue
            if defender_side == 0:
                return True
            p_side = self._signed_side(center)
            if abs(p_side) < 1e-6:
                return True
            if defender_side < 0 and p_side < 0:
                return True
            if defender_side > 0 and p_side > 0:
                return True
        return False

    def _build_net_event(
        self,
        event_type: str,
        impact_px: Tuple[float, float],
        incoming_vx: float,
        frame_idx: int,
        timestamp_s: float,
        via_occlusion: bool,
    ) -> Dict:
        attacking, defending = self._teams_from_incoming_vx(incoming_vx)
        return {
            "event_type": event_type,  # BLOCK or BALL_ON_NET
            "impact_px": (float(impact_px[0]), float(impact_px[1])),
            "attacking_team": attacking,
            "defending_team": defending,
            "incoming_vx": float(incoming_vx),
            "frame_idx": int(frame_idx),
            "timestamp_s": float(timestamp_s),
            "via_occlusion": bool(via_occlusion),
        }

    def _detect_direct_net_event(self, players: List[Dict], frame_idx: int, timestamp_s: float) -> Optional[Dict]:
        if frame_idx - self.last_net_event_frame < self.block_event_cooldown_frames:
            return None
        if len(self.trail) < 3:
            return None
        p0 = (float(self.trail[-3][0]), float(self.trail[-3][1]))
        p1 = (float(self.trail[-2][0]), float(self.trail[-2][1]))
        p2 = (float(self.trail[-1][0]), float(self.trail[-1][1]))
        vx_before = p1[0] - p0[0]
        vx_after = p2[0] - p1[0]
        if not self._is_vx_inversion(vx_before, vx_after):
            return None
        if not self._segment_intersects_net(p1, p2, float(config.net_line_tolerance_px)):
            return None
        contact_px = p1 if self._dist_to_net_line(p1) <= self._dist_to_net_line(p2) else p2
        if not self._height_filter_ok(contact_px):
            return None
        impact_px = self._impact_point_on_net(p1, p2)
        defender_side = self._defender_side(p1)
        has_defender = self._has_defender_near(impact_px, players, defender_side)
        event_type = "BLOCK" if has_defender else "BALL_ON_NET"
        return self._build_net_event(event_type, impact_px, vx_before, frame_idx, timestamp_s, via_occlusion=False)

    def _open_occlusion_if_needed(self, frame_idx: int, timestamp_s: float) -> None:
        if not self.prev_ball_visible:
            return
        if self.last_visible_ball is None:
            return
        pre_pt = self.last_visible_ball["pt"]
        if not self.ball_on_net_line(pre_pt):
            return
        pre_vx = float(self.last_visible_ball["vx"])
        if abs(pre_vx) < self.block_min_vx_px:
            return
        self.pending_net_occlusion = {
            "pre_pt": pre_pt,
            "pre_vx": pre_vx,
            "pre_side": self._signed_side(pre_pt),
            "start_frame": frame_idx,
            "start_ts": timestamp_s,
        }

    def _resolve_occlusion_if_needed(
        self,
        ball_state: BallState,
        players: List[Dict],
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[Dict]:
        if self.pending_net_occlusion is None:
            return None
        occl = self.pending_net_occlusion
        if frame_idx - occl["start_frame"] > self.block_occlusion_max_frames:
            self.pending_net_occlusion = None
            return None
        if frame_idx - self.last_net_event_frame < self.block_event_cooldown_frames:
            self.pending_net_occlusion = None
            return None

        re_pt = (float(ball_state.pixel[0]), float(ball_state.pixel[1]))
        re_vx = float(ball_state.vx)
        if abs(re_vx) < self.block_min_vx_px:
            self.pending_net_occlusion = None
            return None

        pre_pt = (float(occl["pre_pt"][0]), float(occl["pre_pt"][1]))
        pre_vx = float(occl["pre_vx"])
        pre_side = float(occl["pre_side"])
        re_side = self._signed_side(re_pt)
        opposite_side = pre_side * re_side < 0
        crossed_net = self._segment_intersects_net(pre_pt, re_pt, float(config.net_line_tolerance_px))
        inversion = self._is_vx_inversion(pre_vx, re_vx)
        contact_px = pre_pt if self._dist_to_net_line(pre_pt) <= self._dist_to_net_line(re_pt) else re_pt
        impact_px = self._impact_point_on_net(pre_pt, re_pt)
        height_ok = self._height_filter_ok(contact_px)

        self.pending_net_occlusion = None
        if not (opposite_side and crossed_net and inversion and height_ok):
            return None

        defender_side = self._defender_side(pre_pt)
        has_defender = self._has_defender_near(impact_px, players, defender_side)
        event_type = "BLOCK" if has_defender else "BALL_ON_NET"
        return self._build_net_event(event_type, impact_px, pre_vx, frame_idx, timestamp_s, via_occlusion=True)

    def _prune_old_events(self, timestamp_s: float) -> None:
        while self.recent_net_events and timestamp_s - self.recent_net_events[0]["timestamp_s"] > self.block_event_ttl_s:
            self.recent_net_events.popleft()

    def _court_to_pixel(self, court_pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        return self.geometry.court_to_pixel(court_pt)

    def _meters_to_pixels_near_net(self, meters: float) -> float:
        return float(self.geometry.estimate_pixels_per_meter_near_net(meters))

    def predict_impact_point(self) -> Optional[Tuple[int, int]]:
        if len(self.trail) < 2:
            return None
        (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
        dx, dy = x2 - x1, y2 - y1
        if dy <= 0:
            return None
        scale = (config.net_band_height_px * 20) / dy
        return int(x2 + dx * scale), int(y2 + dy * scale)
