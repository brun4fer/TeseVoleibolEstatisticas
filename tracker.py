"""
tracker.py
-------------
Deteção e tracking de bola e jogadores para voleibol.
- YOLO (ultralytics) com ByteTrack para IDs consistentes de jogadores.
- Kalman Filter para previsão da bola em frames perdidos (motion blur).
- Conversão pixel -> coordenadas de campo via homografia H.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import torch

from calibration import pixel_to_court
from config import config

# COCO classes relevantes
CLASS_PERSON = 0
CLASS_SPORTS_BALL = 32


@dataclass
class BallState:
    pixel: Tuple[float, float]
    court: Tuple[float, float]
    speed_px: float


class BallKalman:
    """Filtro de Kalman para posição (x, y) e velocidade (vx, vy) constante."""

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
        else:
            pred = self.kf.predict()
            return float(pred[0]), float(pred[1])


class VolleyballTracker:
    def __init__(self, H: np.ndarray, net_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path("best.pt")
        if not model_path.exists():
            model_path = Path(config.yolo_model)
        self.model = YOLO(str(model_path))
        try:
            self.model.to(self.device)
        except Exception:
            # fallback if model.to not available
            self.device = "cpu"
        self.H = H
        self.net_line = net_line
        self.ball_kf = BallKalman()
        self.trail: Deque[Tuple[int, int]] = deque(maxlen=config.max_trail)
        self.last_ball_detected = False
        self.frames_since_ball = 0
        self.ball_id = 1
        self.ball_missing = 0
        self.ball_last_det: Optional[Dict] = None

    def detect(self, frame) -> Dict:
        """
        Usa YOLO track (ByteTrack) para manter IDs de jogadores; recolhe deteção da bola.
        """
        results = self.model.track(
            source=frame,
            stream=False,
            persist=True,
            conf=0.15,  # mais permissivo com modelo especializado
            iou=config.iou_thresh,
            imgsz=1280,
            vid_stride=1,
            classes=[CLASS_PERSON, CLASS_SPORTS_BALL],
            device=self.device,
            verbose=False,
        )
        if not results:
            return {"players": [], "ball_det": None}
        res = results[0]

        players = []
        ball_det = None
        ball_candidates = []
        for box in res.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cls == CLASS_PERSON and float(box.conf) >= 0.40:
                tid = int(box.id.item()) if box.id is not None else -1
                players.append({"id": tid, "bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
            elif cls == CLASS_SPORTS_BALL and float(box.conf) >= 0.15:
                area = (x2 - x1) * (y2 - y1)
                if area > config.ball_max_area_px:
                    continue
                court_pt = pixel_to_court(self.H, (cx, cy))
                if not self._court_contains(court_pt):
                    continue
                det = {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "conf": float(box.conf), "area": area}
                ball_candidates.append(det)

        if ball_candidates:
            ball_det = max(ball_candidates, key=lambda b: b["conf"])
            ball_det["id"] = self.ball_id  # prioridade para classe bola
            self.ball_missing = 0
            self.ball_last_det = ball_det
        else:
            self.ball_missing += 1
            if self.ball_missing <= config.ball_max_age_frames and self.ball_last_det is not None:
                # mantém ID e última bbox/center por tolerância
                ball_det = self.ball_last_det
            else:
                self.ball_last_det = None

        return {"players": players, "ball_det": ball_det}



    def update_ball(self, ball_det: Optional[Dict]) -> BallState:
        meas = ball_det["center"] if ball_det is not None else None

        if meas is not None:
            x, y = self.ball_kf.update(meas)
            self.frames_since_ball = 0
            self.last_ball_detected = True
        else:
            if self.ball_kf.initialized:
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

        self.trail.append((int(x), int(y)))
        speed_px = self._instant_speed()
        court_xy = pixel_to_court(self.H, (x, y))
        return BallState(pixel=(x, y), court=court_xy, speed_px=speed_px)


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

    def acceleration(self) -> float:
        if len(self.trail) < 3:
            return 0.0
        v1 = np.array(self.trail[-2]) - np.array(self.trail[-3])
        v2 = np.array(self.trail[-1]) - np.array(self.trail[-2])
        return float(np.linalg.norm(v2 - v1))

    def horizontal_inversion(self) -> float:
        """Retorna cos do ângulo entre vetores consecutivos (inversão -> valor negativo)."""
        if len(self.trail) < 3:
            return 1.0
        v1 = np.array(self.trail[-2]) - np.array(self.trail[-3])
        v2 = np.array(self.trail[-1]) - np.array(self.trail[-2])
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 1.0
        return float(np.dot(v1, v2) / norm)

    def trail_points(self, last_n: Optional[int] = None) -> List[Tuple[int, int]]:
        pts = list(self.trail)
        if last_n is not None:
            return pts[-last_n:]
        return pts

    def _court_contains(self, court_pt: Tuple[float, float]) -> bool:
        margin = config.court_margin_m
        x, y = court_pt
        return -margin <= x <= 9 + margin and -margin <= y <= 18 + margin

    def _dist_to_net_line(self, pixel_pt: Tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.net_line
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return 1e9
        t = ((pixel_pt[0] - x1) * dx + (pixel_pt[1] - y1) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(np.hypot(pixel_pt[0] - proj_x, pixel_pt[1] - proj_y))

    def ball_near_net(self, pixel_pt: Tuple[float, float]) -> bool:
        return self._dist_to_net_line(pixel_pt) <= config.net_band_height_px

    def ball_on_net_line(self, pixel_pt: Tuple[float, float]) -> bool:
        return self._dist_to_net_line(pixel_pt) <= config.net_line_tolerance_px

    def crossed_net_line(self) -> bool:
        """Verifica se os dois Ãºltimos pontos do rasto estÃ£o em lados opostos da rede."""
        if len(self.trail) < 2:
            return False
        (x1, y1), (x2, y2) = self.net_line
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        side_prev = a * self.trail[-2][0] + b * self.trail[-2][1] + c
        side_cur = a * self.trail[-1][0] + b * self.trail[-1][1] + c
        return side_prev * side_cur < 0

    def predict_impact_point(self) -> Optional[Tuple[int, int]]:
        """
        Estima impacto no solo a partir de última direção conhecida (linha reta).
        Usa extrapolação simples; evita se rasto < 2.
        """
        if len(self.trail) < 2:
            return None
        (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
        dx, dy = x2 - x1, y2 - y1
        if dy <= 0:  # não descendente
            return None
        scale = (config.net_band_height_px * 20) / dy  # fator arbitrário para atingir chão
        return int(x2 + dx * scale), int(y2 + dy * scale)
