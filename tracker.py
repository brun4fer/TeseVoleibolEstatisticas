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
        self.model = YOLO(config.yolo_model)
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

    def detect(self, frame) -> Dict:
        """
        Usa YOLO track (ByteTrack) para manter IDs de jogadores; recolhe deteção da bola.
        """
        results = self.model.track(
            source=frame,
            stream=False,
            persist=True,
            conf=0.15,  # valor baixo para garantir bola; filtramos por classe abaixo
            iou=config.iou_thresh,
            device=self.device,
            verbose=False,
        )
        if not results:
            return {"players": [], "ball_det": None}
        res = results[0]

        players = []
        ball_det = None
        for box in res.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cls == CLASS_PERSON and float(box.conf) >= 0.40:
                tid = int(box.id.item()) if box.id is not None else -1
                players.append({"id": tid, "bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
            elif cls == CLASS_SPORTS_BALL and float(box.conf) >= 0.15:
                ball_det = {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "conf": float(box.conf)}

        return {"players": players, "ball_det": ball_det}

    def update_ball(self, ball_det: Optional[Dict]) -> BallState:
        meas = ball_det["center"] if ball_det is not None else None

        # Previsão manual curta se desaparecer <=10 frames usando último vetor
        if meas is None and 0 < self.frames_since_ball <= 10 and len(self.trail) >= 2:
            (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
            vx, vy = x2 - x1, y2 - y1
            x_pred, y_pred = x2 + vx, y2 + vy
            x, y = x_pred, y_pred
        else:
            x, y = self.ball_kf.update(meas)

        if ball_det is None:
            self.frames_since_ball += 1
            self.last_ball_detected = False
        else:
            self.frames_since_ball = 0
            self.last_ball_detected = True

        self.trail.append((int(x), int(y)))
        speed_px = self._instant_speed()
        court_xy = pixel_to_court(self.H, (x, y))
        return BallState(pixel=(x, y), court=court_xy, speed_px=speed_px)

    def _instant_speed(self) -> float:
        if len(self.trail) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
        return float(np.hypot(x2 - x1, y2 - y1))

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

    def trail_points(self) -> List[Tuple[int, int]]:
        return list(self.trail)

    def ball_near_net(self, pixel_pt: Tuple[float, float]) -> bool:
        (x1, y1), (x2, y2) = self.net_line
        net_y = (y1 + y2) / 2
        return abs(pixel_pt[1] - net_y) <= config.net_band_height_px

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
