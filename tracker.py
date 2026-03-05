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

from calibration import pixel_to_court
from config import config

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
        model_path = Path("best.pt")
        if not model_path.exists():
            model_path = Path(config.yolo_model)
        self.model = YOLO(str(model_path))
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.net_line = net_line
        self.ball_kf = BallKalman()
        self.trail: Deque[Tuple[int, int]] = deque(maxlen=config.max_trail)
        self.ball_drawer: List[Tuple[float, float, float]] = []
        self.ball_drawer_maxlen: int = 200
        self.ball_drawer_jump_px: float = 700.0
        self.ball_drawer_jump_after_occlusion_px: float = 700.0
        self.ball_drawer_occlusion_frames: int = 5
        self.ball_drawer_infill_min_missing_frames: int = 5
        self.ball_drawer_infill_max_missing_frames: int = 10
        self.ball_drawer_infill_net_dist_px: float = float(config.net_band_height_px) + 40.0
        self.ball_drawer_hold_s: float = 2.0
        self.ball_drawer_hold_until_ts: float = -1e9
        self.ball_drawer_clear_after_ts: Optional[float] = None
        self.ball_drawer_outlier_px: float = 220.0
        self.ball_drawer_outlier_bridge_px: float = 180.0
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

    def detect(self, frame) -> Dict:
        """Run YOLO tracking and return players + ball detection."""
        results = self.model.track(
            source=frame,
            stream=False,
            persist=True,
            conf=0.15,
            iou=config.iou_thresh,
            imgsz=1280,
            vid_stride=1,
            classes=[CLASS_PERSON, CLASS_SPORTS_BALL],
            device=self.device,
            verbose=False,
        )
        if not results:
            return {"players": [], "ball_det": self._ball_fallback_det()}
        res = results[0]

        players: List[Dict] = []
        ball_det: Optional[Dict] = None
        ball_candidates: List[Dict] = []
        for box in res.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if self._is_person_detection(cls, res) and float(box.conf) >= 0.40:
                tid = int(box.id.item()) if box.id is not None else -1
                players.append({"id": tid, "bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
            elif self._is_ball_detection(cls, res) and float(box.conf) >= 0.15:
                area = (x2 - x1) * (y2 - y1)
                if area > config.ball_max_area_px:
                    continue
                if abs(float(cx)) < 1e-6 and abs(float(cy)) < 1e-6:
                    continue
                court_pt = pixel_to_court(self.H, (cx, cy))
                if not self._court_contains(court_pt):
                    continue
                if not self._inside_field_roi((float(cx), float(cy))):
                    continue
                det = {
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": float(box.conf),
                    "area": area,
                    "visible": True,
                }
                ball_candidates.append(det)

        if ball_candidates:
            ball_det = max(ball_candidates, key=lambda b: b["conf"])
            now_t = float(time.time())
            self._infill_ball_drawer_gap_if_needed(float(ball_det["center"][0]), float(ball_det["center"][1]), now_t)
            self._append_ball_drawer(float(ball_det["center"][0]), float(ball_det["center"][1]), now_t)
            self.ball_missing = 0
            self.ball_last_det = dict(ball_det)
        else:
            ball_det = self._ball_fallback_det()

        return {"players": players, "ball_det": ball_det}

    def update_ball(self, ball_det: Optional[Dict], timestamp_s: Optional[float] = None) -> BallState:
        ball_visible = bool(ball_det is not None and bool(ball_det.get("visible", False)))
        meas = ball_det["center"] if ball_visible else None
        was_visible = self.last_ball_detected

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
            if timestamp_s is not None and was_visible:
                # Keep last detections available during short occlusions near block.
                self.ball_drawer_hold_until_ts = float(timestamp_s) + self.ball_drawer_hold_s

        self.trail.append((int(x), int(y)))
        if abs(float(x)) > 1e-6 or abs(float(y)) > 1e-6:
            # Dynamic possession must be tracked on every frame, including short occlusions.
            self.update_possession((float(x), float(y)))
        speed_px = self._instant_speed()
        court_xy = pixel_to_court(self.H, (x, y))
        vx, vy = self._last_velocity()
        return BallState(pixel=(x, y), court=court_xy, speed_px=speed_px, visible=ball_visible, vx=vx, vy=vy)

    def _append_ball_drawer(self, x: float, y: float, t: float) -> None:
        if abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6:
            return
        if not self._inside_field_roi((x, y)):
            return
        dist_to_net = self._dist_to_net_line((x, y))
        if dist_to_net > 400.0:
            # Ignore isolated one-frame detections far from net when there is no continuity.
            if not self.ball_drawer:
                print(f"[CLEANUP] Deteção descartada: Salto de {dist_to_net:.1f}px (Impossível).")
                return
            last_x, last_y, _last_t = self.ball_drawer[-1]
            continuity_dist = float(np.hypot(x - last_x, y - last_y))
            if continuity_dist > 220.0:
                print(f"[CLEANUP] Deteção descartada: Salto de {continuity_dist:.1f}px (Impossível).")
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
            del self.ball_drawer[:overflow]
        print(f"A adicionar à gaveta: {x}, {y}. Total agora: {len(self.ball_drawer)}")

    def _infill_ball_drawer_gap_if_needed(self, x: float, y: float, t: float) -> None:
        if not self.ball_drawer:
            return
        missing = int(self.frames_since_ball)
        if missing < self.ball_drawer_infill_min_missing_frames or missing > self.ball_drawer_infill_max_missing_frames:
            return

        last_x, last_y, last_t = self.ball_drawer[-1]
        dist_last_net = self._dist_to_net_line((float(last_x), float(last_y)))
        dist_now_net = self._dist_to_net_line((float(x), float(y)))
        if min(dist_last_net, dist_now_net) > self.ball_drawer_infill_net_dist_px:
            return

        if missing <= 1:
            return

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
            self.ball_drawer.append((ix, iy, it))
            added += 1

        if added > 0:
            self._prune_isolated_outlier()
            if len(self.ball_drawer) > self.ball_drawer_maxlen:
                overflow = len(self.ball_drawer) - self.ball_drawer_maxlen
                del self.ball_drawer[:overflow]
            print(f"[INFILL] Gap de {missing} frames junto a rede. Pontos interpolados: {added}.")

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
            # Return a compressed full-rally trail so visual debug keeps the whole rally on screen.
            idxs = np.linspace(0, len(pts) - 1, num=target_n, dtype=np.int32).tolist()
            return [pts[i] for i in idxs]
        return pts

    def clear_ball_drawer(self) -> None:
        self.ball_drawer.clear()
        self.ball_drawer_hold_until_ts = -1e9
        self.ball_drawer_clear_after_ts = None
        self.tocou_rede = False

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
            inside = cv2.pointPolygonTest(self.field_roi_polygon.astype(np.float32), (x, y), False) >= 0
            if not inside:
                return False
            if self.field_roi_top_y is not None and y < (self.field_roi_top_y - self.ceiling_margin_px):
                return False
            return True
        court_pt = pixel_to_court(self.H, (x, y))
        return self._court_contains(court_pt)

    def _build_field_roi_polygon(self) -> Optional[np.ndarray]:
        corners_court = ((0.0, 0.0), (9.0, 0.0), (9.0, 18.0), (0.0, 18.0))
        pts: List[Tuple[float, float]] = []
        for c in corners_court:
            px = self._court_to_pixel(c)
            if px is None:
                return None
            pts.append((float(px[0]), float(px[1])))
        return np.array(pts, dtype=np.float32)

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
        margin = config.court_margin_m
        x, y = court_pt
        return -margin <= x <= 9 + margin and -margin <= y <= 18 + margin

    def _side_from_pixel(self, pixel_pt: Tuple[float, float]) -> Optional[str]:
        s = self._signed_side(pixel_pt)
        if abs(s) < 1e-6:
            return self.current_ball_side
        return "CampoA" if s > 0 else "CampoB"

    def update_possession(self, pixel_pt: Tuple[float, float]) -> Optional[str]:
        side = self._side_from_pixel(pixel_pt)
        if side is None:
            return self.attacking_side

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

        # Attacker is frozen at net-intersection using last confirmed possession.
        if self.ball_near_net(pixel_pt):
            net_attacker = self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else previous_side
            if net_attacker in ("CampoA", "CampoB"):
                self.attacking_side = net_attacker
        elif self.attacking_side is None and self.posse_atual in ("CampoA", "CampoB"):
            self.attacking_side = self.posse_atual
        return self.attacking_side

    def _signed_side(self, pixel_pt: Tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.net_line
        return float((x2 - x1) * (pixel_pt[1] - y1) - (y2 - y1) * (pixel_pt[0] - x1))

    def _project_point_to_net(self, pixel_pt: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        (x1, y1), (x2, y2) = self.net_line
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return (float(x1), float(y1)), 1e9
        t = ((pixel_pt[0] - x1) * dx + (pixel_pt[1] - y1) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = float(np.hypot(pixel_pt[0] - proj_x, pixel_pt[1] - proj_y))
        return (float(proj_x), float(proj_y)), dist

    def _dist_to_net_line(self, pixel_pt: Tuple[float, float]) -> float:
        _, dist = self._project_point_to_net(pixel_pt)
        return dist

    def ball_near_net(self, pixel_pt: Tuple[float, float]) -> bool:
        return self._dist_to_net_line(pixel_pt) <= config.net_band_height_px

    def ball_on_net_line(self, pixel_pt: Tuple[float, float]) -> bool:
        return self._dist_to_net_line(pixel_pt) <= config.net_line_tolerance_px

    def crossed_net_line(self) -> bool:
        if len(self.trail) < 2:
            return False
        side_prev = self._signed_side(self.trail[-2])
        side_cur = self._signed_side(self.trail[-1])
        return side_prev * side_cur < 0

    def _segment_intersects_net(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance_px: float) -> bool:
        if self._dist_to_net_line(p1) <= tolerance_px or self._dist_to_net_line(p2) <= tolerance_px:
            return True
        s1 = self._signed_side(p1)
        s2 = self._signed_side(p2)
        return s1 * s2 < 0

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
        p = np.array([[court_pt[0]], [court_pt[1]], [1.0]], dtype=np.float32)
        proj = self.H_inv @ p
        if abs(float(proj[2])) < 1e-8:
            return None
        proj /= proj[2]
        return float(proj[0]), float(proj[1])

    def _meters_to_pixels_near_net(self, meters: float) -> float:
        if meters <= 0:
            return 0.0
        p1 = np.array(self.net_line[0], dtype=np.float32)
        p2 = np.array(self.net_line[1], dtype=np.float32)
        net_center_px = ((p1 + p2) / 2.0).tolist()
        net_center_court = pixel_to_court(self.H, (float(net_center_px[0]), float(net_center_px[1])))

        px_per_meter_samples: List[float] = []
        for dx, dy in ((meters, 0.0), (-meters, 0.0), (0.0, meters), (0.0, -meters)):
            c_pt = (net_center_court[0] + dx, net_center_court[1] + dy)
            if not self._court_contains(c_pt):
                continue
            p_pt = self._court_to_pixel(c_pt)
            if p_pt is None:
                continue
            dist_px = float(np.hypot(p_pt[0] - net_center_px[0], p_pt[1] - net_center_px[1]))
            px_per_meter_samples.append(dist_px / meters)

        if px_per_meter_samples:
            return float(np.median(px_per_meter_samples) * meters)

        net_len_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        approx_px_per_m = max(net_len_px / 9.0, 1.0)
        return approx_px_per_m * meters

    def predict_impact_point(self) -> Optional[Tuple[int, int]]:
        if len(self.trail) < 2:
            return None
        (x1, y1), (x2, y2) = self.trail[-2], self.trail[-1]
        dx, dy = x2 - x1, y2 - y1
        if dy <= 0:
            return None
        scale = (config.net_band_height_px * 20) / dy
        return int(x2 + dx * scale), int(y2 + dy * scale)
