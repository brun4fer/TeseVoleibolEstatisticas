"""
analytics.py
-------------
Rally event/statistics engine for volleyball.
- Detects rally start/end.
- Classifies Ace / Spike / Block / Error with physical heuristics.
- Uses OCR scoreboard changes to validate point end.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd

from config import config


# ----------------------------- OCR ----------------------------------
class ScoreboardOCR:
    def __init__(self):
        self.reader = easyocr.Reader(list(config.score_reader_lang), gpu=True)
        self.stable_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_count: int = 0
        self.old_formatted: Optional[str] = None

    def _extract_line(self, img) -> Tuple[int, int]:
        nums = re.findall(r"\d+", img)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return 0, int(nums[0])
        return 0, 0

    def read(self, frame) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = config.score_roi
        roi = frame[y : y + h, x : x + w]
        # Top line = Team A, bottom line = Team B
        mid_h = h // 2
        top = roi[0:mid_h, :]
        bot = roi[mid_h:h, :]

        def split_and_process(line_img):
            _, lw = line_img.shape[:2]
            split_x = int(lw * 0.3)  # 30% sets, 70% points
            zone_set = line_img[:, :split_x]
            zone_pts = line_img[:, split_x:]

            set_up = cv2.resize(zone_set, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            pts_up = cv2.resize(zone_pts, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

            set_gray = cv2.cvtColor(set_up, cv2.COLOR_BGR2GRAY)
            pts_gray = cv2.cvtColor(pts_up, cv2.COLOR_BGR2GRAY)

            set_bin = cv2.threshold(set_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            pts_bin = cv2.threshold(pts_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            set_bin = cv2.dilate(set_bin, k, iterations=1)
            pts_bin = cv2.dilate(pts_bin, k, iterations=1)

            combined = cv2.hconcat([set_bin, pts_bin])
            return set_bin, pts_bin, combined

        set_top, pts_top, comb_top = split_and_process(top)
        set_bot, pts_bot, comb_bot = split_and_process(bot)

        txt_set_top = "".join(self.reader.readtext(set_top, allowlist="0123456789", detail=0))
        txt_pts_top = "".join(self.reader.readtext(pts_top, allowlist="0123456789", detail=0))
        txt_set_bot = "".join(self.reader.readtext(set_bot, allowlist="0123456789", detail=0))
        txt_pts_bot = "".join(self.reader.readtext(pts_bot, allowlist="0123456789", detail=0))

        a_set = int(re.findall(r"\d+", txt_set_top)[0]) if re.findall(r"\d+", txt_set_top) else 0
        b_set = int(re.findall(r"\d+", txt_set_bot)[0]) if re.findall(r"\d+", txt_set_bot) else 0

        def extract_points(txt: str) -> int:
            nums = re.findall(r"\d+", txt)
            if not nums:
                return 0
            return int("".join(nums))

        a_pts = extract_points(txt_pts_top)
        b_pts = extract_points(txt_pts_bot)

        parsed = (a_set, a_pts, b_set, b_pts)

        h_dbg = max(comb_top.shape[0], comb_bot.shape[0])
        w_dbg = max(comb_top.shape[1], comb_bot.shape[1])
        canvas = 255 * np.ones((h_dbg * 2 + 3, w_dbg, 1), dtype=np.uint8)
        canvas[0:comb_top.shape[0], 0:comb_top.shape[1], 0] = comb_top
        canvas[h_dbg + 3 : h_dbg + 3 + comb_bot.shape[0], 0:comb_bot.shape[1], 0] = comb_bot
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.line(canvas_bgr, (0, h_dbg + 1), (w_dbg, h_dbg + 1), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), 0), (int(w_dbg * 0.3), h_dbg), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), h_dbg + 3), (int(w_dbg * 0.3), h_dbg + 3 + h_dbg), (0, 0, 255), 2)
        cv2.imshow("DEBUG_OCR", canvas_bgr)
        cv2.waitKey(1)

        if self.pending_score == parsed:
            self.pending_count += 1
        else:
            self.pending_score = parsed
            self.pending_count = 1

        if self.pending_count >= 3 and self.stable_score != parsed:
            self.stable_score = parsed
            set_num = parsed[0] + parsed[2] + 1
            formatted = f"SET {set_num} ({parsed[1]}-{parsed[3]})"
            if formatted != self.old_formatted:
                print(formatted)
                self.old_formatted = formatted

        return self.stable_score


# -------------------------- RALLY STATE -----------------------------
@dataclass
class Rally:
    rally_id: int
    start_ts: float
    start_frame: int
    end_ts: Optional[float] = None
    end_frame: Optional[int] = None
    winner_team: Optional[str] = None
    point_type: Optional[str] = None
    max_speed_px: float = 0.0
    ball_trail: List[Tuple[int, int]] = field(default_factory=list)
    impact_point: Optional[Tuple[float, float]] = None  # court coords
    ended_reason: Optional[str] = None


class RallyManager:
    def __init__(self):
        self.active: Optional[Rally] = None
        self.finished: List[Rally] = []
        self.next_id = 1

    def start_if_needed(self, condition: bool, ts: float, frame_idx: int, trail: List[Tuple[int, int]]):
        if condition and self.active is None:
            self.active = Rally(rally_id=self.next_id, start_ts=ts, start_frame=frame_idx, ball_trail=list(trail))
            self.next_id += 1

    def end(self, ts: float, frame_idx: int, winner: str, ptype: str, max_speed: float, impact: Optional[Tuple[float, float]], reason: str):
        if self.active is None:
            return
        self.active.end_ts = ts
        self.active.end_frame = frame_idx
        self.active.winner_team = winner
        self.active.point_type = ptype
        self.active.max_speed_px = max_speed
        self.active.impact_point = impact
        self.active.ended_reason = reason
        self.finished.append(self.active)
        self.active = None

    def dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.finished:
            if r.end_ts is None:
                continue
            rows.append(
                {
                    "rally_id": r.rally_id,
                    "start_timestamp": r.start_ts,
                    "end_timestamp": r.end_ts,
                    "duration": r.end_ts - r.start_ts,
                    "type_of_point": r.point_type,
                    "winner_team": r.winner_team,
                    "ball_speed_max": r.max_speed_px,
                }
            )
        return pd.DataFrame(rows)


# ------------------------ Heuristics -------------------------------
def is_service_lift(ball_history: List[Tuple[int, int]]) -> bool:
    if len(ball_history) < 3:
        return False
    y_values = [p[1] for p in ball_history[-3:]]
    rising = y_values[-1] < y_values[-3] - 8
    return rising


def ball_inside_court(court_pt: Tuple[float, float]) -> bool:
    x, y = court_pt
    return 0 <= x <= 9 and 0 <= y <= 18


def classify_point(
    max_speed_px: float,
    acceleration: float,
    cos_inversion: float,
    near_net: bool,
    on_net_line: bool,
    impact_in: bool,
    duration_s: float,
    contact_players: bool,
) -> str:
    # Block is no longer inferred here. It now comes from geometric net events in tracker.
    direction_flip = cos_inversion <= config.spike_dir_change_cos
    if duration_s <= config.ace_max_duration_s and impact_in and not contact_players:
        return "ACE"
    if direction_flip and near_net and not on_net_line:
        return "POINT_BY_SPIKE"
    if not impact_in:
        return "OPPONENT_ERROR"
    if max_speed_px >= config.spike_speed_thresh and acceleration > config.spike_speed_thresh and near_net:
        return "POINT_BY_SPIKE"
    return "UNKNOWN"


def min_distance_ball_players(ball_px: Tuple[float, float], players: List[Dict]) -> float:
    if not players:
        return 1e9
    b = np.array(ball_px)
    dists = []
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        dists.append(np.linalg.norm(center - b))
    return float(min(dists))


# --------------------------- Visual ---------------------------------
def draw_sidebar(frame, rally_mgr: RallyManager, counts: Dict[str, int], rally_count: int):
    _h, w, _ = frame.shape
    pad = 12
    panel_w = 240
    x0 = w - panel_w - pad
    y0 = pad
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 205), (0, 0, 0), -1)
    cv2.putText(frame, "Stats", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    lines = [
        f"Aces: {counts.get('ACE', 0)}",
        f"Spikes: {counts.get('Spikes', counts.get('POINT_BY_SPIKE', 0))}",
        f"Blocks: {counts.get('Blocks', counts.get('POINT_BY_BLOCK', 0))}",
        f"Bolas na Rede: {counts.get('Bolas na Rede', counts.get('BOLA_NA_REDE', 0))}",
        f"Errors: {counts.get('OPPONENT_ERROR', 0)}",
        f"Rallies: {rally_count}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0 + 10, y0 + 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------------- Main Engine -----------------------------
class AnalyticsEngine:
    def __init__(self):
        self.rally_mgr = RallyManager()
        self.ocr = ScoreboardOCR()
        self.counts = {
            "ACE": 0,
            "POINT_BY_SPIKE": 0,
            "POINT_BY_BLOCK": 0,
            "BOLA_NA_REDE": 0,
            "OPPONENT_ERROR": 0,
            "Spikes": 0,
            "Blocks": 0,
            "Bolas na Rede": 0,
        }
        self.prev_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_change: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_change_ts: Optional[float] = None
        self.post_mortem_wait_s: float = 0.5
        self.rally_counter: int = 0
        self.ball_drawer: List[Tuple[float, float, float]] = []
        self.ball_drawer_maxlen: int = 50
        self.current_side: Optional[str] = None  # CampoA | CampoB
        self.side_since_ts: Optional[float] = None
        self.possession_time: float = 0.0
        self.possession_team: Optional[str] = None
        self.serving_side: Optional[str] = None
        self.attacking_side: Optional[str] = None
        self.point_started: bool = False

        # Reactive timings for volleyball pace.
        self.possession_confirm_s = 0.3
        self.net_confirm_window_s = 7.0
        self.net_touch_tolerance_px = float(config.net_line_tolerance_px)
        self.net_event_cooldown_s = float(getattr(config, "net_event_cooldown_s", 0.35))
        self.occlusion_max_s = float(getattr(config, "net_occlusion_max_s", 0.9))
        self.occlusion_reappear_min_dist_px = float(getattr(config, "net_occlusion_reappear_min_dist_px", 6.0))
        self.same_side_rebound_max_frames = int(getattr(config, "same_side_rebound_max_frames", 4))
        self.vx_inversion_min = float(getattr(config, "vx_inversion_min", 1.0))
        self.net_height_m = float(getattr(config, "net_height_m", 2.43))
        self.block_height_ratio = float(getattr(config, "block_height_ratio", 0.80))

        self.pending_net_events: Deque[Dict] = deque(maxlen=20)
        self.last_net_event_ts: float = -1e9
        self.rebound_candidate: Optional[Dict] = None
        self.occlusion_candidate: Optional[Dict] = None
        self.prev_ball_visible: bool = False
        self.last_visible_ball: Optional[Dict] = None
        self.last_ball_exit_net_side: Optional[Dict] = None
        self.net_disappear_tolerance_px = 50.0
        self.net_ghost_hold_s = 1.0
        self.net_reid_window_s = 0.5
        self.hit_memory_window_s = 3.0
        self.force_block_window_s = 5.0
        self.last_net_touch_time: Optional[float] = None
        self.last_net_touch_attack_side: Optional[str] = None
        self.last_seen_side_b_time: Optional[float] = None
        self.prev_visible_in_block_zone: bool = False
        self.net_top_y: float = 0.0
        self.net_height_threshold: float = 0.0
        self.attacking_team: Optional[str] = None
        self.ball_in_zone_flag: bool = False
        self.last_valid_y: Optional[float] = None
        self.last_route_collision_to_net: bool = False
        self.last_route_collision_ts: Optional[float] = None
        self.prev_valid_ball_px: Optional[Tuple[float, float]] = None
        self.prev_prev_valid_ball_px: Optional[Tuple[float, float]] = None
        self.side_frame_streak: int = 0
        self.last_streak_side: Optional[str] = None
        self.possession_side_5frames: Optional[str] = None

        # 3) Kalman bridge: keep ball track alive through brief net occlusions.
        config.ball_max_age_frames = 15

        # Event State Machine (ID-independent).
        self.STATE_IDLE = "IDLE"
        self.STATE_ATAQUE = "ESTADO_ATAQUE"
        self.STATE_IMPACTO = "ESTADO_IMPACTO"
        self.STATE_REBOUCE = "ESTADO_REBOUCE"
        self.event_state: str = self.STATE_IDLE
        self.event_attack_side: Optional[str] = None
        self.event_attack_ts: Optional[float] = None
        self.event_impact_ts: Optional[float] = None
        self.event_rebouce_ts: Optional[float] = None
        self.prev_visible_side: Optional[str] = None
        self.prev_visible_dist_to_net: Optional[float] = None
        self.net_zone_half_width_px: float = float(getattr(config, "zone_block_half_width_px", 60))
        self.zone_block_below_px: float = float(getattr(config, "zone_block_below_px", 20))
        self.zone_block_above_px: float = float(getattr(config, "zone_block_above_px", 100))

    def _winner_from_score_change(self, new_score: Tuple[int, int, int, int]) -> str:
        if self.prev_score is None:
            return "Unknown"
        if new_score[1] > self.prev_score[1]:
            return "TeamA"
        if new_score[3] > self.prev_score[3]:
            return "TeamB"
        if new_score[0] > self.prev_score[0]:
            return "TeamA"
        if new_score[2] > self.prev_score[2]:
            return "TeamB"
        return "Unknown"

    def _team_for_side(self, side: str) -> str:
        return "TeamA" if side == "CampoA" else "TeamB"

    def _other_side(self, side: str) -> str:
        return "CampoB" if side == "CampoA" else "CampoA"

    def _side_for_team(self, team: str) -> Optional[str]:
        if team == "TeamA":
            return "CampoA"
        if team == "TeamB":
            return "CampoB"
        return None

    def _line_projection(self, ball_px: Tuple[float, float], tracker) -> Tuple[float, float, float]:
        (x1, y1), (x2, y2) = tracker.net_line
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return float(x1), float(y1), 1e9
        t = ((ball_px[0] - x1) * dx + (ball_px[1] - y1) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = float(np.hypot(ball_px[0] - proj_x, ball_px[1] - proj_y))
        return float(proj_x), float(proj_y), dist

    def _block_zone_rect(self, tracker) -> Tuple[int, int, int, int]:
        (x1, y1), (x2, y2) = tracker.net_line
        x_min = int(min(x1, x2) - self.net_zone_half_width_px)
        x_max = int(max(x1, x2) + self.net_zone_half_width_px)
        y_min = int(min(y1, y2) - self.zone_block_above_px)
        y_max = int(max(y1, y2) + self.zone_block_below_px)
        return x_min, y_min, x_max, y_max

    def _point_in_block_zone(self, ball_px: Tuple[float, float], tracker) -> bool:
        x_min, y_min, x_max, y_max = self._block_zone_rect(tracker)
        x, y = ball_px
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _segment_hits_block_zone(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        tracker,
    ) -> bool:
        x_min, y_min, x_max, y_max = self._block_zone_rect(tracker)
        steps = 8
        for i in range(steps + 1):
            t = i / float(steps)
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _draw_net_zone(self, frame, tracker) -> None:
        """Visual debug: draw ZONE_BLOCK."""
        x_min, y_min, x_max, y_max = self._block_zone_rect(tracker)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(frame, "ZONE_BLOCK", (x_min, max(20, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def _sync_ball_drawer(self, tracker) -> None:
        drawer = tracker.drawer_snapshot() if hasattr(tracker, "drawer_snapshot") else []
        synced: List[Tuple[float, float, float]] = []
        for p in drawer:
            if isinstance(p, dict):
                synced.append((float(p["x"]), float(p["y"]), float(p["t"])))
            else:
                synced.append((float(p[0]), float(p[1]), float(p[2])))
        self.ball_drawer = synced[-self.ball_drawer_maxlen :]
        tracker_attack_side = getattr(tracker, "attacking_side", None)
        if tracker_attack_side in ("CampoA", "CampoB"):
            self.attacking_side = tracker_attack_side
            self.attacking_team = self._team_for_side(tracker_attack_side)

    def _drawer_max_y_near_net(self, tracker, timestamp_s: float) -> Optional[float]:
        if not self.ball_drawer:
            return None
        recent = list(self.ball_drawer)
        ys: List[float] = []
        for x, y, _t in recent:
            _px, _py, dist_to_net = self._line_projection((x, y), tracker)
            if dist_to_net <= self.net_touch_tolerance_px or self._point_in_block_zone((x, y), tracker):
                ys.append(float(y))
        if not ys:
            return None
        return float(max(ys))

    def _drawer_peak_y_near_net(self, tracker) -> Optional[float]:
        if not self.ball_drawer:
            return None
        ys: List[float] = []
        for x, y, _t in self.ball_drawer:
            _px, _py, dist_to_net = self._line_projection((x, y), tracker)
            if dist_to_net <= self.net_touch_tolerance_px or self._point_in_block_zone((x, y), tracker):
                ys.append(float(y))
        if not ys:
            return None
        # y grows downwards; lower y means higher point.
        return float(min(ys))

    def _drawer_passed_zone_block(self, tracker) -> bool:
        if not self.ball_drawer:
            return False
        for x, y, _t in self.ball_drawer:
            if self._point_in_block_zone((x, y), tracker):
                return True
        return False

    def _drawer_crossed_net(self, tracker) -> bool:
        if len(self.ball_drawer) < 2:
            return False
        prev_side: Optional[str] = None
        for x, y, _t in self.ball_drawer:
            s = self._side_from_ball((x, y), tracker)
            if s is None:
                continue
            if prev_side is not None and s != prev_side:
                return True
            prev_side = s
        return False

    def _window_side_mean(self, pts: List[Tuple[float, float, float]], tracker, fallback: Optional[str] = None) -> Optional[str]:
        if not pts:
            return fallback
        signed_vals = [self._signed_side_value((float(p[0]), float(p[1])), tracker) for p in pts]
        mean_signed = float(np.mean(signed_vals))
        if abs(mean_signed) < 1e-6:
            return fallback
        return "CampoA" if mean_signed > 0 else "CampoB"

    def _side_from_x_position(self, x: float, tracker) -> Optional[str]:
        (x1, _y1), (x2, _y2) = tracker.net_line
        mid_x = (float(x1) + float(x2)) / 2.0
        return "CampoA" if float(x) <= mid_x else "CampoB"

    def check_post_mortem(self, tracker, end_side: Optional[str]) -> Tuple[str, Optional[str], Optional[str], bool]:
        print(f"[DEBUG] Analisando gaveta com {len(self.ball_drawer)} frames para decisão final...")
        try:
            if len(self.ball_drawer) < 2:
                return "ERROR", None, end_side, False

            x_start = float(self.ball_drawer[0][0])
            x_end = float(self.ball_drawer[-1][0])
            side_origin = self._side_from_x_position(x_start, tracker)
            side_dest = self._side_from_x_position(x_end, tracker)

            passed_zone_block = self._drawer_passed_zone_block(tracker)
            crossed_net = side_origin != side_dest or self._drawer_crossed_net(tracker)
            if abs(x_end - x_start) > 200.0 and passed_zone_block:
                crossed_net = True

            if crossed_net:
                return "SPIKE", side_origin, side_dest, True
            if side_origin is not None and side_dest is not None and side_origin == side_dest and passed_zone_block:
                return "BLOCK", side_origin, side_dest, False
            return "ERROR", side_origin, side_dest, False
        except Exception:
            return "ERROR", None, end_side, False

    def _post_mortem_block_from_drawer(self, tracker, timestamp_s: float) -> Tuple[bool, Optional[str], Optional[str]]:
        # Post-mortem over the full global drawer (last 50 detections).
        if not self.ball_drawer:
            return False, None, None
        recent = list(self.ball_drawer)
        if len(recent) < 2:
            return False, None, None

        first = recent[0]
        last = recent[-1]
        side_start = self._side_from_ball((first[0], first[1]), tracker)
        side_end = self._side_from_ball((last[0], last[1]), tracker)
        if side_start is None:
            side_start = self.attacking_side
        if side_end is None:
            side_end = self.prev_visible_side if self.prev_visible_side is not None else side_start

        max_y_near_net = self._drawer_max_y_near_net(tracker, timestamp_s)
        touch_near_net = max_y_near_net is not None
        same_side_rebound = side_start is not None and side_end is not None and side_start == side_end
        return bool(touch_near_net and same_side_rebound), side_start, side_end

    def _resolve_attack_side(self, fallback: Optional[str]) -> Optional[str]:
        if self.attacking_side is not None:
            return self.attacking_side
        # Fix attacker side from last stable possession before net disappearance.
        if self.possession_side_5frames is not None:
            return self.possession_side_5frames
        if fallback is not None:
            return fallback
        if self.prev_visible_side is not None:
            return self.prev_visible_side
        return self.current_side

    def _remember_net_touch(self, timestamp_s: float, attack_side: Optional[str]) -> None:
        resolved_attack_side = self._resolve_attack_side(attack_side)
        if resolved_attack_side is None:
            return
        self.last_net_touch_time = float(timestamp_s)
        self.last_net_touch_attack_side = resolved_attack_side
        self.attacking_team = self._team_for_side(resolved_attack_side)

    def _retroactive_block_from_hit_memory(self, winner: str, end_side: Optional[str], timestamp_s: float) -> bool:
        if self.last_net_touch_time is None or self.last_net_touch_attack_side is None:
            return False
        if timestamp_s - self.last_net_touch_time > self.hit_memory_window_s:
            return False
        if end_side is None:
            return False
        defending_side = self._other_side(self.last_net_touch_attack_side)
        defending_team = self._team_for_side(defending_side)
        if winner != defending_team:
            return False
        return end_side == self.last_net_touch_attack_side

    def _reset_event_state(self) -> None:
        self.event_state = self.STATE_IDLE
        self.event_attack_side = None
        self.event_attack_ts = None
        self.event_impact_ts = None
        self.event_rebouce_ts = None

    def _update_event_state_machine(
        self,
        ball_state,
        side: Optional[str],
        dist_to_net: float,
        timestamp_s: float,
        tracker,
        ball_px: Tuple[float, float],
        in_block_zone: bool,
    ) -> None:
        """
        FSM for block event independent of ball track ID:
        ATAQUE -> IMPACTO -> REBOUCE.
        """
        moving_towards_net = False
        if ball_state.visible and side == "CampoB" and self.prev_visible_side == "CampoB" and self.prev_visible_dist_to_net is not None:
            moving_towards_net = dist_to_net < (self.prev_visible_dist_to_net - 1.0)

        disappeared_now = (not ball_state.visible) and self.prev_ball_visible
        appeared_now = ball_state.visible and (not self.prev_ball_visible)

        if self.event_state == self.STATE_IDLE:
            if ball_state.visible and side == "CampoB" and (moving_towards_net or in_block_zone):
                self.event_state = self.STATE_ATAQUE
                self.event_attack_side = "CampoB"
                self.event_attack_ts = timestamp_s
                print("[STATE] Ataque detetado")

        elif self.event_state == self.STATE_ATAQUE:
            if self.event_attack_ts is not None and (timestamp_s - self.event_attack_ts) > 2.0:
                self._reset_event_state()
            else:
                if disappeared_now and self.prev_visible_side == "CampoB" and self.prev_visible_in_block_zone:
                    self.event_state = self.STATE_IMPACTO
                    self.event_impact_ts = timestamp_s
                    print("[STATE] Ataque detetado -> [STATE] Impacto na Rede")
                elif ball_state.visible and in_block_zone and side == "CampoB":
                    self.event_state = self.STATE_IMPACTO
                    self.event_impact_ts = timestamp_s
                    print("[STATE] Ataque detetado -> [STATE] Impacto na Rede")
                elif ball_state.visible and side == "CampoA":
                    # Attack crossed to other side; no same-side rebound block pattern.
                    self._reset_event_state()

        elif self.event_state == self.STATE_IMPACTO:
            if self.event_impact_ts is not None and (timestamp_s - self.event_impact_ts) > 2.0:
                self._reset_event_state()
            else:
                if ball_state.visible and side == "CampoB":
                    falling_on_attacker_side = ball_state.vy > 0
                    away_from_net = not in_block_zone
                    if falling_on_attacker_side or away_from_net or appeared_now:
                        self.event_state = self.STATE_REBOUCE
                        self.event_rebouce_ts = timestamp_s
                        print("[STATE] Impacto na Rede -> [STATE] Bola voltou para o atacante")
                        self._register_net_event(
                            event_type="POINT_BY_BLOCK",
                            attacking_side="CampoB",
                            impact_px=ball_px,
                            timestamp_s=timestamp_s,
                            reason="fsm_impact_rebouce",
                            potential_block=False,
                            pre_impact_px=ball_px,
                            tracker=tracker,
                            side_start="CampoB",
                            side_end="CampoB",
                        )

        elif self.event_state == self.STATE_REBOUCE:
            # Wait for OCR confirmation (<= net_confirm_window_s), then reset.
            if self.event_rebouce_ts is not None and (timestamp_s - self.event_rebouce_ts) > self.net_confirm_window_s:
                self._reset_event_state()

        if ball_state.visible:
            self.prev_visible_side = side
            self.prev_visible_dist_to_net = dist_to_net

    def _signed_side_value(self, ball_px: Tuple[float, float], tracker) -> float:
        (x1, y1), (x2, y2) = tracker.net_line
        return float((x2 - x1) * (ball_px[1] - y1) - (y2 - y1) * (ball_px[0] - x1))

    def _side_from_ball(self, ball_px: Tuple[float, float], tracker) -> Optional[str]:
        s = self._signed_side_value(ball_px, tracker)
        if abs(s) < 1e-6:
            return self.current_side
        return "CampoA" if s > 0 else "CampoB"

    def _update_possession(self, side: Optional[str], timestamp_s: float) -> None:
        if side is None:
            return
        if self.last_streak_side == side:
            self.side_frame_streak += 1
        else:
            self.last_streak_side = side
            self.side_frame_streak = 1
        if self.side_frame_streak >= 5:
            self.possession_side_5frames = side

        if self.current_side != side:
            self.current_side = side
            self.side_since_ts = timestamp_s
            self.possession_time = 0.0
            self.possession_team = None
            return
        if self.side_since_ts is None:
            self.side_since_ts = timestamp_s
        self.possession_time = max(0.0, timestamp_s - self.side_since_ts)
        if self.possession_time >= self.possession_confirm_s:
            self.possession_team = self._team_for_side(side)

    def _net_event_type_from_height(self, pre_impact_px: Tuple[float, float], tracker) -> str:
        """
        4) Height validation by calibration:
        block only if last pre-inversion point is above 80% of net height.
        """
        _, proj_y, _ = self._line_projection(pre_impact_px, tracker)
        y_below_top = max(0.0, float(pre_impact_px[1] - proj_y))

        # Estimate px/m near net using tracker geometric conversion when available.
        px_per_meter = None
        if hasattr(tracker, "block_height_margin_px") and hasattr(tracker, "block_height_margin_m"):
            margin_m = float(getattr(tracker, "block_height_margin_m"))
            margin_px = float(getattr(tracker, "block_height_margin_px"))
            if margin_m > 1e-6 and margin_px > 0:
                px_per_meter = margin_px / margin_m
        if px_per_meter is None:
            (x1, y1), (x2, y2) = tracker.net_line
            px_per_meter = max(float(np.hypot(x2 - x1, y2 - y1)) / 9.0, 1.0)

        net_height_px = px_per_meter * self.net_height_m
        min_block_zone_ratio = max(0.0, min(1.0, self.block_height_ratio))
        max_below_top_for_block = (1.0 - min_block_zone_ratio) * net_height_px
        if y_below_top <= max_below_top_for_block:
            return "POINT_BY_BLOCK"
        return "BOLA_NA_REDE"

    def _register_net_event(
        self,
        event_type: str,
        attacking_side: str,
        impact_px: Tuple[float, float],
        timestamp_s: float,
        reason: str,
        potential_block: bool = False,
        pre_impact_px: Optional[Tuple[float, float]] = None,
        tracker=None,
        side_start: Optional[str] = None,
        side_end: Optional[str] = None,
    ) -> None:
        if timestamp_s - self.last_net_event_ts < self.net_event_cooldown_s:
            return
        defending_side = self._other_side(attacking_side)
        attacking_team = self._team_for_side(attacking_side)
        if self.possession_team is not None and self.current_side == attacking_side:
            attacking_team = self.possession_team
        resolved_type = event_type
        if event_type == "POTENCIAL_BLOCO":
            resolved_type = "POINT_BY_BLOCK"
        elif potential_block and pre_impact_px is not None and tracker is not None:
            resolved_type = self._net_event_type_from_height(pre_impact_px, tracker)
        event = {
            "event_type": event_type,
            "resolved_type": resolved_type,
            "attacking_side": attacking_side,
            "defending_side": defending_side,
            "attacking_team": attacking_team,
            "defending_team": self._team_for_side(defending_side),
            "impact_px": (float(impact_px[0]), float(impact_px[1])),
            "pre_impact_px": pre_impact_px,
            "timestamp_s": float(timestamp_s),
            "reason": reason,
            "potential_block": potential_block,
            "side_start": side_start if side_start is not None else attacking_side,
            "side_end": side_end if side_end is not None else attacking_side,
        }
        self.pending_net_events.append(event)
        self.last_net_event_ts = timestamp_s

    def _prune_net_events(self, timestamp_s: float) -> None:
        while self.pending_net_events and timestamp_s - self.pending_net_events[0]["timestamp_s"] > self.net_confirm_window_s:
            self.pending_net_events.popleft()

    def _prune_ghost_buffer(self, timestamp_s: float) -> None:
        if self.last_ball_exit_net_side is None:
            return
        if timestamp_s - self.last_ball_exit_net_side["timestamp_s"] > self.net_ghost_hold_s:
            self.last_ball_exit_net_side = None

    def _update_ghost_disappearance(self, side: Optional[str], dist_to_net: float, timestamp_s: float, disappearance_now: bool) -> None:
        """
        1) Ghost buffer:
        if ANY ball disappears near net (<50px) from Campo B, store for 1s.
        """
        self._prune_ghost_buffer(timestamp_s)
        if not disappearance_now:
            return
        side_before = self.prev_visible_side if self.prev_visible_side is not None else side
        dist_before = self.prev_visible_dist_to_net if self.prev_visible_dist_to_net is not None else dist_to_net
        if side_before != "CampoB":
            return
        if dist_before > self.net_disappear_tolerance_px:
            return
        impact_px = (0.0, 0.0)
        if self.last_visible_ball is not None:
            impact_px = self.last_visible_ball["pixel"]
        self.last_ball_exit_net_side = {
            "side": "CampoB",
            "timestamp_s": float(timestamp_s),
            "impact_px": impact_px,
            "pre_impact_px": impact_px,
        }

    def _try_ghost_reidentification(self, side: Optional[str], ball_px: Tuple[float, float], timestamp_s: float, tracker, reappeared_now: bool) -> None:
        """
        2) Re-identification by side:
        if a NEW ball appears on Campo B within 0.5s after net disappearance, tag potential block.
        """
        self._prune_ghost_buffer(timestamp_s)
        if not reappeared_now:
            return
        if self.last_ball_exit_net_side is None:
            return
        if side != "CampoB":
            return
        dt = timestamp_s - self.last_ball_exit_net_side["timestamp_s"]
        if dt < 0 or dt > self.net_reid_window_s:
            return
        self._register_net_event(
            event_type="POTENCIAL_BLOCO",
            attacking_side="CampoB",
            impact_px=ball_px,
            timestamp_s=timestamp_s,
            reason="ghost_reid_same_side",
            potential_block=True,
            pre_impact_px=self.last_ball_exit_net_side["pre_impact_px"],
            tracker=tracker,
            side_start="CampoB",
            side_end=side,
        )
        self.last_ball_exit_net_side = None

    def _confirm_net_event(self, winner: str, timestamp_s: float) -> Optional[str]:
        self._prune_net_events(timestamp_s)
        candidates: List[Dict] = []
        for ev in reversed(self.pending_net_events):
            age = timestamp_s - ev["timestamp_s"]
            if age < 0 or age > self.net_confirm_window_s:
                continue
            candidates.append(ev)

        if not candidates:
            return None

        # 4) Priority: POINT_BY_BLOCK first, then other pending net events.
        candidates.sort(key=lambda e: 0 if e["resolved_type"] == "POINT_BY_BLOCK" else 1)

        for ev in candidates:
            ponto_confirmado = winner == ev["defending_team"]
            side_start = ev.get("side_start", ev["attacking_side"])
            side_end = ev.get("side_end", ev["attacking_side"])
            print(f"Evento: Bola na Rede | Lado Inicial: {side_start} | Lado Final: {side_end} | OCR_Ponto: {ponto_confirmado}")
            if ponto_confirmado:
                self.pending_net_events.clear()
                return ev["resolved_type"]
        return None

    def update_stats(self, baseline_ptype: str, winner: str, timestamp_s: float) -> str:
        """
        Resolve final point type with pending-net priority.
        POINT_BY_BLOCK has top priority over fallback labels (including OPPONENT_ERROR).
        """
        confirmed_net_ptype = self._confirm_net_event(winner, timestamp_s)
        if confirmed_net_ptype == "POINT_BY_BLOCK":
            return "POINT_BY_BLOCK"
        if confirmed_net_ptype is not None:
            return confirmed_net_ptype
        return baseline_ptype

    def _update_rebound_event(self, ball_state, ball_px: Tuple[float, float], side: Optional[str], dist_to_net: float, frame_idx: int, timestamp_s: float, tracker) -> None:
        self._prune_net_events(timestamp_s)

        if self.rebound_candidate is not None:
            age_frames = frame_idx - self.rebound_candidate["frame_idx"]
            same_side = side == self.rebound_candidate["incoming_side"]
            moved_back_to_side = dist_to_net > self.rebound_candidate["dist_to_net"] + self.occlusion_reappear_min_dist_px
            vx_before = float(self.rebound_candidate["incoming_vx"])
            vx_now = float(ball_state.vx)
            vx_inverted = abs(vx_before) >= self.vx_inversion_min and abs(vx_now) >= self.vx_inversion_min and (vx_before * vx_now < 0)
            if 1 <= age_frames <= self.same_side_rebound_max_frames and same_side and (moved_back_to_side or vx_inverted):
                self._register_net_event(
                    event_type="POTENCIAL_BLOCO",
                    attacking_side=self.rebound_candidate["incoming_side"],
                    impact_px=self.rebound_candidate["impact_px"],
                    timestamp_s=timestamp_s,
                    reason="rebound_same_side",
                    potential_block=True,
                    pre_impact_px=self.rebound_candidate["pre_impact_px"],
                    tracker=tracker,
                    side_start=self.rebound_candidate["incoming_side"],
                    side_end=side,
                )
                self.rebound_candidate = None
                return
            if age_frames > self.same_side_rebound_max_frames:
                self.rebound_candidate = None

        if side is not None and dist_to_net <= self.net_touch_tolerance_px:
            self.rebound_candidate = {
                "incoming_side": side,
                "impact_px": (float(ball_px[0]), float(ball_px[1])),
                "pre_impact_px": (float(ball_px[0]), float(ball_px[1])),
                "dist_to_net": float(dist_to_net),
                "incoming_vx": float(ball_state.vx),
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamp_s),
            }

    def _update_occlusion_event(self, ball_state, ball_px: Tuple[float, float], side: Optional[str], dist_to_net: float, timestamp_s: float, tracker) -> None:
        if ball_state.visible:
            if self.occlusion_candidate is not None:
                age = timestamp_s - self.occlusion_candidate["timestamp_s"]
                same_side = side == self.occlusion_candidate["side_before"]
                coming_from_net = dist_to_net > self.occlusion_candidate["dist_before"] + self.occlusion_reappear_min_dist_px
                if age <= self.occlusion_max_s and same_side and coming_from_net:
                    self._register_net_event(
                        event_type="POTENCIAL_BLOCO",
                        attacking_side=self.occlusion_candidate["side_before"],
                        impact_px=self.occlusion_candidate["impact_px"],
                        timestamp_s=timestamp_s,
                        reason="occlusion_same_side_return",
                        potential_block=True,
                        pre_impact_px=self.occlusion_candidate["pre_impact_px"],
                        tracker=tracker,
                        side_start=self.occlusion_candidate["side_before"],
                        side_end=side,
                    )
                self.occlusion_candidate = None

            self.last_visible_ball = {
                "side": side,
                "dist_to_net": float(dist_to_net),
                "pixel": (float(ball_px[0]), float(ball_px[1])),
                "timestamp_s": float(timestamp_s),
            }
            return

        if self.prev_ball_visible and self.last_visible_ball is not None:
            if self.last_visible_ball["dist_to_net"] <= self.net_touch_tolerance_px and self.last_visible_ball["side"] is not None:
                self.occlusion_candidate = {
                    "side_before": self.last_visible_ball["side"],
                    "dist_before": float(self.last_visible_ball["dist_to_net"]),
                    "impact_px": self.last_visible_ball["pixel"],
                    "pre_impact_px": self.last_visible_ball["pixel"],
                    "timestamp_s": float(timestamp_s),
                }
        if self.occlusion_candidate is not None and timestamp_s - self.occlusion_candidate["timestamp_s"] > self.occlusion_max_s:
            self.occlusion_candidate = None

    def process_frame(
        self,
        frame,
        frame_idx: int,
        timestamp_s: float,
        ball_state,
        players: List[Dict],
        tracker,
    ) -> Tuple[bool, Optional[str]]:
        """
        Update rally state and return (rally_finished, point_label).
        Stats are only recorded when OCR confirms a score change.
        """
        self._sync_ball_drawer(tracker)

        ball_px = (float(ball_state.pixel[0]), float(ball_state.pixel[1]))
        side = self.current_side
        _, _, dist_to_net = self._line_projection(ball_px, tracker)
        (net_x1, net_y1), (net_x2, net_y2) = tracker.net_line
        self.net_top_y = float(min(net_y1, net_y2))
        net_height_px = max(float(abs(net_y2 - net_y1)), 1.0)
        self.net_height_threshold = self.net_top_y + (1.0 - 0.80) * net_height_px
        in_block_zone = self._point_in_block_zone(ball_px, tracker)
        self._draw_net_zone(frame, tracker)
        disappeared_now = (not ball_state.visible) and self.prev_ball_visible

        if ball_state.visible and ball_px[1] > 0.0:
            self.last_valid_y = float(ball_px[1])
            self.prev_prev_valid_ball_px = self.prev_valid_ball_px
            self.prev_valid_ball_px = (float(ball_px[0]), float(ball_px[1]))

            route_collision = in_block_zone
            if self.prev_prev_valid_ball_px is not None:
                route_collision = route_collision or self._segment_hits_block_zone(
                    self.prev_prev_valid_ball_px,
                    self.prev_valid_ball_px,
                    tracker,
                )
            if self.prev_visible_dist_to_net is not None:
                moving_towards_net = dist_to_net < (self.prev_visible_dist_to_net - 1.0)
                route_collision = route_collision or (
                    moving_towards_net and dist_to_net <= (self.net_touch_tolerance_px + 40.0)
                )
            self.last_route_collision_to_net = bool(route_collision)
            if route_collision:
                self.last_route_collision_ts = float(timestamp_s)

        vetor_mesma_posse = False
        if len(self.ball_drawer) >= 2:
            lado0 = self._side_from_x_position(float(self.ball_drawer[0][0]), tracker)
            lado1 = self._side_from_x_position(float(self.ball_drawer[-1][0]), tracker)
            vetor_mesma_posse = lado0 is not None and lado1 is not None and lado0 == lado1
        cv2.putText(
            frame,
            f"[Vetor] Mesma Posse: {'Sim' if vetor_mesma_posse else 'Nao'}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # 1) Side tracking (for block memory, a single frame in Campo B is enough).
        if ball_state.visible:
            side = self._side_from_ball(ball_px, tracker)
            if side is not None and self.attacking_side is None:
                self.attacking_side = side
                self.attacking_team = self._team_for_side(side)

            # 1) Determine attacking side by exclusion from serving side at point start.
            if not self.point_started:
                inferred_service_side = side if side is not None else self.current_side
                if inferred_service_side is not None:
                    if hasattr(tracker, "reset_drawer_for_service"):
                        tracker.reset_drawer_for_service(float(ball_px[0]), float(ball_px[1]), float(timestamp_s))
                        self._sync_ball_drawer(tracker)
                    self.serving_side = inferred_service_side
                    self.point_started = True
                    # Attacking side is now dynamic and updated by tracker possession.
                    if self.attacking_side is None:
                        self.attacking_side = inferred_service_side
                        self.attacking_team = self._team_for_side(inferred_service_side)

            self._update_possession(side, timestamp_s)
            if side == "CampoB":
                self.last_seen_side_b_time = timestamp_s

            if in_block_zone:
                self.ball_in_zone_flag = True
                attack_side = self._resolve_attack_side(side)
                if attack_side is None and self.last_seen_side_b_time is not None and (timestamp_s - self.last_seen_side_b_time) <= self.hit_memory_window_s:
                    attack_side = "CampoB"
                self._remember_net_touch(timestamp_s, attack_side)

        # 2) Event State Machine transitions (ID-agnostic block detector).
        self._update_event_state_machine(ball_state, side, dist_to_net, timestamp_s, tracker, ball_px, in_block_zone)

        # Keep legacy ghost fallback for difficult occlusions (independent of track_id).
        if ball_state.visible:
            self._try_ghost_reidentification(side, ball_px, timestamp_s, tracker, reappeared_now=not self.prev_ball_visible)
        else:
            self._prune_net_events(timestamp_s)
            self._update_ghost_disappearance(side, dist_to_net, timestamp_s, disappearance_now=disappeared_now)
            route_collision_recent = (
                self.last_route_collision_ts is not None
                and (timestamp_s - self.last_route_collision_ts) <= 1.0
            )
            last_y_near_net = (
                self.last_valid_y is not None
                and abs(self.last_valid_y - self.net_top_y) <= self.net_touch_tolerance_px
            )
            if disappeared_now and (
                (self.prev_visible_dist_to_net is not None and self.prev_visible_dist_to_net <= 100.0)
                or self.prev_visible_in_block_zone
                or route_collision_recent
                or last_y_near_net
            ):
                force_attack_side = self._resolve_attack_side(self.prev_visible_side)
                if force_attack_side is None and self.last_seen_side_b_time is not None and (timestamp_s - self.last_seen_side_b_time) <= self.hit_memory_window_s:
                    force_attack_side = "CampoB"
                self._remember_net_touch(timestamp_s, force_attack_side)
                self.ball_in_zone_flag = True
            if disappeared_now and self.prev_visible_in_block_zone:
                attack_side = self._resolve_attack_side(self.prev_visible_side)
                if attack_side is None and self.last_seen_side_b_time is not None and (timestamp_s - self.last_seen_side_b_time) <= self.hit_memory_window_s:
                    attack_side = "CampoB"
                self._remember_net_touch(timestamp_s, attack_side)

        self.prev_ball_visible = bool(ball_state.visible)
        self.prev_visible_in_block_zone = in_block_zone

        if frame_idx % config.ocr_every_n_frames == 0:
            score = self.ocr.read(frame)
            if self.prev_score is None and score is not None:
                self.prev_score = score
            elif score is not None and self.prev_score is not None and score != self.prev_score:
                if self.pending_score_change is None or self.pending_score_change != score:
                    self.pending_score_change = score
                    self.pending_score_change_ts = timestamp_s

        rally_finished = False
        point_label = None

        min_dist = min_distance_ball_players(ball_state.pixel, players)
        contact_players = min_dist < 45

        if self.pending_score_change is not None:
            if self.rally_mgr.active is None:
                self.rally_mgr.start_if_needed(True, timestamp_s, frame_idx, tracker.trail_points())

            if self.pending_score_change_ts is None:
                self.pending_score_change_ts = timestamp_s
            if len(self.ball_drawer) <= 1 and (timestamp_s - self.pending_score_change_ts) < self.post_mortem_wait_s:
                return rally_finished, point_label

            winner = self._winner_from_score_change(self.pending_score_change)
            current_time = timestamp_s
            force_block_recent_touch = (
                self.last_net_touch_time is not None
                and (current_time - self.last_net_touch_time) <= self.force_block_window_s
            )
            x_inicio = float(self.ball_drawer[0][0]) if self.ball_drawer else float(ball_px[0])
            x_fim = float(self.ball_drawer[-1][0]) if self.ball_drawer else float(ball_px[0])
            lado_origem = self._side_from_x_position(x_inicio, tracker)
            lado_destino = self._side_from_x_position(x_fim, tracker)

            # Keep attacker side/team synchronized with trajectory origin.
            self.attacking_side = lado_origem
            equipe_atacante_real = self._team_for_side(lado_origem) if lado_origem is not None else "Unknown"
            if equipe_atacante_real != "Unknown":
                self.attacking_team = equipe_atacante_real

            passed_zone_block = self._drawer_passed_zone_block(tracker)
            touched_net = passed_zone_block or self.ball_in_zone_flag or force_block_recent_touch
            same_side = lado_origem is not None and lado_destino is not None and (lado_origem == lado_destino)
            crossed_side = lado_origem is not None and lado_destino is not None and (lado_origem != lado_destino)

            decision_result = "ERROR"
            ptype = "OPPONENT_ERROR"

            # Absolute side-based decision logic (highest priority on score change).
            if crossed_side:
                ptype = "POINT_BY_SPIKE"
                decision_result = "SPIKE"
            elif same_side and touched_net:
                ptype = "POINT_BY_BLOCK"
                decision_result = "BLOCK"
            else:
                ptype = "OPPONENT_ERROR"
                decision_result = "ERROR"

            print(
                f"[DECISION] Lado Inicio: {lado_origem} | "
                f"Lado Fim: {lado_destino} | Impacto Rede: {touched_net} -> "
                f"Resultado Final: {decision_result}."
            )

            self.prev_score = self.pending_score_change
            self.pending_score_change = None
            self.pending_score_change_ts = None
            self.rally_counter += 1

            # End-of-rally cleanup for next possession/event cycle.
            self.pending_net_events.clear()
            self.rebound_candidate = None
            self.occlusion_candidate = None
            self.last_visible_ball = None
            self.last_ball_exit_net_side = None
            self.last_net_touch_time = None
            self.last_net_touch_attack_side = None
            self.attacking_team = None
            self.ball_in_zone_flag = False
            self.last_valid_y = None
            self.last_route_collision_to_net = False
            self.last_route_collision_ts = None
            self.prev_valid_ball_px = None
            self.prev_prev_valid_ball_px = None
            self.side_frame_streak = 0
            self.last_streak_side = None
            self.possession_side_5frames = None
            self.last_seen_side_b_time = None
            self.prev_visible_side = None
            self.prev_visible_dist_to_net = None
            self.prev_visible_in_block_zone = False
            self._reset_event_state()
            self.prev_ball_visible = False
            self.point_started = False
            self.serving_side = None
            self.attacking_side = None
            self.ball_drawer = []
            if hasattr(tracker, "clear_ball_drawer"):
                tracker.clear_ball_drawer()

            self.rally_mgr.end(
                ts=timestamp_s,
                frame_idx=frame_idx,
                winner=winner,
                ptype=ptype,
                max_speed=tracker._instant_speed(),
                impact=ball_state.court,
                reason="score_change",
            )
            if ptype in self.counts:
                self.counts[ptype] += 1
            if ptype == "POINT_BY_SPIKE":
                self.counts["Spikes"] += 1
            if ptype == "POINT_BY_BLOCK":
                self.counts["Blocks"] += 1
            elif ptype == "BOLA_NA_REDE":
                self.counts["Bolas na Rede"] += 1
            rally_finished = True
            point_label = ptype

        return rally_finished, point_label

