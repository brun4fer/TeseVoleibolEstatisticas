"""
analytics.py
-------------
Rally event/statistics engine for volleyball.
- Detects rally start/end.
- Classifies Ace / Spike / Block / Error with physical heuristics.
- Uses OCR scoreboard changes to validate point end.
"""

from __future__ import annotations

from collections import Counter, deque
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
        self.last_raw_score: Optional[Tuple[int, int, int, int]] = None
        self.last_returned_score: Optional[Tuple[int, int, int, int]] = None
        self.last_read_discarded: bool = False
        self.vote_window: Deque[Tuple[int, int, int, int]] = deque(maxlen=5)
        self.vote_min_hits: int = 3
        self.old_formatted: Optional[str] = None

    def _extreme_binary(self, img, invert: bool = False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        out = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        return out

    def _ocr_digits_only(self, img) -> str:
        txt = "".join(self.reader.readtext(img, allowlist="0123456789", detail=0))
        nums = re.findall(r"\d+", txt)
        return "".join(nums) if nums else ""

    def _to_int(self, txt: str, default: int = 0) -> int:
        if not txt:
            return default
        try:
            return int(txt)
        except ValueError:
            return default

    def _is_plausible_step(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Tuple[int, int, int, int],
    ) -> bool:
        na_set, na_pts, nb_set, nb_pts = new_score

        # Hard sanity limits to reject OCR explosions such as 32-6.
        if not (0 <= na_set <= 5 and 0 <= nb_set <= 5 and 0 <= na_pts <= 60 and 0 <= nb_pts <= 60):
            return False

        if prev_score is None:
            return True
        if new_score == prev_score:
            return True

        pa_set, pa_pts, pb_set, pb_pts = prev_score

        # Priority rule: prefer single logical digit progression.
        if na_set == pa_set and nb_set == pb_set:
            da = na_pts - pa_pts
            db = nb_pts - pb_pts
            return (da == 1 and db == 0) or (da == 0 and db == 1)

        # Allow set increment transition with points reset/near reset.
        if na_set == pa_set + 1 and nb_set == pb_set and na_pts <= 2 and nb_pts <= 2:
            return True
        if nb_set == pb_set + 1 and na_set == pa_set and na_pts <= 2 and nb_pts <= 2:
            return True
        return False

    def read(self, frame) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = config.score_roi
        roi = frame[y : y + h, x : x + w]
        if not getattr(config, "HEADLESS_MODE", False):
            dbg = frame.copy()
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("DEBUG_OCR_ROI", dbg)

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

            # Extreme contrast pass to remove background and highlight digits.
            set_bin = self._extreme_binary(set_up, invert=True)
            pts_bin = self._extreme_binary(pts_up, invert=True)

            combined = cv2.hconcat([set_bin, pts_bin])
            return set_bin, pts_bin, combined

        set_top, pts_top, comb_top = split_and_process(top)
        set_bot, pts_bot, comb_bot = split_and_process(bot)

        txt_set_top = self._ocr_digits_only(set_top)
        txt_pts_top = self._ocr_digits_only(pts_top)
        txt_set_bot = self._ocr_digits_only(set_bot)
        txt_pts_bot = self._ocr_digits_only(pts_bot)

        a_set = self._to_int(txt_set_top[:1], default=0)
        b_set = self._to_int(txt_set_bot[:1], default=0)
        a_pts = self._to_int(txt_pts_top[:2], default=0)
        b_pts = self._to_int(txt_pts_bot[:2], default=0)

        parsed_raw = (a_set, a_pts, b_set, b_pts)
        self.last_raw_score = parsed_raw
        parsed = parsed_raw
        discard_candidate = False
        if not self._is_plausible_step(self.stable_score, parsed):
            # Keep previous stable reading when OCR suggests impossible transitions.
            if self.stable_score is not None:
                parsed = self.stable_score
            else:
                discard_candidate = True
        self.last_read_discarded = bool(discard_candidate)

        h_dbg = max(comb_top.shape[0], comb_bot.shape[0])
        w_dbg = max(comb_top.shape[1], comb_bot.shape[1])
        canvas = 255 * np.ones((h_dbg * 2 + 3, w_dbg, 1), dtype=np.uint8)
        canvas[0:comb_top.shape[0], 0:comb_top.shape[1], 0] = comb_top
        canvas[h_dbg + 3 : h_dbg + 3 + comb_bot.shape[0], 0:comb_bot.shape[1], 0] = comb_bot
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.line(canvas_bgr, (0, h_dbg + 1), (w_dbg, h_dbg + 1), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), 0), (int(w_dbg * 0.3), h_dbg), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), h_dbg + 3), (int(w_dbg * 0.3), h_dbg + 3 + h_dbg), (0, 0, 255), 2)
        if not getattr(config, "HEADLESS_MODE", False):
            cv2.imshow("DEBUG_OCR", canvas_bgr)
            cv2.waitKey(1)

        if not discard_candidate:
            self.vote_window.append(parsed)
        if len(self.vote_window) == self.vote_window.maxlen:
            most_common_score, hits = Counter(self.vote_window).most_common(1)[0]
            if hits >= self.vote_min_hits and self.stable_score != most_common_score:
                self.stable_score = most_common_score
                set_num = most_common_score[0] + most_common_score[2] + 1
                formatted = f"SET {set_num} ({most_common_score[1]}-{most_common_score[3]})"
                if formatted != self.old_formatted:
                    print(formatted)
                    self.old_formatted = formatted

        self.last_returned_score = self.stable_score
        return self.stable_score


# -------------------------- RALLY STATE -----------------------------
@dataclass
class Rally:
    id: int
    rally_id: int
    start_ts: float
    start_frame: int
    end_ts: Optional[float] = None
    end_frame: Optional[int] = None
    duration: float = 0.0
    attacker: Optional[str] = None
    winner_team: Optional[str] = None
    point_type: Optional[str] = None
    net_crossings: int = 0
    ball_speed_max: float = 0.0
    ball_speed_mean: float = 0.0
    trajectory: List[Tuple[int, float, float]] = field(default_factory=list)
    # Backward compatibility fields
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
            self.active = Rally(
                id=self.next_id,
                rally_id=self.next_id,
                start_ts=ts,
                start_frame=frame_idx,
                ball_trail=list(trail),
            )
            self.next_id += 1

    def end(
        self,
        ts: float,
        frame_idx: int,
        winner: str,
        ptype: str,
        max_speed: float,
        impact: Optional[Tuple[float, float]],
        reason: str,
        attacker: Optional[str] = None,
        net_crossings: int = 0,
        ball_speed_mean: float = 0.0,
        trajectory: Optional[List[Tuple[int, float, float]]] = None,
    ):
        if self.active is None:
            return
        self.active.end_ts = ts
        self.active.end_frame = frame_idx
        self.active.duration = max(0.0, float(ts - self.active.start_ts))
        self.active.attacker = attacker
        self.active.winner_team = winner
        self.active.point_type = ptype
        self.active.net_crossings = int(net_crossings)
        self.active.ball_speed_max = float(max_speed)
        self.active.ball_speed_mean = float(ball_speed_mean)
        self.active.trajectory = list(trajectory) if trajectory is not None else []
        # Backward compatibility aliases
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
                    "rally_duration": r.duration if r.duration > 0 else (r.end_ts - r.start_ts),
                    "attacker": r.attacker,
                    "type_of_point": r.point_type,
                    "winner_team": r.winner_team,
                    "net_crossings": r.net_crossings,
                    "ball_speed_mean": r.ball_speed_mean,
                    "ball_speed_max": r.ball_speed_max if r.ball_speed_max > 0 else r.max_speed_px,
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
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 235), (0, 0, 0), -1)
    cv2.putText(frame, "Stats", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    lines = [
        f"Aces: {counts.get('ACE', 0)}",
        f"Spikes: {counts.get('Spikes', counts.get('POINT_BY_SPIKE', 0))}",
        f"Blocks: {counts.get('Blocks', counts.get('POINT_BY_BLOCK', 0))}",
        f"Freeballs: {counts.get('Freeballs', counts.get('FREEBALL', 0))}",
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
            "FREEBALL": 0,
            "BOLA_NA_REDE": 0,
            "OPPONENT_ERROR": 0,
            "Spikes": 0,
            "Blocks": 0,
            "Freeballs": 0,
            "Bolas na Rede": 0,
            "Rallies": 0,
        }
        self.prev_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_change: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_change_ts: Optional[float] = None
        self.pending_score_drawer_snapshot: Optional[List[Tuple[float, float, float]]] = None
        self.pending_score_prev_base: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_forced: bool = False
        self.invalid_ocr_candidate: Optional[Tuple[int, int, int, int]] = None
        self.invalid_ocr_retry_limit: int = 4
        self.ocr_lock: Dict[str, Optional[Dict[str, int]]] = {
            "teamA": None,
            "teamB": None,
        }
        self.valor_bloqueado_ocr: Optional[Tuple[int, int, int, int]] = None
        self.valor_recuperacao_ocr: Optional[Tuple[int, int, int, int]] = None
        self.valor_bloqueado_ocr_until: float = -1e9
        self.high_jump_candidate: Optional[Tuple[int, int, int, int]] = None
        self.high_jump_count: int = 0
        self.high_jump_min_persist: int = 3
        self.ocr_correction_candidate: Optional[Tuple[int, int, int, int]] = None
        self.ocr_correction_counter: int = 0
        self.ocr_correction_min_persist: int = 3
        self.ocr_correction_gap_points: int = 5
        self.tentativas_ocr: int = 0
        self.last_point_time: float = -1e9
        self.last_point_frame: int = -1_000_000
        self.min_frames_between_points: int = 30
        self.point_cooldown_s: float = 10.0
        self.ocr_blocked_until: float = -1e9
        self.ocr_stabilization_lock_s: float = 5.0
        self.ball_dead_confirm_s: float = 1.0
        self.min_frames_for_technical_stats: int = 5
        self.ball_missing_since: Optional[float] = None
        self.ball_ground_stable_since: Optional[float] = None
        self.last_ground_y: Optional[float] = None
        self.ground_zone_ratio: float = 0.82
        self.ground_stable_delta_px: float = 4.0
        self.point_finalized: bool = False
        self.post_mortem_wait_s: float = 0.5
        self.rally_counter: int = 0
        self.ball_drawer: List[Tuple[float, float, float]] = []
        self.ball_drawer_maxlen: int = 200
        self.current_side: Optional[str] = None  # CampoA | CampoB
        self.current_possession: Optional[str] = None  # CampoA | CampoB (tracker continuous possession)
        self.campo_posse_atual: Optional[str] = None  # CampoA | CampoB (global possession)
        self.posse_atual: Optional[str] = None  # CampoA | CampoB (persistent possession)
        self.last_attacker_before_net: Optional[str] = None  # CampoA | CampoB
        self.last_ball_seen_ts: Optional[float] = None
        self.side_since_ts: Optional[float] = None
        self.possession_time: float = 0.0
        self.possession_team: Optional[str] = None
        self.serving_side: Optional[str] = None
        self.last_service_side: Optional[str] = None
        self.attacking_side: Optional[str] = None
        self.point_started: bool = False
        self.service_detection_locked: bool = False
        self.last_ocr_score_stable: Optional[Tuple[int, int, int, int]] = None
        self.last_ocr_score_raw: Optional[Tuple[int, int, int, int]] = None
        self.score_just_updated: bool = False

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
        self.tocou_rede: bool = False
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
        self.rally_closed_by_scoreboard: bool = True
        self.net_buffer_px: float = float(getattr(config, "net_buffer_px", 15.0))
        self.net_cross_confirm_frames: int = int(getattr(config, "net_cross_confirm_frames", 3))
        self.spike_speed_threshold_px: float = float(getattr(config, "spike_speed_threshold_px", 8.0))
        self.current_rally_trajectory: List[Tuple[int, float, float]] = []
        self.rally_crossings: int = 0
        self.crossing_candidate_side: Optional[str] = None
        self.crossing_candidate_frames: int = 0
        self.crossing_confirmed_side: Optional[str] = None
        self.RALLY_STATE_ACTIVE: str = "ACTIVE"
        self.RALLY_ENDED_VISUAL: str = "RALLY_ENDED_VISUAL"
        self.RALLY_CONFIRMED_OCR: str = "RALLY_CONFIRMED_OCR"
        self.visual_end_state: str = self.RALLY_STATE_ACTIVE
        self.visual_end_since_ts: Optional[float] = None

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
        self.estado_retorno_detectado: bool = False
        self.prev_visible_side: Optional[str] = None
        self.prev_visible_dist_to_net: Optional[float] = None
        self.net_zone_half_width_px: float = float(getattr(config, "zone_block_half_width_px", 60))
        self.zone_block_below_px: float = float(getattr(config, "zone_block_below_px", 20))
        self.zone_block_above_px: float = float(getattr(config, "zone_block_above_px", 100))
        self.net_proximity_threshold: float = 150.0
        self.ghost_attack_away_delta_px: float = 8.0
        self.ghost_attack_vx_thresh_px: float = 12.0

    @property
    def toucou_rede(self) -> bool:
        return bool(self.tocou_rede)

    @toucou_rede.setter
    def toucou_rede(self, value: bool) -> None:
        self.tocou_rede = bool(value)

    def _winner_from_score_change(
        self,
        new_score: Tuple[int, int, int, int],
        prev_score: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        base_prev = self.prev_score if prev_score is None else prev_score
        if base_prev is None:
            return "Unknown"
        if new_score[1] > base_prev[1]:
            return "TeamA"
        if new_score[3] > base_prev[3]:
            return "TeamB"
        if new_score[0] > base_prev[0]:
            return "TeamA"
        if new_score[2] > base_prev[2]:
            return "TeamB"
        return "Unknown"

    def _is_logical_score_change(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        if prev_score is None or new_score is None:
            return False
        if new_score == prev_score:
            return False

        prev_a_set, prev_a_pts, prev_b_set, prev_b_pts = prev_score
        new_a_set, new_a_pts, new_b_set, new_b_pts = new_score

        # Strict +1 filter for rally ending:
        # same set, one team +1 point, the other unchanged.
        if prev_a_set != new_a_set or prev_b_set != new_b_set:
            return False
        if new_a_pts < prev_a_pts or new_b_pts < prev_b_pts:
            return False
        d_a = new_a_pts - prev_a_pts
        d_b = new_b_pts - prev_b_pts
        return (d_a == 1 and d_b == 0) or (d_a == 0 and d_b == 1)

    def _is_score_lower_than_current(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        if prev_score is None or new_score is None:
            return False
        return bool(
            new_score[0] < prev_score[0]
            or new_score[2] < prev_score[2]
            or new_score[1] < prev_score[1]
            or new_score[3] < prev_score[3]
        )

    def _is_score_jump_above_plus_one(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        if prev_score is None or new_score is None:
            return False
        if self._is_logical_score_change(prev_score, new_score):
            return False
        # Any non-lower and non-equal transition is considered a jump/noisy progression.
        if new_score == prev_score:
            return False
        if self._is_score_lower_than_current(prev_score, new_score):
            return False
        return True

    def failsafe_score_update(
        self,
        official_score: Optional[Tuple[int, int, int, int]],
        ocr_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Failsafe for impossible OCR regressions:
        - score never decreases
        - one team scores +1 per rally
        """
        if official_score is None or ocr_score is None:
            return None
        if not self._is_score_lower_than_current(official_score, ocr_score):
            return None

        off_a_set, off_a_pts, off_b_set, off_b_pts = official_score
        ocr_a_set, ocr_a_pts, ocr_b_set, ocr_b_pts = ocr_score
        if ocr_a_set != off_a_set or ocr_b_set != off_b_set:
            return None

        drop_a = int(off_a_pts) - int(ocr_a_pts)
        drop_b = int(off_b_pts) - int(ocr_b_pts)

        winner_team: Optional[str] = None
        # If one side dropped in OCR, assume that side actually scored +1.
        if drop_a > 0 and drop_b <= 0:
            winner_team = "TeamA"
        elif drop_b > 0 and drop_a <= 0:
            winner_team = "TeamB"
        elif drop_a > 0 and drop_b > 0:
            if drop_a > drop_b:
                winner_team = "TeamA"
            elif drop_b > drop_a:
                winner_team = "TeamB"

        if winner_team is None:
            return None

        inferred = self._score_plus_one(official_score, winner_team)
        if not self._is_logical_score_change(official_score, inferred):
            return None

        print(f"[FAILSAFE] inferred score: {inferred[1]}-{inferred[3]}")
        return inferred

    def _reset_ocr_correction_tracking(self) -> None:
        self.ocr_correction_candidate = None
        self.ocr_correction_counter = 0

    def _build_large_downward_correction(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Recovery rule for bad initial OCR lock:
        if official points are much larger than repeated OCR points, allow correction.
        """
        if prev_score is None or new_score is None:
            return None

        pa_set, pa_pts, pb_set, pb_pts = prev_score
        na_set, na_pts, nb_set, nb_pts = new_score
        if na_set != pa_set or nb_set != pb_set:
            return None

        gap = int(self.ocr_correction_gap_points)
        candidate: Optional[Tuple[int, int, int, int]] = None
        if (pa_pts - na_pts) >= gap and nb_pts == pb_pts:
            candidate = (pa_set, na_pts, pb_set, pb_pts)
        if (pb_pts - nb_pts) >= gap and na_pts == pa_pts:
            c2 = (pa_set, pa_pts, pb_set, nb_pts)
            if candidate is None:
                candidate = c2
            elif candidate != c2:
                return None
        return candidate

    def _update_ocr_correction_counter(
        self, candidate: Tuple[int, int, int, int]
    ) -> int:
        if self.ocr_correction_candidate == candidate:
            self.ocr_correction_counter += 1
        else:
            self.ocr_correction_candidate = candidate
            self.ocr_correction_counter = 1
        return int(self.ocr_correction_counter)

    def _partial_plus_one_score_update(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        ocr_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Accept side-wise +1 updates when OCR partially misreads the opposite side.
        Example: official (0,5,0,2), OCR (0,6,0,0) -> accept (0,6,0,2).
        """
        if prev_score is None or ocr_score is None:
            return None
        if ocr_score == prev_score:
            return None

        prev_a_set, prev_a_pts, prev_b_set, prev_b_pts = prev_score
        _ocr_a_set, ocr_a_pts, _ocr_b_set, ocr_b_pts = ocr_score

        team_a_plus_one = int(ocr_a_pts) == int(prev_a_pts) + 1
        team_b_plus_one = int(ocr_b_pts) == int(prev_b_pts) + 1

        # Volleyball point progression should only increment one side at a time.
        if team_a_plus_one and team_b_plus_one:
            return None
        if not team_a_plus_one and not team_b_plus_one:
            return None

        new_a_pts = int(prev_a_pts) + (1 if team_a_plus_one else 0)
        new_b_pts = int(prev_b_pts) + (1 if team_b_plus_one else 0)
        return (int(prev_a_set), int(new_a_pts), int(prev_b_set), int(new_b_pts))

    def _team_for_side(self, side: str) -> str:
        return "TeamA" if side == "CampoA" else "TeamB"

    def _other_side(self, side: str) -> str:
        return "CampoB" if side == "CampoA" else "CampoA"

    def _score_plus_one(
        self,
        prev_score: Tuple[int, int, int, int],
        winner_team: str,
    ) -> Tuple[int, int, int, int]:
        a_set, a_pts, b_set, b_pts = prev_score
        if winner_team == "TeamA":
            return (a_set, a_pts + 1, b_set, b_pts)
        if winner_team == "TeamB":
            return (a_set, a_pts, b_set, b_pts + 1)
        return prev_score

    def _clear_ocr_digit_lock(self) -> None:
        self.ocr_lock["teamA"] = None
        self.ocr_lock["teamB"] = None

    def _apply_ocr_digit_lock(
        self,
        score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if score is None:
            return None

        a_set, a_pts, b_set, b_pts = map(int, score)
        lock_a = self.ocr_lock.get("teamA")
        lock_b = self.ocr_lock.get("teamB")

        if lock_a is not None and a_pts == int(lock_a["raw"]):
            a_pts = int(lock_a["correct"])
        if lock_b is not None and b_pts == int(lock_b["raw"]):
            b_pts = int(lock_b["correct"])

        return (a_set, a_pts, b_set, b_pts)

    def _set_ocr_digit_lock_from_failsafe(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        raw_score: Optional[Tuple[int, int, int, int]],
        corrected_score: Optional[Tuple[int, int, int, int]],
    ) -> None:
        if prev_score is None or raw_score is None or corrected_score is None:
            return

        self._clear_ocr_digit_lock()
        prev_a_pts, prev_b_pts = int(prev_score[1]), int(prev_score[3])
        corr_a_pts, corr_b_pts = int(corrected_score[1]), int(corrected_score[3])
        raw_a_pts, raw_b_pts = int(raw_score[1]), int(raw_score[3])

        if corr_a_pts == prev_a_pts + 1 and corr_b_pts == prev_b_pts:
            if raw_a_pts != corr_a_pts:
                self.ocr_lock["teamA"] = {"raw": raw_a_pts, "correct": corr_a_pts}
        elif corr_b_pts == prev_b_pts + 1 and corr_a_pts == prev_a_pts:
            if raw_b_pts != corr_b_pts:
                self.ocr_lock["teamB"] = {"raw": raw_b_pts, "correct": corr_b_pts}

    def _failsafe_plus_one_from_invalid_read(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        invalid_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if prev_score is None or invalid_score is None:
            return None

        old_a, old_b = int(prev_score[1]), int(prev_score[3])
        ocr_a, ocr_b = int(invalid_score[1]), int(invalid_score[3])
        if ocr_a == old_a and ocr_b == old_b:
            return None

        delta_a = abs(ocr_a - old_a)
        delta_b = abs(ocr_b - old_b)
        winner_team = "TeamA" if delta_a > delta_b else "TeamB"
        inferred = self._score_plus_one(prev_score, winner_team)
        if not self._is_logical_score_change(prev_score, inferred):
            return None
        return inferred

    def _infer_forced_score_from_invalid_read(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        invalid_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if prev_score is None or invalid_score is None:
            return None
        if prev_score[0] != invalid_score[0] or prev_score[2] != invalid_score[2]:
            return None
        changed_idxs = [i for i, (old_v, new_v) in enumerate(zip(prev_score, invalid_score)) if old_v != new_v]
        if len(changed_idxs) != 1:
            return None
        changed_idx = changed_idxs[0]
        # Progression +1 is only valid for points, not sets.
        if changed_idx not in (1, 3):
            return None
        # If OCR already read the logical +1, do not force.
        if invalid_score[changed_idx] == int(prev_score[changed_idx]) + 1:
            return None
        inferred = list(prev_score)
        inferred[changed_idx] = int(prev_score[changed_idx]) + 1
        inferred_score = tuple(inferred)
        if not self._is_logical_score_change(prev_score, inferred_score):
            return None
        return inferred_score

    def _infer_plus_one_from_dubious_read(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        dubious_score: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if prev_score is None or dubious_score is None:
            return None
        if dubious_score == prev_score:
            return None

        da = int(dubious_score[1]) - int(prev_score[1])
        db = int(dubious_score[3]) - int(prev_score[3])

        winner_team: Optional[str] = None
        if da > 0 and db <= 0:
            winner_team = "TeamA"
        elif db > 0 and da <= 0:
            winner_team = "TeamB"
        elif da < 0 and db == 0:
            winner_team = "TeamA"
        elif db < 0 and da == 0:
            winner_team = "TeamB"
        elif abs(da) > abs(db) and da != 0:
            winner_team = "TeamA"
        elif abs(db) > abs(da) and db != 0:
            winner_team = "TeamB"
        else:
            if self.last_attacker_before_net in ("CampoA", "CampoB"):
                winner_team = self._team_for_side(self.last_attacker_before_net)
            elif self.posse_atual in ("CampoA", "CampoB"):
                winner_team = self._team_for_side(self.posse_atual)

        if winner_team is None:
            return None
        return self._score_plus_one(prev_score, winner_team)

    def _score_has_blacklisted_zeros(self, score: Optional[Tuple[int, int, int, int]]) -> bool:
        if score is None or self.prev_score is None:
            return False
        if score == self.prev_score:
            return False
        prev_a_set, prev_a_pts, prev_b_set, prev_b_pts = self.prev_score
        _new_a_set, new_a_pts, _new_b_set, new_b_pts = score
        prev_progress = prev_a_pts + prev_b_pts
        zeros_all = sum(1 for v in score if v == 0)

        # Ignore hard zero patterns once the set is already advanced.
        if prev_progress >= 4 and zeros_all >= 2:
            return True
        if prev_a_pts >= 4 and new_a_pts == 0:
            return True
        if prev_b_pts >= 4 and new_b_pts == 0:
            return True
        return False

    def _update_ball_dead_state(self, ball_state, timestamp_s: float, frame_height: int) -> None:
        if not ball_state.visible:
            if self.ball_missing_since is None:
                self.ball_missing_since = float(timestamp_s)
            self.ball_ground_stable_since = None
            self.last_ground_y = None
            return

        self.ball_missing_since = None
        y = float(ball_state.pixel[1])
        ground_y_min = float(frame_height) * self.ground_zone_ratio
        if y < ground_y_min:
            self.ball_ground_stable_since = None
            self.last_ground_y = None
            return

        if self.last_ground_y is None or abs(y - self.last_ground_y) > self.ground_stable_delta_px:
            self.ball_ground_stable_since = float(timestamp_s)
            self.last_ground_y = y
            return

        self.last_ground_y = y

    def _is_ball_dead(self, timestamp_s: float) -> bool:
        if self.ball_missing_since is not None and (timestamp_s - self.ball_missing_since) >= self.ball_dead_confirm_s:
            return True
        if self.ball_ground_stable_since is not None and (timestamp_s - self.ball_ground_stable_since) >= self.ball_dead_confirm_s:
            return True
        return False

    def _cooldown_remaining_s(self, timestamp_s: float) -> float:
        return max(0.0, (self.last_point_time + self.point_cooldown_s) - timestamp_s)

    def _set_ocr_block_value(self, value: Optional[Tuple[int, int, int, int]], timestamp_s: float) -> None:
        self.valor_bloqueado_ocr = value
        self.valor_bloqueado_ocr_until = -1e9

    def _clear_ocr_block_value(self, announce: bool = False) -> None:
        self.valor_bloqueado_ocr = None
        self.valor_bloqueado_ocr_until = -1e9
        if announce:
            print("[OCR-INFO] Cadeado limpo. OCR pronto para nova leitura.")

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

    def _side_from_ball_with_net_zone(self, ball_px: Tuple[float, float], tracker) -> Tuple[Optional[str], bool]:
        side = self._side_from_ball(ball_px, tracker)
        _px, _py, dist_to_net = self._line_projection(ball_px, tracker)
        in_net_zone = bool(dist_to_net < self.net_buffer_px)
        if in_net_zone:
            return None, True
        return side, False

    def _update_crossing_counter(self, side: Optional[str], in_net_zone: bool) -> None:
        if side not in ("CampoA", "CampoB"):
            return
        if in_net_zone:
            return
        if self.crossing_confirmed_side is None:
            self.crossing_confirmed_side = side
            self.crossing_candidate_side = None
            self.crossing_candidate_frames = 0
            return
        if side == self.crossing_confirmed_side:
            self.crossing_candidate_side = None
            self.crossing_candidate_frames = 0
            return
        if self.crossing_candidate_side == side:
            self.crossing_candidate_frames += 1
        else:
            self.crossing_candidate_side = side
            self.crossing_candidate_frames = 1
        if self.crossing_candidate_frames >= self.net_cross_confirm_frames:
            self.rally_crossings += 1
            self.crossing_confirmed_side = side
            self.crossing_candidate_side = None
            self.crossing_candidate_frames = 0

    def _speed_metrics_from_drawer(self, drawer: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        ordered = self._ordered_drawer_copy(drawer)
        if len(ordered) < 2:
            return 0.0, 0.0
        speeds: List[float] = []
        for i in range(1, len(ordered)):
            x0, y0, _t0 = ordered[i - 1]
            x1, y1, _t1 = ordered[i]
            speeds.append(float(np.hypot(x1 - x0, y1 - y0)))
        if not speeds:
            return 0.0, 0.0
        return float(max(speeds)), float(np.mean(speeds))

    def _update_visual_rally_end_state(self, timestamp_s: float) -> None:
        if not self.point_started and self.rally_mgr.active is None:
            self.visual_end_state = self.RALLY_STATE_ACTIVE
            self.visual_end_since_ts = None
            return
        if self._is_ball_dead(timestamp_s):
            if self.visual_end_state == self.RALLY_STATE_ACTIVE:
                self.visual_end_state = self.RALLY_ENDED_VISUAL
                self.visual_end_since_ts = float(timestamp_s)
        elif self.visual_end_state == self.RALLY_ENDED_VISUAL:
            self.visual_end_state = self.RALLY_STATE_ACTIVE
            self.visual_end_since_ts = None

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
        synced.sort(key=lambda p: float(p[2]))
        self.ball_drawer = synced[-self.ball_drawer_maxlen :]
        tracker_attack_side = getattr(tracker, "attacking_side", None)
        if tracker_attack_side in ("CampoA", "CampoB"):
            self.attacking_side = tracker_attack_side
            self.attacking_team = self._team_for_side(tracker_attack_side)
        tracker_possession = getattr(tracker, "posse_atual", None)
        if tracker_possession not in ("CampoA", "CampoB"):
            tracker_possession = getattr(tracker, "campo_posse_atual", None)
        if tracker_possession not in ("CampoA", "CampoB"):
            tracker_possession = getattr(tracker, "current_possession", None)
        if tracker_possession in ("CampoA", "CampoB"):
            self.current_possession = tracker_possession
            self.campo_posse_atual = tracker_possession
            self.posse_atual = tracker_possession
        if bool(getattr(tracker, "tocou_rede", False)):
            self.tocou_rede = True

    def _ordered_ball_drawer(self) -> List[Tuple[float, float, float]]:
        if len(self.ball_drawer) < 2:
            return list(self.ball_drawer)
        return sorted(self.ball_drawer, key=lambda p: float(p[2]))

    def _ordered_drawer_copy(self, drawer: Optional[List[Tuple[float, float, float]]] = None) -> List[Tuple[float, float, float]]:
        source = self.ball_drawer if drawer is None else drawer
        if not source:
            return []
        if len(source) < 2:
            return [(float(source[0][0]), float(source[0][1]), float(source[0][2]))]
        return sorted([(float(p[0]), float(p[1]), float(p[2])) for p in source], key=lambda p: float(p[2]))

    def _drawer_time_bounds(
        self,
        fallback_x: float,
        fallback_y: float,
        fallback_t: float,
        drawer: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        ordered = self._ordered_drawer_copy(drawer)
        if not ordered:
            fb = (float(fallback_x), float(fallback_y), float(fallback_t))
            return fb, fb
        return ordered[0], ordered[-1]

    def _drawer_early_attack_features(
        self,
        drawer: List[Tuple[float, float, float]],
        tracker,
    ) -> Tuple[float, bool, bool, float]:
        ordered = self._ordered_drawer_copy(drawer)
        if not ordered:
            return 1e9, False, False, 0.0

        x0, y0, _t0 = ordered[0]
        _px, _py, dist_inicio_rede = self._line_projection((float(x0), float(y0)), tracker)

        n = min(3, len(ordered))
        sample = ordered[:n]
        avg_vx = 0.0
        if n >= 2:
            avg_vx = float(sample[-1][0] - sample[0][0]) / float(n - 1)

        (nx1, _ny1), (nx2, _ny2) = tracker.net_line
        mid_x = (float(nx1) + float(nx2)) / 2.0
        start_away = abs(float(sample[0][0]) - mid_x)
        end_away = abs(float(sample[-1][0]) - mid_x)
        moving_away_from_net = end_away > (start_away + self.ghost_attack_away_delta_px)

        # Strong early horizontal velocity in the same direction away from the net.
        strong_vx_attack = bool(moving_away_from_net and abs(avg_vx) >= self.ghost_attack_vx_thresh_px)

        return float(dist_inicio_rede), bool(moving_away_from_net), bool(strong_vx_attack), float(avg_vx)

    def _drawer_sides_history(self, drawer: List[Tuple[float, float, float]], tracker) -> List[str]:
        ordered = self._ordered_drawer_copy(drawer)
        sides: List[str] = []
        for x, _y, _t in ordered:
            side = self._side_from_x_position(float(x), tracker)
            if side in ("CampoA", "CampoB"):
                sides.append(side)
        return sides

    def _reset_for_new_service(self, tracker, source: str, seed_point: Optional[Tuple[float, float, float]] = None) -> None:
        if source == "SERVICO":
            if self.service_detection_locked:
                return
            tracker_drawer = getattr(tracker, "ball_drawer", [])
            has_drawer_data = bool(self.ball_drawer) or bool(tracker_drawer)
            if has_drawer_data and not self.rally_closed_by_scoreboard:
                # Safety: never erase rally trajectory before OCR closes the point.
                self.service_detection_locked = True
                print("[SERVICE-RESET-BLOCK] Reset de serviço bloqueado: rally ainda aberto e gaveta já tem dados.")
                return
        if source == "SERVICO":
            self.service_detection_locked = True
        elif source == "OCR":
            self.service_detection_locked = False
        self.ball_drawer.clear()
        self.tocou_rede = False
        self.toucou_rede = False
        self.point_finalized = False
        self.pending_score_drawer_snapshot = None
        self.pending_score_prev_base = None
        self.pending_score_forced = False
        self.invalid_ocr_candidate = None
        self.high_jump_candidate = None
        self.high_jump_count = 0
        self.tentativas_ocr = 0
        self.attacking_side = None
        self.current_rally_trajectory = []
        self.rally_crossings = 0
        self.crossing_candidate_side = None
        self.crossing_candidate_frames = 0
        self.crossing_confirmed_side = None
        self.visual_end_state = self.RALLY_STATE_ACTIVE
        self.visual_end_since_ts = None
        # Keep possession state persistent across resets; attacker-before-net is rally-scoped.
        self.last_attacker_before_net = None
        self.attacking_team = None
        self.ball_in_zone_flag = False
        self.last_net_touch_time = None
        self.last_net_touch_attack_side = None
        self.pending_net_events.clear()
        self.rebound_candidate = None
        self.occlusion_candidate = None
        self.last_visible_ball = None
        self.last_ball_exit_net_side = None
        self.last_route_collision_to_net = False
        self.last_route_collision_ts = None
        self.prev_visible_in_block_zone = False
        self.estado_retorno_detectado = False
        self._reset_event_state()

        if seed_point is not None and hasattr(tracker, "reset_drawer_for_service"):
            sx, sy, st = seed_point
            tracker.reset_drawer_for_service(float(sx), float(sy), float(st))
        elif hasattr(tracker, "clear_ball_drawer"):
            tracker.clear_ball_drawer()

        if hasattr(tracker, "attacking_side"):
            tracker.attacking_side = None
        if hasattr(tracker, "tocou_rede"):
            tracker.tocou_rede = False

        self._sync_ball_drawer(tracker)
        print(f"[SERVICE-RESET] Origem: {source} | Gaveta e estado reiniciados.")

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

    def _drawer_passed_zone_block(self, tracker, drawer: Optional[List[Tuple[float, float, float]]] = None) -> bool:
        ordered = self._ordered_drawer_copy(drawer)
        if not ordered:
            return False
        for x, y, _t in ordered:
            if self._point_in_block_zone((x, y), tracker):
                return True
        return False

    def _drawer_crossed_net(self, tracker) -> bool:
        ordered = self._ordered_ball_drawer()
        if len(ordered) < 2:
            return False
        prev_side: Optional[str] = None
        for x, y, _t in ordered:
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
        return "CampoA" if float(x) < mid_x else "CampoB"

    def check_post_mortem(
        self,
        tracker,
        end_side: Optional[str],
        drawer: Optional[List[Tuple[float, float, float]]] = None,
        attacker_side: Optional[str] = None,
    ) -> Optional[Tuple[str, Optional[str], Optional[str], bool]]:
        if drawer is None and (not self.ball_drawer or len(self.ball_drawer) < 2):
            return None
        base_drawer = self._ordered_drawer_copy(drawer)
        if not base_drawer or len(base_drawer) < 2:
            return None
        try:
            x_end, y_end, _ = base_drawer[-1]
            x_start, y_start, _ = base_drawer[0]
            side_start_drawer = self._side_from_ball((float(x_start), float(y_start)), tracker)
            lado_fim = self._side_from_ball((float(x_end), float(y_end)), tracker)
            lado_atacante = attacker_side if attacker_side in ("CampoA", "CampoB") else self.last_attacker_before_net
            if lado_atacante not in ("CampoA", "CampoB"):
                lado_atacante = self.posse_atual
            if lado_atacante not in ("CampoA", "CampoB"):
                lado_atacante = self.campo_posse_atual
            if lado_atacante not in ("CampoA", "CampoB"):
                lado_atacante = self.current_possession
            if lado_atacante not in ("CampoA", "CampoB"):
                lado_atacante = side_start_drawer

            sides_history: List[str] = []
            for x, y, _t in base_drawer:
                s, in_net_zone = self._side_from_ball_with_net_zone((float(x), float(y)), tracker)
                if in_net_zone or s not in ("CampoA", "CampoB"):
                    continue
                sides_history.append(s)
            crossed = False
            if len(sides_history) >= 2:
                prev_s = sides_history[0]
                for s in sides_history[1:]:
                    if s != prev_s:
                        crossed = True
                        break
                    prev_s = s
            if sides_history:
                lado_fim = sides_history[-1]

            peak_speed, _mean_speed = self._speed_metrics_from_drawer(base_drawer)
            tocou_rede = bool(
                self._drawer_passed_zone_block(tracker, drawer=base_drawer)
                or self.ball_in_zone_flag
                or self.event_impact_ts is not None
                or self.event_state == self.STATE_IMPACTO
                or self.estado_retorno_detectado
                or self.event_rebouce_ts is not None
                or self.tocou_rede
            )
            if lado_atacante is None or lado_fim is None:
                resultado = "ERROR"
            elif tocou_rede and lado_fim == lado_atacante:
                resultado = "BLOCK"
            elif crossed and lado_fim != lado_atacante and peak_speed > self.spike_speed_threshold_px:
                resultado = "SPIKE"
            elif crossed and lado_fim != lado_atacante and peak_speed <= self.spike_speed_threshold_px:
                resultado = "FREEBALL"
            else:
                resultado = "ERROR"
            print(
                f"[FINAL] Origem: {lado_atacante} | Fim: {lado_fim} | Rede: {tocou_rede} | "
                f"Cruzou: {crossed} | Vmax: {peak_speed:.2f} -> RESULTADO: {resultado}."
            )
            print(f"[POSSE] Bola no {self.posse_atual} | Fim da Seta: {lado_fim}")
            return resultado, lado_atacante, lado_fim, bool(crossed)
        except Exception:
            return None

    def _post_mortem_block_from_drawer(self, tracker, timestamp_s: float) -> Tuple[bool, Optional[str], Optional[str]]:
        # Post-mortem over the full global drawer (last 50 detections).
        if not self.ball_drawer:
            return False, None, None
        recent = self._ordered_ball_drawer()
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
        self.tocou_rede = True
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
                        self.estado_retorno_detectado = True
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
        Stats lock:
        this function is only allowed to run once OCR has confirmed a score change.
        Net events are annotations and cannot finalize stats on their own.
        """
        if not self.point_finalized:
            return baseline_ptype
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
        self.point_finalized = False
        self.score_just_updated = False
        self._sync_ball_drawer(tracker)

        ball_px = (float(ball_state.pixel[0]), float(ball_state.pixel[1]))
        self._update_ball_dead_state(ball_state, timestamp_s, frame.shape[0])
        self._update_visual_rally_end_state(timestamp_s)
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
        ordered_drawer = self._ordered_ball_drawer()
        if len(ordered_drawer) >= 2:
            lado0 = self._side_from_x_position(float(ordered_drawer[0][0]), tracker)
            lado1 = self._side_from_x_position(float(ordered_drawer[-1][0]), tracker)
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
        posse_txt = "Posse: --"
        posse_side = self.campo_posse_atual if self.campo_posse_atual in ("CampoA", "CampoB") else self.current_possession
        if posse_side == "CampoA":
            posse_txt = "Posse: Equipa A"
        elif posse_side == "CampoB":
            posse_txt = "Posse: Equipa B"
        cv2.putText(
            frame,
            posse_txt,
            (20, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 1) Side tracking (for block memory, a single frame in Campo B is enough).
        if ball_state.visible:
            prev_side_before_frame = self.current_side
            side_detected, in_net_zone = self._side_from_ball_with_net_zone(ball_px, tracker)
            self._update_crossing_counter(side_detected, in_net_zone)
            if side_detected in ("CampoA", "CampoB"):
                side = side_detected
            else:
                side = self.current_side
            tracker_possession = getattr(tracker, "posse_atual", None)
            if tracker_possession not in ("CampoA", "CampoB"):
                tracker_possession = getattr(tracker, "campo_posse_atual", None)
            if tracker_possession not in ("CampoA", "CampoB"):
                tracker_possession = getattr(tracker, "current_possession", None)
            if tracker_possession in ("CampoA", "CampoB"):
                self.current_possession = tracker_possession
                self.campo_posse_atual = tracker_possession
                self.posse_atual = tracker_possession
            if side_detected in ("CampoA", "CampoB") and self.attacking_side is None:
                self.attacking_side = side_detected
                self.attacking_team = self._team_for_side(side_detected)

            # Service detection must happen once per point.
            if not self.point_started and not self.service_detection_locked:
                inferred_service_side = side if side is not None else self.current_side
                if inferred_service_side is not None:
                    self._reset_for_new_service(
                        tracker,
                        source="SERVICO",
                        seed_point=(float(ball_px[0]), float(ball_px[1]), float(timestamp_s)),
                    )
                    self.serving_side = inferred_service_side
                    self.last_service_side = inferred_service_side
                    self.current_possession = inferred_service_side
                    self.campo_posse_atual = inferred_service_side
                    self.posse_atual = inferred_service_side
                    self.last_attacker_before_net = inferred_service_side
                    self.point_started = True
                    self.rally_closed_by_scoreboard = False
            if self.point_started and self.rally_mgr.active is None:
                self.rally_mgr.start_if_needed(True, timestamp_s, frame_idx, tracker.trail_points())

            if not in_net_zone:
                self._update_possession(side, timestamp_s)
            if side in ("CampoA", "CampoB") and not in_net_zone:
                transition_happened = bool(
                    dist_to_net <= self.net_touch_tolerance_px
                    or in_block_zone
                    or (
                        prev_side_before_frame in ("CampoA", "CampoB")
                        and side in ("CampoA", "CampoB")
                        and side != prev_side_before_frame
                    )
                )
                if transition_happened:
                    attacker_candidate = self.posse_atual if self.posse_atual in ("CampoA", "CampoB") else prev_side_before_frame
                    if attacker_candidate in ("CampoA", "CampoB"):
                        self.last_attacker_before_net = attacker_candidate
            self.last_ball_seen_ts = float(timestamp_s)
            if side == "CampoB":
                self.last_seen_side_b_time = timestamp_s
            if self.point_started and abs(float(ball_px[0])) > 1e-6 and abs(float(ball_px[1])) > 1e-6:
                if not self.current_rally_trajectory or self.current_rally_trajectory[-1][0] != int(frame_idx):
                    self.current_rally_trajectory.append((int(frame_idx), float(ball_px[0]), float(ball_px[1])))

            if in_block_zone:
                self.ball_in_zone_flag = True
                self.tocou_rede = True
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

        if frame_idx % config.ocr_every_n_frames == 0 and self.pending_score_change is None:
            score_stable_raw = self.ocr.read(frame)
            score_raw_raw = getattr(self.ocr, "last_raw_score", None)
            score_stable = self._apply_ocr_digit_lock(score_stable_raw)
            score_raw = self._apply_ocr_digit_lock(score_raw_raw)
            self.last_ocr_score_stable = score_stable
            self.last_ocr_score_raw = score_raw

            score = score_stable
            score_for_failsafe_lock = score_stable_raw
            # Score-first closure: if stable OCR lags, use raw changed reading.
            if self.prev_score is not None:
                if (score is None or score == self.prev_score) and score_raw is not None and score_raw != self.prev_score:
                    score = score_raw
                    score_for_failsafe_lock = score_raw_raw
                # Release OCR locks as soon as OCR aligns with official score.
                if score_stable == self.prev_score or score_raw == self.prev_score:
                    self._clear_ocr_block_value(announce=False)
                    self.valor_recuperacao_ocr = None

            if score is not None and self.valor_bloqueado_ocr is not None:
                if score == self.valor_bloqueado_ocr:
                    print(f"[OCR-LOCKED] Mantendo placar forçado e ignorando leitura errada repetida: {score}.")
                    score = None
                else:
                    self._clear_ocr_block_value(announce=False)

            if score is not None and self.valor_recuperacao_ocr is not None:
                if self.prev_score is not None and score == self.prev_score:
                    # OCR caught up to the current reference score.
                    self.valor_recuperacao_ocr = None
                elif score == self.valor_recuperacao_ocr:
                    print(f"[OCR-RECOVERY] Ignorando leitura transitória antiga: {score}.")
                    score = None

            if self.prev_score is None:
                if score is not None:
                    self.prev_score = score
                    self._clear_ocr_digit_lock()
                    self.invalid_ocr_candidate = None
                    self.tentativas_ocr = 0
                    self._reset_ocr_correction_tracking()
                elif score_raw is not None:
                    self.prev_score = score_raw
                    self._clear_ocr_digit_lock()
                    self.invalid_ocr_candidate = None
                    self.tentativas_ocr = 0
                    self._reset_ocr_correction_tracking()
            elif score is not None:
                # Release OCR lock once reading aligns with official score.
                if score == self.prev_score:
                    self._clear_ocr_block_value(announce=False)
                    self.valor_recuperacao_ocr = None
                    self.invalid_ocr_candidate = None
                    self.tentativas_ocr = 0
                    self.high_jump_candidate = None
                    self.high_jump_count = 0
                    self._reset_ocr_correction_tracking()
                else:
                    accepted_score: Optional[Tuple[int, int, int, int]] = None
                    forced_update = False

                    if self._is_logical_score_change(self.prev_score, score):
                        accepted_score = score
                        # New valid OCR +1 progression: drop any temporary digit lock.
                        self._clear_ocr_digit_lock()
                        self.invalid_ocr_candidate = None
                        self.tentativas_ocr = 0
                    else:
                        points_changed = bool(
                            int(score[1]) != int(self.prev_score[1])
                            or int(score[3]) != int(self.prev_score[3])
                        )
                        if points_changed:
                            self.invalid_ocr_candidate = score
                            self.tentativas_ocr += 1
                            print(
                                f"[OCR-RETRY] leitura inválida {self.prev_score} -> {score} "
                                f"({self.tentativas_ocr}/{self.invalid_ocr_retry_limit})."
                            )

                            if self.tentativas_ocr >= self.invalid_ocr_retry_limit:
                                failsafe_score = self._failsafe_plus_one_from_invalid_read(self.prev_score, score)
                                if failsafe_score is not None:
                                    accepted_score = failsafe_score
                                    forced_update = True
                        else:
                            # Sets-only noise or no points movement: ignore and reset retry streak.
                            print(f"[OCR-REJECT] Mudança inválida sem alteração de pontos: {self.prev_score} -> {score}.")
                            self.invalid_ocr_candidate = None
                            self.tentativas_ocr = 0

                    if accepted_score is not None:
                        # Temporal protection: avoid duplicate rally counting.
                        if int(frame_idx - self.last_point_frame) < self.min_frames_between_points:
                            print(
                                f"[OCR-REJECT] Mudança +1 ignorada por janela temporal "
                                f"({frame_idx - self.last_point_frame} < {self.min_frames_between_points} frames)."
                            )
                            self.high_jump_candidate = None
                            self.high_jump_count = 0
                        else:
                            if forced_update:
                                self._set_ocr_digit_lock_from_failsafe(
                                    prev_score=self.prev_score,
                                    raw_score=score_for_failsafe_lock,
                                    corrected_score=accepted_score,
                                )
                                print(
                                    f"[OCR-FAILSAFE] {self.tentativas_ocr} tentativas inválidas. "
                                    f"Forçando {self.prev_score} -> {accepted_score} "
                                    f"com base na leitura {score}."
                                )
                            else:
                                print(f"[OCR-VALID] Mudança +1 confirmada: {self.prev_score} -> {accepted_score}.")
                            self.pending_score_prev_base = self.prev_score
                            self.pending_score_change = accepted_score
                            self.pending_score_change_ts = timestamp_s
                            self.pending_score_drawer_snapshot = self._ordered_ball_drawer()
                            self.pending_score_forced = forced_update
                            self.invalid_ocr_candidate = None
                            self.tentativas_ocr = 0
                            self.high_jump_candidate = None
                            self.high_jump_count = 0
                            self._reset_ocr_correction_tracking()
                    else:
                        self.high_jump_candidate = None
                        self.high_jump_count = 0
                        self._reset_ocr_correction_tracking()

        rally_finished = False
        point_label = None

        if self.pending_score_change is not None:
            if self.rally_mgr.active is None:
                self.rally_mgr.start_if_needed(True, timestamp_s, frame_idx, tracker.trail_points())

            analysis_drawer = self._ordered_drawer_copy(self.pending_score_drawer_snapshot)
            if not analysis_drawer:
                analysis_drawer = self._ordered_ball_drawer()
            speed_peak, speed_mean = self._speed_metrics_from_drawer(analysis_drawer)
            trajectory_for_rally: List[Tuple[int, float, float]] = list(self.current_rally_trajectory)
            if not trajectory_for_rally and analysis_drawer:
                start_frame_guess = int(self.rally_mgr.active.start_frame) if self.rally_mgr.active is not None else int(frame_idx - len(analysis_drawer))
                for idx, (x, y, _t) in enumerate(analysis_drawer):
                    trajectory_for_rally.append((int(start_frame_guess + idx), float(x), float(y)))

            winner = self._winner_from_score_change(
                self.pending_score_change,
                prev_score=self.pending_score_prev_base,
            )
            drawer_start, drawer_end = self._drawer_time_bounds(
                fallback_x=float(ball_px[0]),
                fallback_y=float(ball_px[1]),
                fallback_t=float(timestamp_s),
                drawer=analysis_drawer,
            )
            x_inicio = float(drawer_start[0])
            lado_origem = self._side_from_x_position(x_inicio, tracker)

            print(
                f"[STAS-VALID] Placar mudou. Frames na gaveta: {len(analysis_drawer)}. "
                "Contando rali e verificando estatística técnica..."
            )
            inconclusivo = True
            resultado = "RALLY_ONLY"
            ptype = "RALLY_ONLY"
            net_crossing_detected = bool(self.rally_crossings > 0 or self._drawer_crossed_net(tracker))
            can_classify_short = bool(len(analysis_drawer) >= self.min_frames_for_technical_stats or net_crossing_detected)
            if not can_classify_short:
                print("[INFO] Rali contabilizado por placar, mas sem frames/crossing suficientes para classificação técnica.")
            else:
                attacker_hint = self.last_attacker_before_net
                if attacker_hint not in ("CampoA", "CampoB"):
                    tracker_attacker = getattr(tracker, "attacking_side", None)
                    attacker_hint = tracker_attacker if tracker_attacker in ("CampoA", "CampoB") else self.campo_posse_atual
                pm = self.check_post_mortem(
                    tracker=tracker,
                    end_side=None,
                    drawer=analysis_drawer,
                    attacker_side=attacker_hint,
                )
                if pm is not None:
                    decision_result, lado_origem_pm, _lado_destino_pm, _crossed_pm = pm
                    if lado_origem_pm is not None:
                        lado_origem = lado_origem_pm
                    if decision_result == "BLOCK":
                        inconclusivo = False
                        resultado = "BLOCK"
                        ptype = "POINT_BY_BLOCK"
                    elif decision_result == "SPIKE":
                        inconclusivo = False
                        resultado = "SPIKE"
                        ptype = "POINT_BY_SPIKE"
                    elif decision_result == "FREEBALL":
                        inconclusivo = False
                        resultado = "FREEBALL"
                        ptype = "FREEBALL"

            winner_effective = winner
            if winner_effective == "Unknown":
                if lado_origem in ("CampoA", "CampoB"):
                    if resultado == "BLOCK":
                        winner_effective = self._team_for_side(self._other_side(lado_origem))
                    elif resultado in ("SPIKE", "FREEBALL"):
                        winner_effective = self._team_for_side(lado_origem)
                elif self.posse_atual in ("CampoA", "CampoB"):
                    winner_effective = self._team_for_side(self.posse_atual)

            final_score = self.pending_score_change

            self.point_finalized = True
            ptype = self.update_stats(baseline_ptype=ptype, winner=winner_effective, timestamp_s=timestamp_s)
            self.visual_end_state = self.RALLY_CONFIRMED_OCR

            if final_score is not None:
                self.prev_score = final_score
                self.ocr.stable_score = final_score
                self.score_just_updated = True
            self.pending_score_change = None
            self.pending_score_change_ts = None
            self.pending_score_drawer_snapshot = None
            self.pending_score_prev_base = None
            self.pending_score_forced = False
            self.invalid_ocr_candidate = None
            self.high_jump_candidate = None
            self.high_jump_count = 0
            self.tentativas_ocr = 0
            self.last_point_time = float(timestamp_s)
            self.last_point_frame = int(frame_idx)
            self.rally_counter += 1
            self.counts["Rallies"] += 1
            self.rally_closed_by_scoreboard = True

            self.rally_mgr.end(
                ts=timestamp_s,
                frame_idx=frame_idx,
                winner=winner_effective,
                ptype="RALLY_ONLY" if inconclusivo else resultado,
                max_speed=speed_peak,
                impact=ball_state.court,
                reason="score_change",
                attacker=lado_origem,
                net_crossings=self.rally_crossings,
                ball_speed_mean=speed_mean,
                trajectory=trajectory_for_rally,
            )
            if inconclusivo:
                print("[FINAL] Rali terminado, mas trajetória inconclusiva. Apenas +1 no contador de ralis.")
            elif self.point_finalized:
                if resultado == "SPIKE":
                    self.counts["Spikes"] += 1
                    self.counts["POINT_BY_SPIKE"] += 1
                elif resultado == "BLOCK":
                    self.counts["Blocks"] += 1
                    self.counts["POINT_BY_BLOCK"] += 1
                elif resultado == "FREEBALL":
                    self.counts["Freeballs"] += 1
                    self.counts["FREEBALL"] += 1

            # End-of-rally cleanup for next possession/event cycle (hard reset).
            self._reset_for_new_service(tracker, source="OCR")
            self.last_valid_y = None
            self.prev_valid_ball_px = None
            self.prev_prev_valid_ball_px = None
            self.side_frame_streak = 0
            self.last_streak_side = None
            self.possession_side_5frames = None
            self.last_seen_side_b_time = None
            self.prev_visible_side = None
            self.prev_visible_dist_to_net = None
            self.prev_ball_visible = False
            self.point_started = False
            self.serving_side = None

            rally_finished = True
            point_label = "RALLY_ONLY" if inconclusivo else resultado

        return rally_finished, point_label

