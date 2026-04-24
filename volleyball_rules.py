"""
volleyball_rules.py
-------------------
Incremental volleyball game intelligence.

This module does not try to model the full official rulebook. It keeps a small,
explicit temporal state that is useful for computer vision: ball continuity,
net crossing, likely possession, rally lifecycle hints and scoreboard
confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ball_tracking_core import TRACK_LOST, TRACK_OBSERVED, TRACK_PREDICTED
from court_geometry import CourtGeometry, SIDE_A, SIDE_B


@dataclass
class VolleyballRulesConfig:
    enabled: bool = True
    validate_ball_candidates: bool = True
    suppress_dubious_ball_for_analytics: bool = True
    debug_log: bool = False
    net_neutral_px: float = 15.0
    net_cross_tolerance_px: float = 80.0
    net_cross_confirm_frames: int = 3
    max_ball_step_px: float = 260.0
    missing_frame_step_allowance_px: float = 35.0
    max_missing_grace_frames: int = 8
    teleport_net_tolerance_px: float = 130.0
    abrupt_turn_cos: float = -0.88
    abrupt_turn_min_step_px: float = 45.0
    abrupt_turn_net_grace_px: float = 140.0
    stats_penalty_threshold: float = 45.0
    reject_penalty_threshold: float = 100.0
    large_step_near_net_penalty: float = 55.0
    impossible_step_penalty: float = 120.0
    side_teleport_penalty: float = 120.0
    abrupt_turn_penalty: float = 45.0
    possession_confirm_frames: int = 3
    rally_lost_confirm_s: float = 1.0
    score_confirm_window_s: float = 2.0
    out_of_bounds_margin_m: float = 0.75
    out_of_bounds_penalty: float = 120.0
    # Velocidade máxima física de uma bola de voleibol (~126 km/h).
    # Qualquer candidato que implique velocidade superior é rejeitado.
    max_ball_speed_ms: float = 35.0
    # Rejeição binária por velocidade ao nível das regras.
    # DESACTIVADO por defeito: a homografia projecta a bola (no ar) sobre o
    # plano do chão, inflando a "distância em metros" sempre que a bola sobe
    # ou desce verticalmente — gera cascatas de PHYSICS-REJECT em jogadas
    # normais. O sistema de scoring em ball_tracking_core.py já tem uma
    # componente de velocidade graduada (com homografia também) que penaliza
    # candidatos rápidos sem rejeitar binariamente.
    physics_reject_enabled: bool = False

    @classmethod
    def from_config(cls, config) -> "VolleyballRulesConfig":
        return cls(
            enabled=bool(getattr(config, "GAME_RULES_ENABLED", True)),
            validate_ball_candidates=bool(getattr(config, "GAME_RULES_VALIDATE_BALL", True)),
            suppress_dubious_ball_for_analytics=bool(getattr(config, "GAME_RULES_SUPPRESS_DUBIOUS_BALL_FOR_ANALYTICS", True)),
            debug_log=bool(getattr(config, "GAME_RULES_DEBUG_LOG", False)),
            net_neutral_px=float(getattr(config, "game_net_neutral_px", getattr(config, "net_buffer_px", 15.0))),
            net_cross_tolerance_px=float(getattr(config, "game_net_cross_tolerance_px", getattr(config, "net_line_tolerance_px", 80.0))),
            net_cross_confirm_frames=int(getattr(config, "game_net_cross_confirm_frames", getattr(config, "net_cross_confirm_frames", 3))),
            max_ball_step_px=float(getattr(config, "game_ball_max_step_px", 260.0)),
            missing_frame_step_allowance_px=float(getattr(config, "game_ball_missing_step_allowance_px", 35.0)),
            max_missing_grace_frames=int(getattr(config, "game_ball_max_missing_grace_frames", 8)),
            teleport_net_tolerance_px=float(getattr(config, "game_ball_teleport_net_tolerance_px", 130.0)),
            abrupt_turn_cos=float(getattr(config, "game_ball_abrupt_turn_cos", -0.88)),
            abrupt_turn_min_step_px=float(getattr(config, "game_ball_abrupt_turn_min_step_px", 45.0)),
            abrupt_turn_net_grace_px=float(getattr(config, "game_ball_abrupt_turn_net_grace_px", 140.0)),
            stats_penalty_threshold=float(getattr(config, "game_ball_stats_penalty_threshold", 45.0)),
            reject_penalty_threshold=float(getattr(config, "game_ball_reject_penalty_threshold", 100.0)),
            large_step_near_net_penalty=float(getattr(config, "game_ball_large_step_near_net_penalty", 55.0)),
            impossible_step_penalty=float(getattr(config, "game_ball_impossible_step_penalty", 120.0)),
            side_teleport_penalty=float(getattr(config, "game_ball_side_teleport_penalty", 120.0)),
            abrupt_turn_penalty=float(getattr(config, "game_ball_abrupt_turn_penalty", 45.0)),
            possession_confirm_frames=int(getattr(config, "game_possession_confirm_frames", 3)),
            rally_lost_confirm_s=float(getattr(config, "game_rally_lost_confirm_s", 1.0)),
            score_confirm_window_s=float(getattr(config, "game_score_confirm_window_s", 2.0)),
            out_of_bounds_margin_m=float(getattr(config, "game_ball_out_of_bounds_margin_m", getattr(config, "court_margin_m", 0.75))),
            out_of_bounds_penalty=float(getattr(config, "game_ball_out_of_bounds_penalty", 120.0)),
            max_ball_speed_ms=float(getattr(config, "max_ball_speed_ms", 35.0)),
            physics_reject_enabled=bool(getattr(config, "GAME_RULES_PHYSICS_REJECT_ENABLED", False)),
        )


@dataclass
class BallRuleDecision:
    quality: str = "trusted"
    penalty: float = 0.0
    reject: bool = False
    accepted_for_stats: bool = True
    reason: str = "ok"
    side: Optional[str] = None
    crosses_net: bool = False
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "quality": self.quality,
            "penalty": float(self.penalty),
            "reject": bool(self.reject),
            "accepted_for_stats": bool(self.accepted_for_stats),
            "reason": self.reason,
            "side": self.side,
            "crosses_net": bool(self.crosses_net),
            "debug": dict(self.debug),
        }


@dataclass
class GameEvent:
    event_type: str
    frame_idx: int
    timestamp_s: float
    details: Dict = field(default_factory=dict)


@dataclass
class GameContext:
    frame_idx: int = 0
    timestamp_s: float = 0.0
    current_rally_id: int = 0
    rally_active: bool = False
    ball_track_state: str = TRACK_LOST
    ball_quality: str = "lost"
    ball_accepted_for_stats: bool = False
    ball_side: Optional[str] = None
    possession_side: Optional[str] = None
    possession_team: Optional[str] = None
    last_net_crossing_frame: Optional[int] = None
    last_net_crossing_direction: Optional[str] = None
    rally_net_crossings: int = 0
    last_score: Optional[Tuple[int, int, int, int]] = None
    score_confirmed_point: bool = False
    rally_start: bool = False
    rally_end_candidate: bool = False
    rally_end_confirmed: bool = False
    events: List[GameEvent] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_s": float(self.timestamp_s),
            "current_rally_id": int(self.current_rally_id),
            "rally_active": bool(self.rally_active),
            "ball_track_state": self.ball_track_state,
            "ball_quality": self.ball_quality,
            "ball_accepted_for_stats": bool(self.ball_accepted_for_stats),
            "ball_side": self.ball_side,
            "possession_side": self.possession_side,
            "possession_team": self.possession_team,
            "last_net_crossing_frame": self.last_net_crossing_frame,
            "last_net_crossing_direction": self.last_net_crossing_direction,
            "rally_net_crossings": int(self.rally_net_crossings),
            "last_score": self.last_score,
            "score_confirmed_point": bool(self.score_confirmed_point),
            "rally_start": bool(self.rally_start),
            "rally_end_candidate": bool(self.rally_end_candidate),
            "rally_end_confirmed": bool(self.rally_end_confirmed),
            "events": [event.__dict__ for event in self.events[-6:]],
            "reasons": list(self.reasons),
            "debug": dict(self.debug),
        }


class VolleyballGameIntelligence:
    def __init__(
        self,
        geometry: CourtGeometry,
        cfg: VolleyballRulesConfig,
    ) -> None:
        self.geometry = geometry
        self.net_line = geometry.net_line
        self.cfg = cfg
        self.current_rally_id = 0
        self.rally_active = False
        self.ball_side: Optional[str] = None
        self.candidate_side: Optional[str] = None
        self.candidate_side_frames = 0
        self.possession_side: Optional[str] = None
        self.possession_team: Optional[str] = None
        self.prev_observed_center: Optional[Tuple[float, float]] = None
        self.last_observed_center: Optional[Tuple[float, float]] = None
        self.last_observed_frame: Optional[int] = None
        self.last_observed_ts: Optional[float] = None
        self.lost_since_ts: Optional[float] = None
        self.last_net_crossing_frame: Optional[int] = None
        self.last_net_crossing_direction: Optional[str] = None
        self.rally_net_crossings = 0
        self.last_score: Optional[Tuple[int, int, int, int]] = None
        self.last_context = GameContext()
        self.events: List[GameEvent] = []

    def _emit(self, event_type: str, frame_idx: int, timestamp_s: float, **details) -> GameEvent:
        event = GameEvent(event_type=event_type, frame_idx=int(frame_idx), timestamp_s=float(timestamp_s), details=details)
        self.events.append(event)
        if len(self.events) > 120:
            self.events = self.events[-120:]
        if self.cfg.debug_log:
            print(f"[GAME-RULES] {event_type} frame={frame_idx} ts={timestamp_s:.2f} {details}")
        return event

    def _signed_side_value(self, point: Tuple[float, float]) -> float:
        return float(self.geometry.signed_distance_to_net(point))

    def _signed_distance_to_net(self, point: Tuple[float, float]) -> float:
        return float(self.geometry.signed_distance_to_net(point))

    def _side_from_point(self, point: Tuple[float, float]) -> Optional[str]:
        return self.geometry.get_side_of_net(point, neutral_tolerance_px=self.cfg.net_neutral_px)

    def _team_for_side(self, side: Optional[str]) -> Optional[str]:
        if side == SIDE_A:
            return "TeamA"
        if side == SIDE_B:
            return "TeamB"
        return None

    def _project_to_net(self, point: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        return self.geometry.project_point_to_net(point)

    def _dist_to_net(self, point: Tuple[float, float]) -> float:
        return float(self.geometry.distance_to_net(point))

    def _segment_crosses_net(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        return bool(self.geometry.did_cross_net(p0, p1, neutral_tolerance_px=self.cfg.net_cross_tolerance_px))

    def _logical_score_change(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        if prev_score is None or new_score is None or prev_score == new_score:
            return False
        prev_a_set, prev_a_pts, prev_b_set, prev_b_pts = prev_score
        new_a_set, new_a_pts, new_b_set, new_b_pts = new_score
        if prev_a_set != new_a_set or prev_b_set != new_b_set:
            return False
        da = int(new_a_pts) - int(prev_a_pts)
        db = int(new_b_pts) - int(prev_b_pts)
        return (da == 1 and db == 0) or (da == 0 and db == 1)

    def evaluate_candidate(
        self,
        center: Optional[Tuple[float, float]],
        timestamp_s: Optional[float],
        frame_idx: Optional[int],
    ) -> BallRuleDecision:
        if not self.cfg.enabled or not self.cfg.validate_ball_candidates or center is None:
            return BallRuleDecision()

        point = (float(center[0]), float(center[1]))
        court_point = self.geometry.pixel_to_court(point)
        side = self._side_from_point(point)
        penalty = 0.0
        reasons: List[str] = []
        court_inside = self.geometry.is_inside_court(
            court_point,
            point_space="court",
            margin_m=self.cfg.out_of_bounds_margin_m,
        )
        court_polygon = self.geometry.get_court_zones()["court"]
        pixel_inside = self.geometry.contains_pixel(court_polygon, point)
        x_min = float(np.min(court_polygon[:, 0]))
        x_max = float(np.max(court_polygon[:, 0]))
        top_grace_px = max(self.cfg.net_cross_tolerance_px, 120.0)
        pixel_inside_vertical_grace = x_min <= point[0] <= x_max and point[1] >= (self.geometry.court_top_y - top_grace_px)
        inside_court = bool(court_inside or pixel_inside or pixel_inside_vertical_grace)
        zone_name = self.geometry.point_zone(point)
        debug: Dict = {
            "candidate_center": point,
            "candidate_court": court_point,
            "candidate_side": side,
            "inside_court": bool(inside_court),
            "zone": zone_name,
        }
        crosses_net = False

        if not inside_court:
            penalty += self.cfg.out_of_bounds_penalty
            reasons.append("outside_court")

        if self.last_observed_center is not None:
            step_px = float(np.hypot(point[0] - self.last_observed_center[0], point[1] - self.last_observed_center[1]))
            last_dist_net = self._dist_to_net(self.last_observed_center)
            this_dist_net = self._dist_to_net(point)
            net_motion = self.geometry.net_relative_motion(self.last_observed_center, point)
            crosses_net = self._segment_crosses_net(self.last_observed_center, point)
            debug.update(
                {
                    "step_px": step_px,
                    "last_dist_net": last_dist_net,
                    "candidate_dist_net": this_dist_net,
                    "crosses_net": crosses_net,
                    "towards_net_delta": float(net_motion["towards_net_delta"]),
                }
            )
            missing_frames = 0
            if frame_idx is not None and self.last_observed_frame is not None:
                missing_frames = max(0, int(frame_idx) - int(self.last_observed_frame) - 1)
            max_step = self.cfg.max_ball_step_px + (
                min(missing_frames, self.cfg.max_missing_grace_frames) * self.cfg.missing_frame_step_allowance_px
            )
            if step_px > max_step:
                if crosses_net or min(last_dist_net, this_dist_net) <= self.cfg.teleport_net_tolerance_px:
                    penalty += self.cfg.large_step_near_net_penalty
                    reasons.append(f"large_step_near_net:{step_px:.0f}>{max_step:.0f}")
                else:
                    penalty += self.cfg.impossible_step_penalty
                    reasons.append(f"impossible_step:{step_px:.0f}>{max_step:.0f}")

            # --- Validação física em metros reais (homografia calibrada + timestamps) ---
            # A distância em court-space é calculada para debug/telemetria, mas
            # a rejeição binária só corre se `physics_reject_enabled=True`.
            # Por defeito: desligada — o `score_candidate` em ball_tracking_core
            # tem componente de velocidade graduada (também via homografia) que
            # penaliza sem produzir cascatas de rejeição quando a bola está no
            # ar (a homografia de chão infla artificialmente a distância para
            # qualquer movimento vertical real da bola).
            if timestamp_s is not None and self.last_observed_ts is not None:
                time_elapsed_s = max(float(timestamp_s) - float(self.last_observed_ts), 1.0 / 120.0)
                last_court = self.geometry.pixel_to_court(self.last_observed_center)
                court_dist_m = float(np.hypot(
                    court_point[0] - last_court[0],
                    court_point[1] - last_court[1],
                ))
                required_speed_ms = court_dist_m / time_elapsed_s
                debug["court_dist_m"] = round(court_dist_m, 3)
                debug["required_speed_ms"] = round(required_speed_ms, 1)
                if self.cfg.physics_reject_enabled and required_speed_ms > self.cfg.max_ball_speed_ms:
                    penalty += self.cfg.impossible_step_penalty
                    reasons.append(
                        f"impossible_physical_speed:{required_speed_ms:.1f}m/s"
                        f">{self.cfg.max_ball_speed_ms:.0f}m/s"
                        f"({court_dist_m:.2f}m/{time_elapsed_s*1000:.0f}ms)"
                    )
                    print(
                        f"[PHYSICS-REJECT] Leitura impossível: {court_dist_m:.2f}m "
                        f"em {time_elapsed_s*1000:.0f}ms = {required_speed_ms:.0f}m/s "
                        f"(max={self.cfg.max_ball_speed_ms:.0f}m/s) "
                        f"pixel {self.last_observed_center} → {point}"
                    )

            last_side = self.ball_side
            if last_side in (SIDE_A, SIDE_B) and side in (SIDE_A, SIDE_B) and side != last_side:
                if not crosses_net and min(last_dist_net, this_dist_net) > self.cfg.teleport_net_tolerance_px:
                    penalty += self.cfg.side_teleport_penalty
                    reasons.append(f"side_teleport:{last_side}->{side}")

            if self.prev_observed_center is not None:
                v0 = np.array(self.last_observed_center, dtype=np.float32) - np.array(self.prev_observed_center, dtype=np.float32)
                v1 = np.array(point, dtype=np.float32) - np.array(self.last_observed_center, dtype=np.float32)
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 >= self.cfg.abrupt_turn_min_step_px and n1 >= self.cfg.abrupt_turn_min_step_px:
                    cos_turn = float(np.dot(v0, v1) / max(n0 * n1, 1e-6))
                    debug["cos_turn"] = cos_turn
                    if cos_turn <= self.cfg.abrupt_turn_cos and last_dist_net > self.cfg.abrupt_turn_net_grace_px:
                        penalty += self.cfg.abrupt_turn_penalty
                        reasons.append(f"abrupt_turn_far_net:{cos_turn:.2f}")

        reject = penalty >= self.cfg.reject_penalty_threshold
        accepted_for_stats = penalty < self.cfg.stats_penalty_threshold
        if reject:
            quality = "rejected"
        elif not accepted_for_stats:
            quality = "dubious"
        else:
            quality = "trusted"
        return BallRuleDecision(
            quality=quality,
            penalty=penalty,
            reject=reject,
            accepted_for_stats=accepted_for_stats,
            reason=";".join(reasons) if reasons else "ok",
            side=side,
            crosses_net=crosses_net,
            debug=debug,
        )

    def update_ball(
        self,
        ball_state,
        timestamp_s: float,
        frame_idx: int,
    ) -> GameContext:
        track_state = str(getattr(ball_state, "track_state", TRACK_LOST))
        center = getattr(ball_state, "accepted_ball_center", None)
        if center is None and bool(getattr(ball_state, "visible", False)):
            center = getattr(ball_state, "pixel", None)

        frame_events: List[GameEvent] = []
        reasons: List[str] = []
        decision = BallRuleDecision(quality="lost", accepted_for_stats=False, reason="lost")
        rally_start = False
        rally_end_candidate = False

        if track_state == TRACK_OBSERVED and center is not None and bool(getattr(ball_state, "visible", False)):
            point = (float(center[0]), float(center[1]))
            decision = self.evaluate_candidate(point, timestamp_s, frame_idx)
            side = decision.side
            if decision.reason != "ok":
                reasons.append(decision.reason)

            if not decision.reject:
                if not self.rally_active:
                    self.current_rally_id += 1
                    self.rally_active = True
                    self.rally_net_crossings = 0
                    rally_start = True
                    frame_events.append(self._emit("RALLY_START", frame_idx, timestamp_s, center=point))

                if side in (SIDE_A, SIDE_B):
                    self._update_side_and_possession(side, frame_idx, timestamp_s, frame_events)

                self.prev_observed_center = self.last_observed_center
                self.last_observed_center = point
                self.last_observed_frame = int(frame_idx)
                self.last_observed_ts = float(timestamp_s)
                self.lost_since_ts = None
            else:
                frame_events.append(
                    self._emit(
                        "BALL_CONTEXT_REJECTED",
                        frame_idx,
                        timestamp_s,
                        reason=decision.reason,
                        penalty=decision.penalty,
                        center=point,
                    )
                )
        else:
            if self.rally_active and self.lost_since_ts is None:
                self.lost_since_ts = float(timestamp_s)
            if self.rally_active and self.lost_since_ts is not None:
                if timestamp_s - self.lost_since_ts >= self.cfg.rally_lost_confirm_s:
                    rally_end_candidate = True
                    reasons.append("ball_lost_long_enough")

        context = GameContext(
            frame_idx=int(frame_idx),
            timestamp_s=float(timestamp_s),
            current_rally_id=int(self.current_rally_id),
            rally_active=bool(self.rally_active),
            ball_track_state=track_state,
            ball_quality=decision.quality,
            ball_accepted_for_stats=bool(decision.accepted_for_stats and not decision.reject),
            ball_side=self.ball_side,
            possession_side=self.possession_side,
            possession_team=self.possession_team,
            last_net_crossing_frame=self.last_net_crossing_frame,
            last_net_crossing_direction=self.last_net_crossing_direction,
            rally_net_crossings=self.rally_net_crossings,
            last_score=self.last_score,
            score_confirmed_point=False,
            rally_start=rally_start,
            rally_end_candidate=rally_end_candidate,
            rally_end_confirmed=False,
            events=frame_events,
            reasons=reasons,
            debug=decision.debug,
        )
        self.last_context = context
        return context

    def _update_side_and_possession(
        self,
        side: str,
        frame_idx: int,
        timestamp_s: float,
        frame_events: List[GameEvent],
    ) -> None:
        previous_side = self.ball_side
        if previous_side is None:
            self.ball_side = side
            self.candidate_side = side
            self.candidate_side_frames = 1
            self.possession_side = side
            self.possession_team = self._team_for_side(side)
            return

        if side == previous_side:
            self.candidate_side = side
            self.candidate_side_frames = min(self.candidate_side_frames + 1, self.cfg.possession_confirm_frames)
            if self.candidate_side_frames >= self.cfg.possession_confirm_frames:
                self.possession_side = side
                self.possession_team = self._team_for_side(side)
            return

        if self.candidate_side == side:
            self.candidate_side_frames += 1
        else:
            self.candidate_side = side
            self.candidate_side_frames = 1

        if self.candidate_side_frames < self.cfg.net_cross_confirm_frames:
            return

        old_side = previous_side
        self.ball_side = side
        self.possession_side = side
        self.possession_team = self._team_for_side(side)
        self.last_net_crossing_frame = int(frame_idx)
        self.last_net_crossing_direction = f"{old_side}->{side}"
        self.rally_net_crossings += 1
        frame_events.append(
            self._emit(
                "NET_CROSSING",
                frame_idx,
                timestamp_s,
                direction=self.last_net_crossing_direction,
                possession_side=side,
                possession_team=self.possession_team,
            )
        )

    def confirm_score_change(
        self,
        prev_score: Optional[Tuple[int, int, int, int]],
        new_score: Optional[Tuple[int, int, int, int]],
        timestamp_s: float,
        frame_idx: int,
    ) -> Optional[GameEvent]:
        if new_score is None:
            return None
        if self.last_score is None and prev_score is not None:
            self.last_score = prev_score
        logical = self._logical_score_change(prev_score, new_score)
        self.last_score = new_score
        if not logical:
            return None

        winner_team: Optional[str] = None
        if prev_score is not None:
            if new_score[1] > prev_score[1] or new_score[0] > prev_score[0]:
                winner_team = "TeamA"
            elif new_score[3] > prev_score[3] or new_score[2] > prev_score[2]:
                winner_team = "TeamB"
        event = self._emit(
            "POINT_CONFIRMED_BY_SCOREBOARD",
            frame_idx,
            timestamp_s,
            prev_score=prev_score,
            new_score=new_score,
            winner_team=winner_team,
            rally_id=self.current_rally_id,
        )
        self.rally_active = False
        context = self.last_context
        context.score_confirmed_point = True
        context.rally_end_confirmed = True
        context.rally_active = False
        context.last_score = new_score
        context.events.append(event)
        self.last_context = context
        return event

    def reset_for_new_rally(self) -> None:
        self.rally_active = False
        self.ball_side = None
        self.candidate_side = None
        self.candidate_side_frames = 0
        self.possession_side = None
        self.possession_team = None
        self.prev_observed_center = None
        self.last_observed_center = None
        self.last_observed_frame = None
        self.last_observed_ts = None
        self.lost_since_ts = None
        self.last_net_crossing_frame = None
        self.last_net_crossing_direction = None
        self.rally_net_crossings = 0
        self.last_context = GameContext(
            current_rally_id=self.current_rally_id,
            rally_active=False,
            ball_track_state=TRACK_LOST,
            ball_quality="lost",
            ball_accepted_for_stats=False,
            last_score=self.last_score,
            events=self.events[-1:],
        )

    def context_snapshot(self) -> Dict:
        return self.last_context.to_dict()
