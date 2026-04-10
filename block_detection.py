"""
block_detection.py
------------------
Trajectory-driven block detector using calibrated court geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from court_geometry import CourtGeometry, SIDE_A, SIDE_B


@dataclass
class BlockDetectorConfig:
    attack_min_speed_px: float = 6.0
    attack_min_approach_px: float = 10.0
    attack_window_points: int = 6
    net_contact_window_points: int = 8
    reversal_window_points: int = 6
    net_max_dwell_points: int = 4
    return_min_distance_px: float = 12.0
    post_contact_min_speed_px: float = 4.0
    live_window_points: int = 18
    net_zone_tolerance_px: float = 25.0
    debug_log: bool = False

    @classmethod
    def from_config(cls, config) -> "BlockDetectorConfig":
        return cls(
            attack_min_speed_px=float(getattr(config, "block_attack_min_speed_px", 6.0)),
            attack_min_approach_px=float(getattr(config, "block_attack_min_approach_px", 10.0)),
            attack_window_points=int(getattr(config, "block_attack_window_points", 6)),
            net_contact_window_points=int(getattr(config, "block_net_contact_window_points", 8)),
            reversal_window_points=int(getattr(config, "block_reversal_window_points", 6)),
            net_max_dwell_points=int(getattr(config, "block_net_max_dwell_points", 4)),
            return_min_distance_px=float(getattr(config, "block_return_min_distance_px", 12.0)),
            post_contact_min_speed_px=float(getattr(config, "block_post_contact_min_speed_px", 4.0)),
            live_window_points=int(getattr(config, "block_live_window_points", 18)),
            net_zone_tolerance_px=float(getattr(config, "net_band_height_px", 25.0)),
            debug_log=bool(getattr(config, "BLOCK_DEBUG_LOG", False)),
        )


@dataclass
class TrajectorySample:
    idx: int
    point: Tuple[float, float]
    timestamp_s: float
    speed_px: float
    side: Optional[str]
    dist_to_net: float
    signed_dist_to_net: float
    in_net_zone: bool
    inside_court: bool


@dataclass
class BlockAssessment:
    state: str = "idle"
    event_type: Optional[str] = None
    attack_side: Optional[str] = None
    attack_team: Optional[str] = None
    defending_side: Optional[str] = None
    blocking_side: Optional[str] = None
    blocking_team: Optional[str] = None
    end_side: Optional[str] = None
    return_side: Optional[str] = None
    crossed_net: bool = False
    attack_confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    signals: Dict = field(default_factory=dict)

    def reason_text(self) -> str:
        return ";".join(self.reasons) if self.reasons else "ok"

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "event_type": self.event_type,
            "attack_side": self.attack_side,
            "attack_team": self.attack_team,
            "defending_side": self.defending_side,
            "blocking_side": self.blocking_side,
            "blocking_team": self.blocking_team,
            "end_side": self.end_side,
            "return_side": self.return_side,
            "crossed_net": bool(self.crossed_net),
            "attack_confidence": float(self.attack_confidence),
            "reason": self.reason_text(),
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


class BlockDetector:
    def __init__(self, geometry: CourtGeometry, cfg: BlockDetectorConfig) -> None:
        self.geometry = geometry
        self.cfg = cfg
        self.last_live_assessment = BlockAssessment()
        self.last_final_assessment = BlockAssessment()

    def reset(self) -> None:
        self.last_live_assessment = BlockAssessment()
        self.last_final_assessment = BlockAssessment()

    def update_live(
        self,
        drawer: Sequence[Tuple[float, float, float]],
        attacker_side_hint: Optional[str] = None,
        attacker_team_hint: Optional[str] = None,
    ) -> BlockAssessment:
        trimmed = list(drawer)[-max(2, int(self.cfg.live_window_points)) :]
        assessment = self.analyze(
            trimmed,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            winner_team=None,
        )
        if self.cfg.debug_log and assessment.state != self.last_live_assessment.state:
            print(
                f"[BLOCK] state={assessment.state} attack={assessment.attack_side} "
                f"end={assessment.end_side} reason={assessment.reason_text()}"
            )
        self.last_live_assessment = assessment
        return assessment

    def finalize(
        self,
        drawer: Sequence[Tuple[float, float, float]],
        attacker_side_hint: Optional[str] = None,
        attacker_team_hint: Optional[str] = None,
        winner_team: Optional[str] = None,
    ) -> BlockAssessment:
        assessment = self.analyze(
            drawer,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            winner_team=winner_team,
        )
        if self.cfg.debug_log:
            print(
                f"[BLOCK-FINAL] state={assessment.state} event={assessment.event_type} "
                f"attack={assessment.attack_side} end={assessment.end_side} reason={assessment.reason_text()}"
            )
        self.last_final_assessment = assessment
        return assessment

    def snapshot(self) -> Dict:
        snapshot = self.last_live_assessment.to_dict()
        snapshot["final"] = self.last_final_assessment.to_dict()
        return snapshot

    def analyze(
        self,
        drawer: Sequence[Tuple[float, float, float]],
        attacker_side_hint: Optional[str] = None,
        attacker_team_hint: Optional[str] = None,
        winner_team: Optional[str] = None,
    ) -> BlockAssessment:
        samples = self._build_samples(drawer)
        if len(samples) < 2:
            return BlockAssessment(state="idle", reasons=["insufficient_trajectory"])

        end_side = self._last_side(samples)
        crossed_net = self._crossed_net(samples)
        attack_signal = self._latest_attack_signal(samples, attacker_side_hint, attacker_team_hint)
        contact_signal = self._latest_contact_signal(samples, attacker_side_hint, attacker_team_hint)

        if contact_signal is None:
            if attack_signal is None:
                return BlockAssessment(
                    state="idle",
                    end_side=end_side,
                    crossed_net=crossed_net,
                    reasons=["no_attack_pattern"],
                    signals={
                        "samples": len(samples),
                        "attacker_side_hint": attacker_side_hint,
                        "attacker_team_hint": attacker_team_hint,
                    },
                )
            return BlockAssessment(
                state="attack_candidate",
                attack_side=attack_signal["attack_side"],
                attack_team=attack_signal.get("attack_team"),
                defending_side=self.geometry.other_side(attack_signal["attack_side"]),
                blocking_side=self.geometry.other_side(attack_signal["attack_side"]),
                blocking_team=self._team_for_side(self.geometry.other_side(attack_signal["attack_side"])),
                end_side=end_side,
                return_side=end_side,
                crossed_net=crossed_net,
                attack_confidence=float(attack_signal["attack_confidence"]),
                reasons=["attack_towards_net_detected", "attacker_from_current_possession"],
                signals=dict(attack_signal),
            )

        attack_side = contact_signal["attack_side"]
        attack_team = contact_signal.get("attack_team") or self._team_for_side(attack_side)
        defending_side = self.geometry.other_side(attack_side)
        if defending_side is None:
            defending_side = SIDE_B if attack_side == SIDE_A else SIDE_A
        blocking_side = defending_side
        blocking_team = self._team_for_side(blocking_side)

        reasons: List[str] = []
        signals = dict(contact_signal)
        attack_confidence = float(contact_signal.get("attack_confidence", 0.0))
        dwell_points = int(contact_signal["net_dwell_points"])
        returned_sample = contact_signal.get("returned_sample")
        crossed_before_return = bool(contact_signal.get("crossed_before_return", False))
        post_contact_speed_peak = float(contact_signal.get("post_contact_speed_peak", 0.0))
        return_side = returned_sample.side if returned_sample is not None else end_side
        clean_reversal = bool(
            returned_sample is not None
            and not crossed_before_return
            and dwell_points <= self.cfg.net_max_dwell_points
            and post_contact_speed_peak >= self.cfg.post_contact_min_speed_px
        )

        if clean_reversal:
            reasons.extend(
                [
                    "attack_candidate",
                    "net_contact",
                    "trajectory_reversal",
                    "returned_to_attack_side",
                ]
            )
            if winner_team == self._team_for_side(defending_side):
                return BlockAssessment(
                    state="confirmed_block",
                    event_type="POINT_BY_BLOCK",
                    attack_side=attack_side,
                    attack_team=attack_team,
                    defending_side=defending_side,
                    blocking_side=blocking_side,
                    blocking_team=blocking_team,
                    end_side=end_side,
                    return_side=return_side,
                    crossed_net=crossed_net,
                    attack_confidence=attack_confidence,
                    reasons=reasons + ["scoreboard_confirms_defender_point"],
                    signals=signals,
                )
            event_type = None
            state = "block_candidate" if winner_team is None else "possible_block"
            if winner_team is not None and winner_team != self._team_for_side(defending_side):
                reasons.append("scoreboard_did_not_confirm_block_point")
            return BlockAssessment(
                state=state,
                event_type=event_type,
                attack_side=attack_side,
                attack_team=attack_team,
                defending_side=defending_side,
                blocking_side=blocking_side,
                blocking_team=blocking_team,
                end_side=end_side,
                return_side=return_side,
                crossed_net=crossed_net,
                attack_confidence=attack_confidence,
                reasons=reasons,
                signals=signals,
            )

        reasons.extend(["attack_candidate", "net_contact"])
        if dwell_points > self.cfg.net_max_dwell_points:
            reasons.append("net_zone_dwell_too_long")
        if returned_sample is None:
            reasons.append("no_clean_reversal")
        if crossed_before_return:
            reasons.append("crossed_to_defender_side")
        if post_contact_speed_peak < self.cfg.post_contact_min_speed_px:
            reasons.append("post_contact_speed_too_low")

        event_type = None
        if winner_team == self._team_for_side(defending_side):
            event_type = "BOLA_NA_REDE"
        return BlockAssessment(
            state="net_contact",
            event_type=event_type,
            attack_side=attack_side,
            attack_team=attack_team,
            defending_side=defending_side,
            blocking_side=blocking_side,
            blocking_team=blocking_team,
            end_side=end_side,
            return_side=return_side,
            crossed_net=crossed_net,
            attack_confidence=attack_confidence,
            reasons=reasons,
            signals=signals,
        )

    def _build_samples(self, drawer: Sequence[Tuple[float, float, float]]) -> List[TrajectorySample]:
        ordered = sorted(
            [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in drawer
                if point is not None and len(point) >= 3
            ],
            key=lambda item: float(item[2]),
        )
        samples: List[TrajectorySample] = []
        prev_point: Optional[Tuple[float, float]] = None
        for idx, (x, y, ts) in enumerate(ordered):
            point = (float(x), float(y))
            speed_px = 0.0 if prev_point is None else float(np.hypot(point[0] - prev_point[0], point[1] - prev_point[1]))
            side = self.geometry.get_side_of_net(point, neutral_tolerance_px=1.0)
            signed_dist = float(self.geometry.signed_distance_to_net(point))
            dist_to_net = abs(signed_dist)
            samples.append(
                TrajectorySample(
                    idx=idx,
                    point=point,
                    timestamp_s=float(ts),
                    speed_px=float(speed_px),
                    side=side,
                    dist_to_net=float(dist_to_net),
                    signed_dist_to_net=float(signed_dist),
                    in_net_zone=bool(self.geometry.is_in_net_zone(point, tolerance_px=self.cfg.net_zone_tolerance_px)),
                    inside_court=bool(self.geometry.is_inside_court(point, point_space="pixel")),
                )
            )
            prev_point = point
        return samples

    def _latest_attack_signal(
        self,
        samples: Sequence[TrajectorySample],
        attacker_side_hint: Optional[str],
        attacker_team_hint: Optional[str],
    ) -> Optional[Dict]:
        if len(samples) < 2:
            return None
        tail = list(samples)[-max(2, int(self.cfg.attack_window_points) + 1) :]
        visible = [sample for sample in tail if sample.side in (SIDE_A, SIDE_B) and not sample.in_net_zone and sample.inside_court]
        if len(visible) < 2:
            return None

        attack_side = visible[-1].side
        cluster: List[TrajectorySample] = []
        for sample in reversed(visible):
            if sample.side != attack_side:
                break
            cluster.append(sample)
        cluster.reverse()
        if len(cluster) < 2:
            return None

        approach_gain = float(cluster[0].dist_to_net - cluster[-1].dist_to_net)
        speed_peak = max((sample.speed_px for sample in cluster[1:]), default=0.0)
        towards_steps = sum(
            1
            for prev, curr in zip(cluster, cluster[1:])
            if prev.dist_to_net > (curr.dist_to_net + 0.5)
        )
        if approach_gain < self.cfg.attack_min_approach_px or speed_peak < self.cfg.attack_min_speed_px or towards_steps <= 0:
            return None

        confidence = 0.0
        confidence += min(0.45, approach_gain / max(self.cfg.attack_min_approach_px * 2.0, 1.0))
        confidence += min(0.45, speed_peak / max(self.cfg.attack_min_speed_px * 3.0, 1.0))
        if attack_side == attacker_side_hint:
            confidence += 0.10
        return {
            "attack_side": attack_side,
            "attack_team": attacker_team_hint or self._team_for_side(attack_side),
            "attack_confidence": float(min(confidence, 1.0)),
            "attack_frame": int(cluster[-1].idx),
            "attack_start_idx": int(cluster[0].idx),
            "attack_end_idx": int(cluster[-1].idx),
            "attack_vector": (
                float(cluster[-1].point[0] - cluster[0].point[0]),
                float(cluster[-1].point[1] - cluster[0].point[1]),
            ),
            "approach_gain": float(approach_gain),
            "attack_speed_peak": float(speed_peak),
            "towards_steps": int(towards_steps),
        }

    def _latest_contact_signal(
        self,
        samples: Sequence[TrajectorySample],
        attacker_side_hint: Optional[str],
        attacker_team_hint: Optional[str],
    ) -> Optional[Dict]:
        if len(samples) < 3:
            return None
        max_contact_window = max(2, int(self.cfg.net_contact_window_points))
        start_idx = max(1, len(samples) - max_contact_window)

        for idx in range(len(samples) - 1, start_idx - 1, -1):
            sample = samples[idx]
            prev_sample = samples[idx - 1]
            contact_here = sample.in_net_zone or self.geometry.segment_hits_net_zone(
                prev_sample.point,
                sample.point,
                tolerance_px=self.cfg.net_zone_tolerance_px,
            )
            if not contact_here:
                continue

            attack_signal = self._attack_signal_before_contact(samples, idx, attacker_side_hint, attacker_team_hint)
            if attack_signal is None:
                continue

            attack_side = attack_signal["attack_side"]
            contact_distance = min(prev_sample.dist_to_net, sample.dist_to_net)
            dwell_points = 1 if sample.in_net_zone else 0
            post_contact_speed_peak = 0.0
            returned_sample = None
            crossed_before_return = False

            post_limit = min(
                len(samples),
                idx + 1 + int(self.cfg.net_max_dwell_points) + int(self.cfg.reversal_window_points),
            )
            for post_idx in range(idx + 1, post_limit):
                post_sample = samples[post_idx]
                if post_sample.in_net_zone:
                    dwell_points += 1
                    continue
                if post_sample.side not in (SIDE_A, SIDE_B):
                    continue
                post_contact_speed_peak = max(post_contact_speed_peak, float(post_sample.speed_px))
                if post_sample.side != attack_side:
                    crossed_before_return = True
                    continue
                moved_away = post_sample.dist_to_net >= (contact_distance + self.cfg.return_min_distance_px)
                strong_return = max(post_sample.speed_px, post_sample.dist_to_net - contact_distance) >= self.cfg.post_contact_min_speed_px
                if moved_away and strong_return:
                    returned_sample = post_sample
                    post_contact_speed_peak = max(post_contact_speed_peak, float(post_sample.speed_px))
                    break

            return {
                **attack_signal,
                "contact_idx": int(idx),
                "contact_point": sample.point,
                "contact_distance": float(contact_distance),
                "net_dwell_points": int(dwell_points),
                "post_contact_speed_peak": float(post_contact_speed_peak),
                "returned_sample": returned_sample,
                "crossed_before_return": bool(crossed_before_return),
            }
        return None

    def _attack_signal_before_contact(
        self,
        samples: Sequence[TrajectorySample],
        contact_idx: int,
        attacker_side_hint: Optional[str],
        attacker_team_hint: Optional[str],
    ) -> Optional[Dict]:
        if contact_idx < 1:
            return None
        window_start = max(0, contact_idx - int(self.cfg.attack_window_points))
        window = list(samples[window_start : contact_idx + 1])
        non_net = [sample for sample in window if sample.side in (SIDE_A, SIDE_B) and not sample.in_net_zone and sample.inside_court]
        if len(non_net) < 2:
            return None

        attack_side = non_net[-1].side
        cluster: List[TrajectorySample] = []
        for sample in reversed(non_net):
            if sample.side != attack_side:
                break
            cluster.append(sample)
        cluster.reverse()
        if len(cluster) < 2:
            return None

        approach_gain = float(cluster[0].dist_to_net - cluster[-1].dist_to_net)
        speed_peak = max((sample.speed_px for sample in cluster[1:]), default=0.0)
        towards_steps = sum(
            1
            for prev, curr in zip(cluster, cluster[1:])
            if prev.dist_to_net > (curr.dist_to_net + 0.5)
        )
        if approach_gain < self.cfg.attack_min_approach_px or speed_peak < self.cfg.attack_min_speed_px or towards_steps <= 0:
            return None

        confidence = 0.0
        confidence += min(0.45, approach_gain / max(self.cfg.attack_min_approach_px * 2.0, 1.0))
        confidence += min(0.45, speed_peak / max(self.cfg.attack_min_speed_px * 3.0, 1.0))
        if attack_side == attacker_side_hint:
            confidence += 0.10
        return {
            "attack_side": attack_side,
            "attack_team": attacker_team_hint or self._team_for_side(attack_side),
            "attack_confidence": float(min(confidence, 1.0)),
            "attack_frame": int(cluster[-1].idx),
            "attack_start_idx": int(cluster[0].idx),
            "attack_end_idx": int(cluster[-1].idx),
            "attack_vector": (
                float(cluster[-1].point[0] - cluster[0].point[0]),
                float(cluster[-1].point[1] - cluster[0].point[1]),
            ),
            "approach_gain": float(approach_gain),
            "attack_speed_peak": float(speed_peak),
            "towards_steps": int(towards_steps),
        }

    def _crossed_net(self, samples: Sequence[TrajectorySample]) -> bool:
        previous_side: Optional[str] = None
        for sample in samples:
            if sample.side not in (SIDE_A, SIDE_B) or sample.in_net_zone:
                continue
            if previous_side is not None and sample.side != previous_side:
                return True
            previous_side = sample.side
        return False

    @staticmethod
    def _last_side(samples: Sequence[TrajectorySample]) -> Optional[str]:
        for sample in reversed(samples):
            if sample.side in (SIDE_A, SIDE_B):
                return sample.side
        return None

    @staticmethod
    def _team_for_side(side: Optional[str]) -> Optional[str]:
        if side == SIDE_A:
            return "TeamA"
        if side == SIDE_B:
            return "TeamB"
        return None
