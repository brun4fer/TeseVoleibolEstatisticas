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
    gap_min_missing_frames: int = 3
    gap_near_net_tolerance_px: float = 40.0
    gap_jump_min_distance_px: float = 10.0
    gap_return_min_distance_px: float = 6.0
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
            gap_min_missing_frames=int(getattr(config, "block_gap_min_missing_frames", 3)),
            gap_near_net_tolerance_px=float(
                getattr(
                    config,
                    "block_gap_near_net_tolerance_px",
                    float(getattr(config, "net_band_height_px", 25.0)) + 15.0,
                )
            ),
            gap_jump_min_distance_px=float(getattr(config, "block_gap_jump_min_distance_px", 10.0)),
            gap_return_min_distance_px=float(getattr(config, "block_gap_return_min_distance_px", 6.0)),
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
        visible_trajectory: Optional[Sequence[Tuple[int, float, float]]] = None,
    ) -> BlockAssessment:
        assessment = self.analyze(
            drawer,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            winner_team=winner_team,
            visible_trajectory=visible_trajectory,
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
        visible_trajectory: Optional[Sequence[Tuple[int, float, float]]] = None,
    ) -> BlockAssessment:
        samples = self._build_samples(drawer)
        if len(samples) < 2:
            return BlockAssessment(state="idle", reasons=["insufficient_trajectory"])

        origin_side = self._first_side(samples)
        end_side = self._last_side(samples)
        crossed_net = self._crossed_net(samples)
        attack_signal = self._latest_attack_signal(samples, attacker_side_hint, attacker_team_hint)
        contact_signal = self._latest_contact_signal(samples, attacker_side_hint, attacker_team_hint)
        gap_signal = self._gap_contact_signal(visible_trajectory, attacker_side_hint, attacker_team_hint)
        if gap_signal is not None:
            crossed_net = bool(crossed_net or gap_signal.get("crossed_net_inferred", False))
            if attack_signal is None or bool(gap_signal.get("attack_inferred", False)):
                attack_signal = dict(gap_signal)
            if contact_signal is None or (
                bool(gap_signal.get("block_from_gap", False))
                and not bool(contact_signal.get("trajectory_reversed", False))
            ):
                contact_signal = dict(gap_signal)

        if contact_signal is None:
            if attack_signal is None:
                return BlockAssessment(
                    state="idle",
                    end_side=end_side,
                    crossed_net=crossed_net,
                    reasons=["no_attack_pattern"],
                    signals={
                        "samples": len(samples),
                        "origin_side": origin_side,
                        "end_side": end_side,
                        "approach_detected": False,
                        "net_contact": False,
                        "trajectory_reversed": False,
                        "block_candidate": False,
                        "gap_near_net": False,
                        "attacker_side_hint": attacker_side_hint,
                        "attacker_team_hint": attacker_team_hint,
                    },
                )
            attack_reasons = ["attack_towards_net_detected"]
            if attack_signal.get("attack_inferred", False):
                attack_reasons.append("attack_inferred_from_trajectory")
            elif attack_signal["attack_side"] == attacker_side_hint:
                attack_reasons.append("attacker_from_current_possession")
            signals = dict(attack_signal)
            signals.update(
                {
                    "origin_side": attack_signal["attack_side"],
                    "end_side": end_side,
                    "approach_detected": bool(attack_signal.get("approach_detected", False)),
                    "net_contact": False,
                    "trajectory_reversed": False,
                    "block_candidate": False,
                }
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
                reasons=attack_reasons,
                signals=signals,
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
        approach_detected = bool(contact_signal.get("approach_detected", False))
        net_contact = bool(contact_signal.get("net_contact", True))
        trajectory_reversed = bool(contact_signal.get("trajectory_reversed", False))
        dwell_points = int(contact_signal["net_dwell_points"])
        returned_sample = contact_signal.get("returned_sample")
        crossed_before_return = bool(contact_signal.get("crossed_before_return", False))
        post_contact_speed_peak = float(contact_signal.get("post_contact_speed_peak", 0.0))
        return_distance_gain = float(contact_signal.get("return_distance_gain", 0.0))
        origin_side = contact_signal.get("origin_side") or attack_side
        return_side = returned_sample.side if returned_sample is not None else end_side
        same_side_return = bool(origin_side in (SIDE_A, SIDE_B) and return_side == origin_side)
        block_candidate = bool(approach_detected and net_contact and trajectory_reversed and same_side_return)
        reversal_speed_ok = bool(
            post_contact_speed_peak >= self.cfg.post_contact_min_speed_px
            or return_distance_gain >= self.cfg.return_min_distance_px
        )
        clean_reversal = bool(
            block_candidate
            and not crossed_before_return
            and dwell_points <= self.cfg.net_max_dwell_points
            and reversal_speed_ok
        )
        signals.update(
            {
                "origin_side": origin_side,
                "end_side": end_side,
                "approach_detected": approach_detected,
                "net_contact": net_contact,
                "trajectory_reversed": trajectory_reversed,
                "block_candidate": block_candidate,
                "same_side_return": same_side_return,
            }
        )

        if clean_reversal:
            reasons.extend(
                [
                    "approach_detected",
                    "net_contact",
                    "trajectory_reversal",
                    "returned_to_origin_side",
                ]
            )
            if contact_signal.get("attack_inferred", False):
                reasons.append("attack_inferred_from_trajectory")
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

        if approach_detected:
            reasons.append("approach_detected")
        else:
            reasons.append("no_attack_pattern")
        if net_contact:
            reasons.append("net_contact")
        if dwell_points > self.cfg.net_max_dwell_points:
            reasons.append("net_zone_dwell_too_long")
        if not trajectory_reversed:
            reasons.append("no_clean_reversal")
        if crossed_before_return:
            reasons.append("crossed_to_defender_side")
        if not same_side_return:
            reasons.append("did_not_return_to_origin_side")
        if not reversal_speed_ok:
            reasons.append("post_contact_speed_too_low")
        if contact_signal.get("attack_inferred", False):
            reasons.append("attack_inferred_from_trajectory")

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
        strict_signal = self._attack_signal_from_cluster(
            cluster,
            attack_side=attack_side,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            inferred=False,
            strict=True,
        )
        if strict_signal is not None:
            return strict_signal
        return self._attack_signal_from_cluster(
            cluster,
            attack_side=attack_side,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            inferred=True,
            strict=False,
        )

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
            proximity_contact = min(prev_sample.dist_to_net, sample.dist_to_net) <= self._contact_proximity_threshold()
            segment_hits = self.geometry.segment_hits_net_zone(
                prev_sample.point,
                sample.point,
                tolerance_px=self.cfg.net_zone_tolerance_px,
            )
            contact_here = sample.in_net_zone or segment_hits or proximity_contact
            if not contact_here:
                continue

            attack_signal = self._attack_signal_before_contact(samples, idx, attacker_side_hint, attacker_team_hint)
            if attack_signal is None:
                attack_signal = self._infer_attack_before_contact(samples, idx, attacker_side_hint, attacker_team_hint)
            if attack_signal is None:
                continue

            attack_side = attack_signal["attack_side"]
            contact_distance = min(prev_sample.dist_to_net, sample.dist_to_net)
            dwell_points = 1 if sample.in_net_zone else 0
            post_contact_speed_peak = 0.0
            returned_sample = None
            crossed_before_return = False
            trajectory_reversed = False
            max_return_distance_gain = 0.0
            max_post_contact_dist = contact_distance
            net_zone_hits = int(sample.in_net_zone) + int(prev_sample.in_net_zone) + int(segment_hits)
            reversal_reason = "no_reversal_detected"

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
                return_distance_gain = float(post_sample.dist_to_net - contact_distance)
                max_return_distance_gain = max(max_return_distance_gain, return_distance_gain)
                max_post_contact_dist = max(max_post_contact_dist, float(post_sample.dist_to_net))
                moved_away = return_distance_gain >= self._soft_return_min_distance()
                strong_return = max(post_sample.speed_px, return_distance_gain) >= self._soft_post_contact_speed()
                if moved_away:
                    trajectory_reversed = True
                    reversal_reason = "dist_to_net_decreased_then_increased"
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
                "origin_side": attack_signal["attack_side"],
                "net_contact": True,
                "trajectory_reversed": bool(trajectory_reversed),
                "return_distance_gain": float(max_return_distance_gain),
                "min_dist_to_net": float(contact_distance),
                "net_zone_hits": int(net_zone_hits),
                "reversal_reason": reversal_reason,
                "gap_near_net": False,
                "block_from_gap": False,
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
        return self._attack_signal_from_cluster(
            cluster,
            attack_side=attack_side,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            inferred=False,
            strict=True,
        )

    def _infer_attack_before_contact(
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
        non_net = [sample for sample in window if sample.side in (SIDE_A, SIDE_B) and sample.inside_court]
        if not non_net:
            return None

        attack_side = attacker_side_hint if attacker_side_hint in (SIDE_A, SIDE_B) else non_net[-1].side
        cluster: List[TrajectorySample] = []
        for sample in reversed(non_net):
            if sample.side != attack_side:
                break
            cluster.append(sample)
        cluster.reverse()
        if not cluster:
            return None

        contact_sample = window[-1]
        if cluster[-1].idx != contact_sample.idx:
            cluster.append(
                TrajectorySample(
                    idx=int(contact_sample.idx),
                    point=contact_sample.point,
                    timestamp_s=float(contact_sample.timestamp_s),
                    speed_px=float(contact_sample.speed_px),
                    side=attack_side,
                    dist_to_net=float(contact_sample.dist_to_net),
                    signed_dist_to_net=float(contact_sample.signed_dist_to_net),
                    in_net_zone=bool(contact_sample.in_net_zone),
                    inside_court=bool(contact_sample.inside_court),
                )
            )
        return self._attack_signal_from_cluster(
            cluster,
            attack_side=attack_side,
            attacker_side_hint=attacker_side_hint,
            attacker_team_hint=attacker_team_hint,
            inferred=True,
            strict=False,
        )

    def _attack_signal_from_cluster(
        self,
        cluster: Sequence[TrajectorySample],
        attack_side: Optional[str],
        attacker_side_hint: Optional[str],
        attacker_team_hint: Optional[str],
        inferred: bool,
        strict: bool,
    ) -> Optional[Dict]:
        if attack_side not in (SIDE_A, SIDE_B) or len(cluster) < 2:
            return None

        metrics = self._approach_metrics(cluster)
        approach_gain = float(metrics["approach_gain"])
        speed_peak = max((sample.speed_px for sample in cluster[1:]), default=0.0)
        towards_steps = int(metrics["towards_steps"])

        if strict:
            min_approach = float(self.cfg.attack_min_approach_px)
            min_speed = float(self.cfg.attack_min_speed_px)
        else:
            min_approach = self._soft_attack_min_approach()
            min_speed = self._soft_attack_min_speed()

        majority_approach = bool(metrics["majority_towards"])
        if approach_gain < min_approach or towards_steps < 1:
            if strict or not majority_approach:
                return None
        if strict and not majority_approach and towards_steps < 2:
            return None
        if strict and speed_peak < min_speed:
            return None
        if not strict and speed_peak < min_speed and approach_gain < (min_approach * 1.5):
            if not majority_approach:
                return None

        confidence = 0.25 if inferred else 0.0
        confidence += min(0.40, approach_gain / max(min_approach * 2.0, 1.0))
        confidence += min(0.30, speed_peak / max(max(min_speed, 1.0) * 3.0, 1.0))
        if attack_side == attacker_side_hint:
            confidence += 0.15
        approach_reason = "majority_decreasing_dist_to_net" if majority_approach else "gain_and_speed_thresholds"
        attack_inferred_reason = "relaxed_approach_from_current_possession" if inferred else "explicit_attack_pattern"
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
            "decreasing_steps": int(metrics["decreasing_steps"]),
            "min_dist_to_net": float(metrics["min_dist_to_net"]),
            "net_zone_hits": int(sum(1 for sample in cluster if sample.in_net_zone)),
            "approach_detected": True,
            "approach_reason": approach_reason,
            "attack_inferred": bool(inferred),
            "attack_inferred_reason": attack_inferred_reason,
        }

    def _gap_contact_signal(
        self,
        visible_trajectory: Optional[Sequence[Tuple[int, float, float]]],
        attacker_side_hint: Optional[str],
        attacker_team_hint: Optional[str],
    ) -> Optional[Dict]:
        if not visible_trajectory or len(visible_trajectory) < 2:
            return None

        ordered = sorted(
            [
                (int(point[0]), float(point[1]), float(point[2]))
                for point in visible_trajectory
                if point is not None and len(point) >= 3
            ],
            key=lambda item: int(item[0]),
        )
        if len(ordered) < 2:
            return None

        visible_samples: List[Dict] = []
        for frame_idx, x, y in ordered:
            point = (float(x), float(y))
            side = self.geometry.get_side_of_net(point, neutral_tolerance_px=1.0)
            visible_samples.append(
                {
                    "frame_idx": int(frame_idx),
                    "point": point,
                    "side": side,
                    "dist_to_net": float(self.geometry.distance_to_net(point)),
                    "in_net_zone": bool(self.geometry.is_in_net_zone(point, tolerance_px=self.cfg.net_zone_tolerance_px)),
                }
            )

        end_side = None
        for sample in reversed(visible_samples):
            if sample["side"] in (SIDE_A, SIDE_B):
                end_side = sample["side"]
                break

        for idx in range(len(visible_samples) - 2, -1, -1):
            before = visible_samples[idx]
            after = visible_samples[idx + 1]
            gap_frames = max(0, int(after["frame_idx"]) - int(before["frame_idx"]) - 1)
            if gap_frames < int(self.cfg.gap_min_missing_frames):
                continue

            before_side = before["side"]
            after_side = after["side"]
            # Quando a bola está mesmo junto à rede, get_side_of_net(tolerance=1px)
            # pode retornar None. Usamos o sinal da distância como fallback.
            if before_side not in (SIDE_A, SIDE_B):
                signed = float(self.geometry.signed_distance_to_net(before["point"]))
                before_side = SIDE_A if signed > 0 else (SIDE_B if signed < 0 else None)
            if after_side not in (SIDE_A, SIDE_B):
                signed = float(self.geometry.signed_distance_to_net(after["point"]))
                after_side = SIDE_A if signed > 0 else (SIDE_B if signed < 0 else None)
            if before_side not in (SIDE_A, SIDE_B) or after_side not in (SIDE_A, SIDE_B):
                continue
            if before_side != after_side:
                continue

            before_dist = float(before["dist_to_net"])
            after_dist = float(after["dist_to_net"])
            min_dist_to_net = min(before_dist, after_dist)
            gap_near_net = bool(min_dist_to_net <= self._gap_near_net_threshold())
            if not gap_near_net:
                continue

            jump_distance = float(np.hypot(after["point"][0] - before["point"][0], after["point"][1] - before["point"][1]))
            if jump_distance < float(self.cfg.gap_jump_min_distance_px):
                continue

            cluster_before = self._visible_cluster_before_gap(visible_samples, idx, before_side)
            approach_metrics = self._visible_approach_metrics(cluster_before)
            approach_detected = bool(
                approach_metrics["approach_gain"] >= self._soft_attack_min_approach()
                or (
                    approach_metrics["towards_steps"] >= 1
                    and approach_metrics["majority_towards"]
                )
            )
            approach_reason = "pre_gap_dist_to_net_decreasing"
            if not approach_detected and attacker_side_hint == before_side:
                approach_detected = True
                approach_reason = "possession_hint_before_gap"

            post_cluster = self._visible_cluster_after_gap(visible_samples, idx + 1, before_side)
            reversal_stats = self._visible_reversal_metrics(before_dist, after_dist, post_cluster)
            trajectory_reversed = bool(reversal_stats["reversed"])
            same_side_return = bool(end_side == before_side)
            block_from_gap = bool(gap_near_net and same_side_return and trajectory_reversed)
            if not block_from_gap and not approach_detected:
                continue

            if block_from_gap:
                print(f"[BLOCK-INFERRED] Gap de {gap_frames} frames junto à rede, bola regressou ao {before_side}")

            attack_team = attacker_team_hint or self._team_for_side(before_side)
            attack_inferred_reason = "current_possession_gap_same_side_return" if attacker_team_hint in ("TeamA", "TeamB") else "team_from_origin_side_gap_same_side_return"
            return {
                "attack_side": before_side,
                "attack_team": attack_team,
                "attack_confidence": float(min(0.55 + (0.1 if approach_detected else 0.0), 0.9)),
                "attack_frame": int(before["frame_idx"]),
                "attack_start_idx": int(cluster_before[0]["frame_idx"]) if cluster_before else int(before["frame_idx"]),
                "attack_end_idx": int(before["frame_idx"]),
                "attack_vector": (
                    float(after["point"][0] - before["point"][0]),
                    float(after["point"][1] - before["point"][1]),
                ),
                "approach_gain": float(approach_metrics["approach_gain"]),
                "attack_speed_peak": float(max(approach_metrics["speed_peak"], reversal_stats["jump_speed"])),
                "towards_steps": int(approach_metrics["towards_steps"]),
                "decreasing_steps": int(approach_metrics["decreasing_steps"]),
                "approach_detected": bool(approach_detected),
                "approach_reason": approach_reason,
                "attack_inferred": True,
                "attack_inferred_reason": attack_inferred_reason,
                "contact_idx": int(before["frame_idx"]),
                "contact_point": before["point"] if before_dist <= after_dist else after["point"],
                "contact_distance": float(min_dist_to_net),
                "net_dwell_points": 1,
                "post_contact_speed_peak": float(reversal_stats["jump_speed"]),
                "returned_sample": None,
                "crossed_before_return": False,
                "origin_side": before_side,
                "net_contact": True,
                "trajectory_reversed": bool(trajectory_reversed),
                "return_distance_gain": float(reversal_stats["return_distance_gain"]),
                "min_dist_to_net": float(min_dist_to_net),
                "net_zone_hits": int(int(before["in_net_zone"]) + int(after["in_net_zone"]) + 1),
                "reversal_reason": reversal_stats["reason"],
                "gap_near_net": True,
                "gap_frames": int(gap_frames),
                "before_gap_point": before["point"],
                "after_gap_point": after["point"],
                "block_from_gap": bool(block_from_gap),
                "same_side_return": bool(same_side_return),
                "crossed_net_inferred": bool(block_from_gap),
                "end_side": end_side,
            }
        return None

    @staticmethod
    def _approach_metrics(cluster: Sequence[TrajectorySample]) -> Dict[str, float]:
        if len(cluster) < 2:
            return {
                "approach_gain": 0.0,
                "towards_steps": 0,
                "decreasing_steps": 0,
                "majority_towards": False,
                "min_dist_to_net": min((sample.dist_to_net for sample in cluster), default=0.0),
            }
        decreasing_steps = sum(
            1
            for prev, curr in zip(cluster, cluster[1:])
            if prev.dist_to_net > (curr.dist_to_net + 0.5)
        )
        transitions = max(1, len(cluster) - 1)
        return {
            "approach_gain": float(cluster[0].dist_to_net - cluster[-1].dist_to_net),
            "towards_steps": int(decreasing_steps),
            "decreasing_steps": int(decreasing_steps),
            "majority_towards": bool(decreasing_steps >= max(1, int(np.ceil(transitions * 0.5)))),
            "min_dist_to_net": float(min(sample.dist_to_net for sample in cluster)),
        }

    def _visible_cluster_before_gap(self, visible_samples: Sequence[Dict], gap_start_idx: int, side: str) -> List[Dict]:
        start_idx = max(0, int(gap_start_idx) - int(self.cfg.attack_window_points))
        cluster: List[Dict] = []
        for sample in visible_samples[start_idx : gap_start_idx + 1]:
            if sample["side"] == side:
                cluster.append(sample)
        return cluster

    def _visible_cluster_after_gap(self, visible_samples: Sequence[Dict], start_idx: int, side: str) -> List[Dict]:
        limit = min(len(visible_samples), int(start_idx) + int(self.cfg.reversal_window_points) + 1)
        cluster: List[Dict] = []
        for sample in visible_samples[start_idx:limit]:
            if sample["side"] == side:
                cluster.append(sample)
            elif cluster:
                break
        return cluster

    def _visible_approach_metrics(self, cluster: Sequence[Dict]) -> Dict[str, float]:
        if len(cluster) < 2:
            return {
                "approach_gain": 0.0,
                "towards_steps": 0,
                "decreasing_steps": 0,
                "majority_towards": False,
                "speed_peak": 0.0,
            }
        decreasing_steps = 0
        speed_peak = 0.0
        for prev, curr in zip(cluster, cluster[1:]):
            if float(prev["dist_to_net"]) > (float(curr["dist_to_net"]) + 0.5):
                decreasing_steps += 1
            speed_peak = max(speed_peak, float(np.hypot(curr["point"][0] - prev["point"][0], curr["point"][1] - prev["point"][1])))
        transitions = max(1, len(cluster) - 1)
        return {
            "approach_gain": float(cluster[0]["dist_to_net"] - cluster[-1]["dist_to_net"]),
            "towards_steps": int(decreasing_steps),
            "decreasing_steps": int(decreasing_steps),
            "majority_towards": bool(decreasing_steps >= max(1, int(np.ceil(transitions * 0.5)))),
            "speed_peak": float(speed_peak),
        }

    def _visible_reversal_metrics(self, before_dist: float, after_dist: float, post_cluster: Sequence[Dict]) -> Dict[str, float]:
        max_dist = float(after_dist)
        away_steps = 0
        prev_dist = float(after_dist)
        for sample in post_cluster[1:]:
            curr_dist = float(sample["dist_to_net"])
            max_dist = max(max_dist, curr_dist)
            if curr_dist > (prev_dist + 0.5):
                away_steps += 1
            prev_dist = curr_dist
        return_distance_gain = float(max(max_dist - before_dist, after_dist - before_dist))
        jump_speed = float(return_distance_gain)
        if after_dist >= (before_dist + self.cfg.gap_return_min_distance_px):
            return {
                "reversed": True,
                "return_distance_gain": float(return_distance_gain),
                "jump_speed": jump_speed,
                "reason": "gap_same_side_reappearance_away_from_net",
            }
        if away_steps >= 1 and max_dist >= (before_dist + self._soft_return_min_distance()):
            return {
                "reversed": True,
                "return_distance_gain": float(return_distance_gain),
                "jump_speed": jump_speed,
                "reason": "post_gap_dist_to_net_trend_increasing",
            }
        return {
            "reversed": False,
            "return_distance_gain": float(return_distance_gain),
            "jump_speed": jump_speed,
            "reason": "no_post_gap_return_trend",
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
    def _first_side(samples: Sequence[TrajectorySample]) -> Optional[str]:
        for sample in samples:
            if sample.side in (SIDE_A, SIDE_B):
                return sample.side
        return None

    def _soft_attack_min_approach(self) -> float:
        return max(3.0, float(self.cfg.attack_min_approach_px) * 0.35)

    def _soft_attack_min_speed(self) -> float:
        return max(1.0, float(self.cfg.attack_min_speed_px) * 0.20)

    def _contact_proximity_threshold(self) -> float:
        return max(float(self.cfg.net_zone_tolerance_px), float(self.cfg.gap_near_net_tolerance_px))

    def _gap_near_net_threshold(self) -> float:
        return max(float(self.cfg.net_zone_tolerance_px), float(self.cfg.gap_near_net_tolerance_px))

    def _soft_return_min_distance(self) -> float:
        return max(4.0, float(self.cfg.return_min_distance_px) * 0.50)

    def _soft_post_contact_speed(self) -> float:
        return max(1.0, float(self.cfg.post_contact_min_speed_px) * 0.50)

    @staticmethod
    def _team_for_side(side: Optional[str]) -> Optional[str]:
        if side == SIDE_A:
            return "TeamA"
        if side == SIDE_B:
            return "TeamB"
        return None
