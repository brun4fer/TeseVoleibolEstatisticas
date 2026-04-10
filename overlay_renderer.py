"""
overlay_renderer.py
-------------------
Runtime overlay rendering for normal UI and technical debug modes.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from config import config


def scoreboard_safe_text_pos(frame, roi, line_idx: int):
    x, y, w, h = roi
    right_x = int(x + w + 20)
    if right_x <= frame.shape[1] - 520:
        return right_x, int(y + 24 + (line_idx * 24))
    return int(x), int(y + h + 30 + (line_idx * 24))


def _score_points_text(score):
    if score is None:
        return "--"
    return f"{score[1]}-{score[3]}"


def draw_players(frame, players):
    for p in players:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(
            frame,
            f"ID {p['id']}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )


def draw_ball_trail(frame, trail, max_points: int = 40, color=(0, 255, 255), max_segment_px: float = 120.0):
    trail = [(int(p[0]), int(p[1])) for p in trail[-max_points:]]
    if not trail:
        return

    for i in range(1, len(trail)):
        p0 = trail[i - 1]
        p1 = trail[i]
        segment_distance = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if segment_distance > max_segment_px:
            continue
        cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)

    cv2.circle(frame, trail[-1], 4, (0, 0, 255), -1)


def draw_ball_focus(frame, ball_det):
    if ball_det is None or not bool(ball_det.get("visible", False)):
        return
    x1, y1, x2, y2 = [int(round(v)) for v in ball_det["bbox"]]
    cx, cy = [int(round(v)) for v in ball_det["center"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


def draw_ball_debug_visual(frame, ball_det, ball_state, debug_snapshot):
    quality = debug_snapshot.get("quality", {})
    reason = quality.get("reason") or "na"
    missed = int(quality.get("missed_frames", 0))
    max_missed = int(quality.get("max_missed_frames", 0))

    y0 = 268
    if ball_det is not None and bool(ball_det.get("visible", False)):
        x1, y1, x2, y2 = [int(round(v)) for v in ball_det["bbox"]]
        cx, cy = [int(round(v)) for v in ball_det["center"]]
        conf = float(ball_det.get("conf", ball_det.get("confidence", 0.0)))
        score = ball_det.get("final_score")
        speed_value = ball_det.get("speed_px_mean", ball_state.speed_px)
        speed = float(ball_state.speed_px if speed_value is None else speed_value)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ball {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        score_txt = f"{float(score):.1f}" if score is not None else "na"
        cv2.putText(
            frame,
            f"Ball OK | speed={speed:.1f}px/f | score={score_txt} | {reason}",
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return

    track_state = quality.get("track_state", "lost")
    cv2.putText(
        frame,
        f"Ball {str(track_state).upper()} | missed={missed}/{max_missed} | {reason}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_game_intelligence_debug(frame, context, *, show_rules: bool, show_geometry: bool):
    if not context or not (show_rules or show_geometry):
        return
    rally_id = int(context.get("current_rally_id", 0))
    rally_active = "ON" if context.get("rally_active") else "OFF"
    side = context.get("ball_side") or "--"
    possession = context.get("possession_side") or "--"
    team = context.get("possession_team") or "--"
    quality = context.get("ball_quality") or "--"
    accepted_stats = "ok" if context.get("ball_accepted_for_stats") else "hold"
    crossings = int(context.get("rally_net_crossings", 0))
    direction = context.get("last_net_crossing_direction") or "--"
    reasons = context.get("reasons") or []
    reason_txt = ";".join(str(r) for r in reasons) if reasons else "ok"
    if len(reason_txt) > 80:
        reason_txt = reason_txt[:77] + "..."

    events = context.get("events") or []
    last_event = events[-1].get("event_type", "--") if events and isinstance(events[-1], dict) else "--"
    debug = context.get("debug") or {}
    zone = debug.get("zone") or "--"
    inside_court = "in" if bool(debug.get("inside_court", True)) else "out"
    dist_to_net = debug.get("candidate_dist_net")
    dist_txt = "--" if dist_to_net is None else f"{float(dist_to_net):.1f}px"
    y0 = 292
    if show_rules:
        cv2.putText(
            frame,
            f"Rules | rally={rally_id}:{rally_active} | side={side} | poss={possession}/{team} | net={crossings} {direction}",
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Rules ball={quality}:{accepted_stats} | event={last_event} | {reason_txt}",
            (20, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
    if show_geometry:
        geometry_y = y0 + (44 if show_rules else 0)
        cv2.putText(
            frame,
            f"Geom zone={zone} | court={inside_court} | dist_net={dist_txt}",
            (20, geometry_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )


def draw_block_debug(frame, snapshot, roi):
    if not snapshot:
        return
    state = snapshot.get("state") or "--"
    event_type = snapshot.get("event_type") or "--"
    attack_side = snapshot.get("attack_side") or "--"
    attack_team = snapshot.get("attack_team") or "--"
    blocking_team = snapshot.get("blocking_team") or "--"
    blocking_side = snapshot.get("blocking_side") or "--"
    possession_side = snapshot.get("current_possession_side") or "--"
    possession_team = snapshot.get("current_possession_team") or "--"
    ball_side = snapshot.get("current_ball_side") or "--"
    end_side = snapshot.get("end_side") or "--"
    return_side = snapshot.get("return_side") or end_side
    last_attack = snapshot.get("last_attack") or {}
    last_attack_side = last_attack.get("side") or "--"
    last_attack_team = last_attack.get("team") or "--"
    last_attack_frame = last_attack.get("frame")
    last_attack_frame_txt = "--" if last_attack_frame is None else str(last_attack_frame)
    confidence = float(snapshot.get("confidence", snapshot.get("attack_confidence") or 0.0))
    reason = snapshot.get("reason") or "--"
    if len(reason) > 72:
        reason = reason[:69] + "..."

    line0 = scoreboard_safe_text_pos(frame, roi, 6)
    cv2.putText(
        frame,
        f"Block {state} | event={event_type} | poss={possession_side}/{possession_team} | conf={confidence:.2f}",
        line0,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    line1 = scoreboard_safe_text_pos(frame, roi, 7)
    cv2.putText(
        frame,
        f"Ball={ball_side} | Atk={attack_side}/{attack_team} | Block={blocking_side}/{blocking_team}",
        line1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    line2 = scoreboard_safe_text_pos(frame, roi, 8)
    cv2.putText(
        frame,
        f"LastAttack={last_attack_side}/{last_attack_team}@{last_attack_frame_txt} | Return={return_side} | {reason}",
        line2,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def draw_ocr_roi_debug(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_scoreboard_debug(frame, snapshot, roi):
    if not snapshot:
        return
    raw = snapshot.get("raw_score")
    confirmed = snapshot.get("confirmed_score")
    reference = snapshot.get("reference_score")
    pending = snapshot.get("pending_score")
    status = snapshot.get("status") or "--"
    rejected = snapshot.get("rejected_score")
    reject_reason = snapshot.get("reject_reason") or "--"
    confidence = float(snapshot.get("vote_confidence") or 0.0)
    match_score = float(snapshot.get("reader_match_score") or 0.0)
    pending_hits = int(snapshot.get("pending_hits") or 0)
    confirm_reads = int(snapshot.get("change_confirm_reads") or 0)
    changed = bool(snapshot.get("score_just_updated") or snapshot.get("score_changed"))

    color = (0, 255, 0) if changed else (255, 255, 255)
    line0 = scoreboard_safe_text_pos(frame, roi, 0)
    cv2.putText(
        frame,
        f"Placar Ref: {_score_points_text(reference)} | Conf: {_score_points_text(confirmed)} | Raw: {_score_points_text(raw)}",
        line0,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    line1 = scoreboard_safe_text_pos(frame, roi, 1)
    cv2.putText(
        frame,
        f"Scoreboard {status} | pending={_score_points_text(pending)} {pending_hits}/{confirm_reads} | vote={confidence:.2f} | match={match_score:.2f}",
        line1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if rejected is not None:
        line2 = scoreboard_safe_text_pos(frame, roi, 2)
        cv2.putText(
            frame,
            f"Scoreboard rejected {_score_points_text(rejected)} | {reject_reason}",
            line2,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )


def draw_live_score_panel(frame, snapshot):
    if not snapshot:
        return
    confirmed = snapshot.get("reference_score") or snapshot.get("confirmed_score")
    pending = snapshot.get("pending_score")
    changed = bool(snapshot.get("score_just_updated") or snapshot.get("score_changed"))
    label = _score_points_text(confirmed)
    panel_w = 220
    panel_h = 70
    x0 = max(12, frame.shape[1] - panel_w - 12)
    y0 = 12
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (230, 230, 230), 1)
    cv2.putText(
        frame,
        "Placar",
        (x0 + 14, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x0 + 14, y0 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 0) if changed else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if pending is not None and pending != confirmed:
        cv2.putText(
            frame,
            f"pending {_score_points_text(pending)}",
            (x0 + 110, y0 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )


def draw_simple_stats_panel(frame, counts: Dict[str, int]):
    panel_w = 240
    panel_h = 145
    x0 = max(12, frame.shape[1] - panel_w - 12)
    y0 = 96
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 1)
    cv2.putText(frame, "Estatisticas", (x0 + 14, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
    lines = [
        f"Spikes: {int(counts.get('spike', 0))}",
        f"Blocos: {int(counts.get('block', 0))}",
        f"Indefinidos: {int(counts.get('undefined', 0))}",
        f"Rallies: {int(counts.get('rallies', 0))}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x0 + 14, y0 + 55 + (index * 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_event_banner(frame, event_snapshot: Optional[Dict], timestamp_s: float, hold_s: float):
    if not event_snapshot:
        return
    end_ts = event_snapshot.get("end_time_seconds")
    if end_ts is not None and (float(timestamp_s) - float(end_ts)) > float(hold_s):
        return
    event_type = str(event_snapshot.get("type") or "undefined")
    point_team = event_snapshot.get("point_team") or "--"
    confidence = float(event_snapshot.get("confidence") or 0.0)
    reason = str(event_snapshot.get("reason") or "--")
    if len(reason) > 42:
        reason = reason[:39] + "..."
    label_map = {
        "spike": "Spike",
        "block": "Bloco",
        "undefined": "Indefinido",
    }
    label = label_map.get(event_type, event_type.title())
    banner_w = 420
    banner_h = 58
    x0 = max(12, (frame.shape[1] - banner_w) // 2)
    y0 = 14
    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + banner_w, y0 + banner_h), (0, 200, 255), 2)
    cv2.putText(
        frame,
        f"Evento: {label} | Equipa: {point_team} | conf={confidence:.2f}",
        (x0 + 12, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        reason,
        (x0 + 12, y0 + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )


def draw_technical_info(frame, analytics, tracker, timestamp_s: float, roi) -> None:
    cv2.putText(
        frame,
        f"Device: {tracker.device.upper()}",
        scoreboard_safe_text_pos(frame, roi, 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cooldown_left = max(0.0, (analytics.last_point_time + analytics.point_cooldown_s) - timestamp_s)
    if cooldown_left > 0.0:
        cv2.putText(
            frame,
            f"Cooldown: {cooldown_left:.1f}s",
            scoreboard_safe_text_pos(frame, roi, 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )
    ocr_lock_left = max(0.0, analytics.ocr_blocked_until - timestamp_s)
    if ocr_lock_left > 0.0:
        cv2.putText(
            frame,
            f"OCR Lock: {ocr_lock_left:.1f}s",
            scoreboard_safe_text_pos(frame, roi, 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )


def render_analysis_overlay(frame, detections, ball_state, tracker, analytics, trail_color, timestamp_s: float) -> None:
    ball_debug = tracker.ball_debug_snapshot()
    ball_trail = tracker.accepted_ball_points(last_n=config.ball_debug_trajectory_length)
    scoreboard_snapshot = analytics.scoreboard_snapshot()
    game_context = tracker.game_context_snapshot()
    block_snapshot = analytics.block_snapshot()
    event_snapshot = analytics.latest_event_snapshot()
    event_counts = analytics.event_counts_snapshot()

    if bool(getattr(config, "SHOW_PLAYER_DEBUG", False)):
        draw_players(frame, detections["players"])

    draw_ball_trail(
        frame,
        ball_trail,
        max_points=config.ball_debug_trajectory_length,
        color=trail_color,
        max_segment_px=config.ball_debug_max_segment_px,
    )

    if bool(getattr(config, "SHOW_BALL_DEBUG", False)):
        draw_ball_debug_visual(frame, detections["ball_det"], ball_state, ball_debug)
    else:
        draw_ball_focus(frame, detections["ball_det"])

    if bool(getattr(config, "SHOW_SCOREBOARD_PANEL", True)):
        draw_live_score_panel(frame, scoreboard_snapshot)

    if bool(getattr(config, "SHOW_STATS_PANEL", True)):
        draw_simple_stats_panel(frame, event_counts)

    if bool(getattr(config, "SHOW_EVENT_PANEL", True)):
        draw_event_banner(
            frame,
            event_snapshot,
            timestamp_s=timestamp_s,
            hold_s=float(getattr(config, "EVENT_BANNER_HOLD_S", 4.0)),
        )

    if bool(getattr(config, "SHOW_RULES_DEBUG", False)) or bool(getattr(config, "SHOW_GEOMETRY_DEBUG", False)):
        draw_game_intelligence_debug(
            frame,
            game_context,
            show_rules=bool(getattr(config, "SHOW_RULES_DEBUG", False)),
            show_geometry=bool(getattr(config, "SHOW_GEOMETRY_DEBUG", False)),
        )

    if bool(getattr(config, "SHOW_BLOCK_DEBUG", False)):
        draw_block_debug(frame, block_snapshot, config.score_roi)

    if bool(getattr(config, "SHOW_SCOREBOARD_DEBUG", False)):
        draw_scoreboard_debug(frame, scoreboard_snapshot, config.score_roi)

    if bool(getattr(config, "SHOW_NET_ZONE_DEBUG", False)):
        analytics._draw_net_zone(frame, tracker)

    if bool(getattr(config, "SHOW_OCR_ROI_DEBUG", False)):
        draw_ocr_roi_debug(frame, config.score_roi)

    if bool(getattr(config, "SHOW_TECHNICAL_INFO", False)):
        draw_technical_info(frame, analytics, tracker, timestamp_s, config.score_roi)
