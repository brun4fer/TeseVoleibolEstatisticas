"""
main.py
---------
Loop principal do sistema de análise de voleibol.
"""

import cv2
import numpy as np

from analytics import AnalyticsEngine, draw_sidebar
from calibration import run_calibration
from config import config
from scoreboard_template_reader import ScoreboardReader
from tracker import VolleyballTracker


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


def draw_game_intelligence_debug(frame, context):
    if not context:
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
    y0 = 292
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


def draw_ocr_roi_debug(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


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


def main():
    config.ensure_dirs()
    H, net_line, score_roi = run_calibration(config, force=True)
    # Override ROI carregado/selecionado
    if score_roi is not None:
        config.score_roi = tuple(score_roi)

    reader = ScoreboardReader(templates_dir="digit_templates")
    tracker = VolleyballTracker(H, net_line)

    cap = cv2.VideoCapture(str(config.video_path()))
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir o video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f, end_f = config.time_window_frames(fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Nao foi possivel ler o primeiro frame do video.")

    config.score_roi = tuple(reader.set_roi(first_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    analytics = AnalyticsEngine(reader)

    frame_idx = start_f
    skip_rate = 1  # processar todos os frames para não perder o início do serviço
    trail_color = (0, 255, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx > end_f:
            break

        # frame skipping para velocidade
        if (frame_idx - start_f) % skip_rate == 1:
            frame_idx += 1
            continue

        scoreboard_frame = frame.copy()
        ts = frame_idx / fps
        detections = tracker.detect(frame, timestamp_s=ts, fps=fps, frame_idx=frame_idx)
        ball_state = tracker.update_ball(detections["ball_det"], timestamp_s=ts, frame_idx=frame_idx)

        rally_finished, ptype = analytics.process_frame(
            frame=frame,
            frame_idx=frame_idx,
            timestamp_s=ts,
            ball_state=ball_state,
            players=detections["players"],
            tracker=tracker,
            scoreboard_frame=scoreboard_frame,
        )

        if rally_finished:
            if ptype in ("POINT_BY_BLOCK", "BLOCK"):
                trail_color = (255, 0, 0)  # Blue (BGR)
            elif ptype in ("POINT_BY_SPIKE", "SPIKE"):
                trail_color = (0, 0, 255)  # Red (BGR)
            else:
                trail_color = (0, 255, 0)

        if not config.HEADLESS_MODE:
            draw_players(frame, detections["players"])
            ball_debug = tracker.ball_debug_snapshot()
            ball_trail = tracker.accepted_ball_points(last_n=config.ball_debug_trajectory_length)
            draw_ball_trail(
                frame,
                ball_trail,
                max_points=config.ball_debug_trajectory_length,
                color=trail_color,
                max_segment_px=config.ball_debug_max_segment_px,
            )
            if getattr(config, "BALL_DEBUG_VISUAL", False):
                draw_ball_debug_visual(frame, detections["ball_det"], ball_state, ball_debug)
            if getattr(config, "GAME_RULES_DEBUG_VISUAL", False):
                draw_game_intelligence_debug(frame, tracker.game_context_snapshot())
            draw_sidebar(frame, analytics.rally_mgr, analytics.counts, analytics.rally_counter)
            if getattr(config, "SCOREBOARD_DEBUG_VISUAL", False):
                draw_scoreboard_debug(frame, analytics.scoreboard_snapshot(), config.score_roi)

            # ROI do marcador em verde (debug OCR)
            draw_ocr_roi_debug(frame, config.score_roi)

            # info dispositivo
            cv2.putText(
                frame,
                f"Device: {tracker.device.upper()}",
                scoreboard_safe_text_pos(frame, config.score_roi, 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cooldown_left = max(0.0, (analytics.last_point_time + analytics.point_cooldown_s) - ts)
            if cooldown_left > 0.0:
                cv2.putText(
                    frame,
                    f"Cooldown: {cooldown_left:.1f}s",
                    scoreboard_safe_text_pos(frame, config.score_roi, 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )
            ocr_lock_left = max(0.0, analytics.ocr_blocked_until - ts)
            if ocr_lock_left > 0.0:
                cv2.putText(
                    frame,
                    f"OCR Lock: {ocr_lock_left:.1f}s",
                    scoreboard_safe_text_pos(frame, config.score_roi, 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Voleibol - Analytics", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        else:
            if frame_idx % max(int(fps * 5), 1) == 0:
                print(f"[HEADLESS] frame {frame_idx} ts={frame_idx/fps:.2f}s")

        frame_idx += 1

    cap.release()
    if not config.HEADLESS_MODE:
        cv2.destroyAllWindows()

    df = analytics.rally_mgr.dataframe()
    out_csv = config.output_dir / "tese_volleyball_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"CSV salvo em {out_csv.resolve()}")


if __name__ == "__main__":
    main()
