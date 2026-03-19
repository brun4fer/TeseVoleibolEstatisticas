"""
main.py
---------
Loop principal do sistema de análise de voleibol.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from analytics import AnalyticsEngine, draw_sidebar
from calibration import run_calibration
from config import config
from scoreboard_template_reader import ScoreboardReader
from tracker import VolleyballTracker


VALID_SET_VALUES = {0, 1, 2, 3}


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


def draw_ball_trail(frame, trail, max_points: int = 10, color=(0, 255, 0)):
    trail = trail[-max_points:]
    if len(trail) < 2:
        if trail:
            cv2.circle(frame, trail[-1], 6, (0, 0, 255), -1)
        return

    # Smooth polyline with a short moving average window.
    smoothed = []
    win = 5
    for i in range(len(trail)):
        a = max(0, i - win + 1)
        chunk = trail[a : i + 1]
        sx = int(sum(p[0] for p in chunk) / len(chunk))
        sy = int(sum(p[1] for p in chunk) / len(chunk))
        smoothed.append((sx, sy))

    pts = np.array(smoothed, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
    cv2.circle(frame, tuple(smoothed[-1]), 6, (0, 0, 255), -1)
    if len(smoothed) >= 2:
        p0 = smoothed[-2]
        p1 = smoothed[-1]
        cv2.arrowedLine(frame, p0, p1, color, 2, tipLength=0.35)


def draw_ocr_roi_debug(frame, roi):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def validate_score(
    prev_score: Optional[Tuple[int, int, int, int]],
    new_score: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    sets_a, points_a, sets_b, points_b = new_score

    prev_sets_a = prev_score[0] if prev_score is not None else 0
    prev_sets_b = prev_score[2] if prev_score is not None else 0

    if sets_a not in VALID_SET_VALUES:
        sets_a = prev_sets_a
    if sets_b not in VALID_SET_VALUES:
        sets_b = prev_sets_b

    return (sets_a, points_a, sets_b, points_b)


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
    raw_read = reader.read
    prev_score = validate_score(None, raw_read(first_frame))

    def validated_read(frame):
        nonlocal prev_score
        raw_score = raw_read(frame)
        score = validate_score(prev_score, raw_score)
        prev_score = score
        return score

    reader.read = validated_read
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

        detections = tracker.detect(frame)
        ts = frame_idx / fps
        ball_state = tracker.update_ball(detections["ball_det"], timestamp_s=ts)

        rally_finished, ptype = analytics.process_frame(
            frame=frame,
            frame_idx=frame_idx,
            timestamp_s=ts,
            ball_state=ball_state,
            players=detections["players"],
            tracker=tracker,
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
            draw_ball_trail(frame, tracker.drawer_points(last_n=50), max_points=50, color=trail_color)
            draw_sidebar(frame, analytics.rally_mgr, analytics.counts, analytics.rally_counter)
            if analytics.prev_score is not None:
                _a_set, a_pts, _b_set, b_pts = analytics.prev_score
                placar_color = (0, 255, 0) if getattr(analytics, "score_just_updated", False) else (255, 255, 255)
                cv2.putText(
                    frame,
                    f"Placar Ref: {a_pts}-{b_pts}",
                    (20, 164),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    placar_color,
                    2,
                )
                if getattr(analytics, "score_just_updated", False):
                    cv2.putText(
                        frame,
                        "Placar Ref atualizado",
                        (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            ocr_stable = getattr(analytics, "last_ocr_score_stable", None)
            ocr_raw = getattr(analytics, "last_ocr_score_raw", None)
            if ocr_stable is not None:
                cv2.putText(
                    frame,
                    f"OCR Stable: {ocr_stable[1]}-{ocr_stable[3]}",
                    (20, 216),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
            if ocr_raw is not None:
                cv2.putText(
                    frame,
                    f"OCR Raw: {ocr_raw[1]}-{ocr_raw[3]}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            # ROI do marcador em verde (debug OCR)
            draw_ocr_roi_debug(frame, config.score_roi)

            # info dispositivo
            cv2.putText(frame, f"Device: {tracker.device.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cooldown_left = max(0.0, (analytics.last_point_time + analytics.point_cooldown_s) - ts)
            if cooldown_left > 0.0:
                cv2.putText(frame, f"Cooldown: {cooldown_left:.1f}s", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            ocr_lock_left = max(0.0, analytics.ocr_blocked_until - ts)
            if ocr_lock_left > 0.0:
                cv2.putText(frame, f"OCR Lock: {ocr_lock_left:.1f}s", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

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
