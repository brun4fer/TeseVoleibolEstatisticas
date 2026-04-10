"""
main.py
-------
Loop principal do sistema de analise de voleibol.
"""

import cv2

from analytics import AnalyticsEngine
from calibration import run_calibration
from config import config
from overlay_renderer import render_analysis_overlay
from scoreboard_template_reader import ScoreboardReader
from tracker import VolleyballTracker


def main():
    config.ensure_dirs()
    H, net_line, score_roi = run_calibration(config, force=True)
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
    skip_rate = 1
    trail_color = (0, 255, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx > end_f:
            break

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
                trail_color = (255, 0, 0)
            elif ptype in ("POINT_BY_SPIKE", "SPIKE"):
                trail_color = (0, 0, 255)
            else:
                trail_color = (0, 255, 0)

        if not config.HEADLESS_MODE:
            render_analysis_overlay(
                frame=frame,
                detections=detections,
                ball_state=ball_state,
                tracker=tracker,
                analytics=analytics,
                trail_color=trail_color,
                timestamp_s=ts,
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
    print(f"Eventos salvos em {config.event_store_file.resolve()}")


if __name__ == "__main__":
    main()
