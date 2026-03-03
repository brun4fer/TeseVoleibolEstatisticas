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


def main():
    config.ensure_dirs()
    H, net_line, score_roi = run_calibration(config, force=True)
    # Override ROI carregado/selecionado
    if score_roi is not None:
        config.score_roi = tuple(score_roi)

    tracker = VolleyballTracker(H, net_line)
    analytics = AnalyticsEngine()

    cap = cv2.VideoCapture(str(config.video_path()))
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir o video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f, end_f = config.time_window_frames(fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frame_idx = start_f
    skip_rate = 2  # processar 1 em cada 2 frames
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
            if ptype == "POINT_BY_BLOCK":
                trail_color = (255, 0, 0)  # Blue (BGR)
            elif ptype == "POINT_BY_SPIKE":
                trail_color = (0, 0, 255)  # Red (BGR)
            else:
                trail_color = (0, 255, 0)

        if not config.HEADLESS_MODE:
            draw_players(frame, detections["players"])
            draw_ball_trail(frame, tracker.drawer_points(last_n=50), max_points=50, color=trail_color)
            draw_sidebar(frame, analytics.rally_mgr, analytics.counts, analytics.rally_counter)

            # ROI do marcador em verde
            x, y, w, h = config.score_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # info dispositivo
            cv2.putText(frame, f"Device: {tracker.device.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
