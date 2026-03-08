import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import config

# Court dimensions in meters (FIVB)
COURT_W = 9.0
COURT_L = 18.0

CALIB_FILE = config.calibration_dir / "field_params.json"


def _save_calibration(homography: np.ndarray, net_line: Tuple[Tuple[int, int], Tuple[int, int]], score_roi: Tuple[int, int, int, int]) -> None:
    config.ensure_dirs()
    data = {
        "H": homography.tolist(),
        "net_line": {"p1": net_line[0], "p2": net_line[1]},
        "score_roi": score_roi,
        "scoreboard_roi": {
            "x": int(score_roi[0]),
            "y": int(score_roi[1]),
            "w": int(score_roi[2]),
            "h": int(score_roi[3]),
        },
    }
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_score_roi(data: dict) -> Optional[Tuple[int, int, int, int]]:
    scoreboard_roi = data.get("scoreboard_roi")
    if isinstance(scoreboard_roi, dict):
        try:
            return (
                int(scoreboard_roi["x"]),
                int(scoreboard_roi["y"]),
                int(scoreboard_roi["w"]),
                int(scoreboard_roi["h"]),
            )
        except (KeyError, TypeError, ValueError):
            pass

    raw_roi = data.get("score_roi")
    if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        try:
            return tuple(int(v) for v in raw_roi)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None

    return None


def _load_calibration() -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int, int, int]]:
    if CALIB_FILE.exists():
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        H = np.array(data["H"])
        net_line_data = data["net_line"]
        if isinstance(net_line_data, dict) and "p1" in net_line_data:
            p1 = tuple(net_line_data["p1"])
            p2 = tuple(net_line_data["p2"])
        else:
            # legacy integer net_y
            net_y = int(net_line_data) if isinstance(net_line_data, (int, float)) else 0
            p1 = (0, net_y)
            p2 = (1920, net_y)
        score_roi = _parse_score_roi(data) or tuple(config.score_roi)
        return H, (p1, p2), score_roi  # type: ignore

    # backward compat: older H.npy/net.json
    H_path = config.calibration_dir / "H.npy"
    net_path = config.calibration_dir / "net.json"
    if H_path.exists() and net_path.exists():
        H = np.load(H_path)
        with open(net_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return H, (tuple(d["p1"]), tuple(d["p2"])), config.score_roi  # type: ignore

    return None, None, None  # type: ignore


def _load_stored_score_roi() -> Tuple[int, int, int, int]:
    if CALIB_FILE.exists():
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        score_roi = _parse_score_roi(data)
        if score_roi is not None:
            return score_roi

    raise RuntimeError(
        "Scoreboard ROI nao encontrado em calibration/field_params.json. "
        "Defina 'score_roi' ou 'scoreboard_roi' no arquivo de calibracao."
    )


def collect_points(frame: np.ndarray) -> List[Tuple[int, int]]:
    """Clique 4 cantos (sentido horário) + 2 pontos no topo da rede."""
    cv2.startWindowThread()
    clone = frame.copy()
    points: List[Tuple[int, int]] = []
    window = "Calibracao - 4 cantos + 2 pontos no topo da rede"
    labels = [
        "Canto Superior Esquerdo",
        "Canto Superior Direito",
        "Canto Inferior Direito",
        "Canto Inferior Esquerdo",
        "Topo Rede Esquerda",
        "Topo Rede Direita",
    ]

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(clone, str(len(points)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(points) <= len(labels):
                print(f"Ponto {len(points)}/6: {labels[len(points)-1]} detetado")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, callback)

    while True:
        cv2.imshow(window, clone)
        for _ in range(10):
            cv2.waitKey(1)  # força render no Windows
        if len(points) == 6:
            break

    cv2.destroyWindow(window)
    return points


def compute_homography(court_px: List[Tuple[int, int]]) -> np.ndarray:
    real = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32)
    img = np.array(court_px[:4], dtype=np.float32)
    H, _ = cv2.findHomography(img, real, method=cv2.RANSAC)
    return H


def run_calibration(cfg=config, force: bool = False) -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int, int, int]]:
    """Carrega calibração existente ou pede cliques no primeiro frame, depois pede ROI do marcador."""
    if not force:
        H, net_line, score_roi = _load_calibration()
        if H is not None and net_line is not None and score_roi is not None:
            return H, net_line, score_roi

    cap = cv2.VideoCapture(str(config.video_path()))
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir o video para calibracao.")
    # tentar direto na posição 26:02
    target_ms = (26 * 60 + 2) * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ok, frame = cap.read()
    if not ok or frame is None:
        # limpar buffer: ler 30 frames sequenciais
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                break

    cap.release()
    if not ok:
        raise RuntimeError("Nao foi possivel ler frame para calibracao.")

    pts = collect_points(frame)
    if len(pts) != 6:
        raise RuntimeError("Calibracao cancelada/incompleta (6 cliques).")

    score_roi = _load_stored_score_roi()
    H = compute_homography(pts)
    net_line = (pts[4], pts[5])
    _save_calibration(H, net_line, score_roi)
    return H, net_line, score_roi


def pixel_to_court(H: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    """Projecta coordenada de pixel para plano do campo (metros)."""
    p = np.array([[pt[0], pt[1], 1.0]], dtype=np.float32).T
    proj = H @ p
    proj /= proj[2]
    return float(proj[0]), float(proj[1])
