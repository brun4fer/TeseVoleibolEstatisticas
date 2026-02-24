"""
analytics.py
-------------
Motor de eventos/estatísticas para ralis de voleibol.
- Deteta início/fim de rali.
- Classifica Ace / Spike / Block / Error usando heurísticas físicas.
- OCR do marcador para validar fim de ponto.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import re

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
        self.pending_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_count: int = 0
        self.old_formatted: Optional[str] = None

    def _extract_line(self, img) -> Tuple[int, int]:
        """Retorna (sets, pontos) para uma linha já extraída."""
        nums = re.findall(r"\d+", img)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return 0, int(nums[0])
        return 0, 0

    def read(self, frame) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = config.score_roi
        roi = frame[y : y + h, x : x + w]
        # divide horizontalmente (topo = Equipa A, base = Equipa B)
        mid_h = h // 2
        top = roi[0:mid_h, :]
        bot = roi[mid_h:h, :]

        def split_and_process(line_img):
            lh, lw = line_img.shape[:2]
            split_x = int(lw * 0.3)  # 30% para Sets, 70% para Pontos
            zone_set = line_img[:, :split_x]
            zone_pts = line_img[:, split_x:]

            # upscale
            set_up = cv2.resize(zone_set, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            pts_up = cv2.resize(zone_pts, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

            # binarização diferenciada
            set_gray = cv2.cvtColor(set_up, cv2.COLOR_BGR2GRAY)
            pts_gray = cv2.cvtColor(pts_up, cv2.COLOR_BGR2GRAY)

            set_bin = cv2.threshold(set_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            pts_bin = cv2.threshold(pts_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # engrossar
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            set_bin = cv2.dilate(set_bin, k, iterations=1)
            pts_bin = cv2.dilate(pts_bin, k, iterations=1)

            # juntar horizontalmente para OCR e debug
            combined = cv2.hconcat([set_bin, pts_bin])
            return set_bin, pts_bin, combined

        set_top, pts_top, comb_top = split_and_process(top)
        set_bot, pts_bot, comb_bot = split_and_process(bot)

        # OCR separado
        txt_set_top = "".join(self.reader.readtext(set_top, allowlist="0123456789", detail=0))
        txt_pts_top = "".join(self.reader.readtext(pts_top, allowlist="0123456789", detail=0))
        txt_set_bot = "".join(self.reader.readtext(set_bot, allowlist="0123456789", detail=0))
        txt_pts_bot = "".join(self.reader.readtext(pts_bot, allowlist="0123456789", detail=0))

        a_set = int(re.findall(r"\d+", txt_set_top)[0]) if re.findall(r"\d+", txt_set_top) else 0
        b_set = int(re.findall(r"\d+", txt_set_bot)[0]) if re.findall(r"\d+", txt_set_bot) else 0

        def extract_points(txt):
            nums = re.findall(r"\d+", txt)
            if not nums:
                return 0
            # juntar todos os dígitos lidos na zona de pontos
            return int("".join(nums))

        a_pts = extract_points(txt_pts_top)
        b_pts = extract_points(txt_pts_bot)

        parsed = (a_set, a_pts, b_set, b_pts)

        # Debug visual com 4 quadrantes
        # criar canvas
        h_dbg = max(comb_top.shape[0], comb_bot.shape[0])
        w_dbg = max(comb_top.shape[1], comb_bot.shape[1])
        canvas = 255 * np.ones((h_dbg * 2 + 3, w_dbg, 1), dtype=np.uint8)
        canvas[0:comb_top.shape[0], 0:comb_top.shape[1], 0] = comb_top
        canvas[h_dbg + 3 : h_dbg + 3 + comb_bot.shape[0], 0:comb_bot.shape[1], 0] = comb_bot
        # linhas vermelhas
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.line(canvas_bgr, (0, h_dbg + 1), (w_dbg, h_dbg + 1), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), 0), (int(w_dbg * 0.3), h_dbg), (0, 0, 255), 2)
        cv2.line(canvas_bgr, (int(w_dbg * 0.3), h_dbg + 3), (int(w_dbg * 0.3), h_dbg + 3 + h_dbg), (0, 0, 255), 2)
        cv2.imshow("DEBUG_OCR", canvas_bgr)
        cv2.waitKey(1)

        if self.pending_score == parsed:
            self.pending_count += 1
        else:
            self.pending_score = parsed
            self.pending_count = 1

        if self.pending_count >= 3 and self.stable_score != parsed:
            self.stable_score = parsed
            set_num = parsed[0] + parsed[2] + 1
            formatted = f"SET {set_num} ({parsed[1]}-{parsed[3]})"
            if formatted != self.old_formatted:
                print(formatted)
                self.old_formatted = formatted

        return self.stable_score


# -------------------------- RALLY STATE -----------------------------
@dataclass
class Rally:
    rally_id: int
    start_ts: float
    start_frame: int
    end_ts: Optional[float] = None
    end_frame: Optional[int] = None
    winner_team: Optional[str] = None
    point_type: Optional[str] = None
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
            self.active = Rally(rally_id=self.next_id, start_ts=ts, start_frame=frame_idx, ball_trail=list(trail))
            self.next_id += 1

    def end(self, ts: float, frame_idx: int, winner: str, ptype: str, max_speed: float, impact: Optional[Tuple[float, float]], reason: str):
        if self.active is None:
            return
        self.active.end_ts = ts
        self.active.end_frame = frame_idx
        self.active.winner_team = winner
        self.active.point_type = ptype
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
                    "type_of_point": r.point_type,
                    "winner_team": r.winner_team,
                    "ball_speed_max": r.max_speed_px,
                }
            )
        return pd.DataFrame(rows)


# ------------------------ Heurísticas -------------------------------
def is_service_lift(ball_history: List[Tuple[int, int]]) -> bool:
    """Bola sobe rapidamente da zona de serviço (baseline). Assume ecrã y cresce para baixo."""
    if len(ball_history) < 3:
        return False
    # zona de serviço: últimos 15% do comprimento da quadra (y próximo de 0 ou 18). Usamos pixel y.
    y_values = [p[1] for p in ball_history[-3:]]
    rising = y_values[-1] < y_values[-3] - 8  # subida
    return rising


def ball_inside_court(court_pt: Tuple[float, float]) -> bool:
    x, y = court_pt
    return 0 <= x <= 9 and 0 <= y <= 18


def classify_point(
    max_speed_px: float,
    acceleration: float,
    cos_inversion: float,
    near_net: bool,
    impact_in: bool,
    duration_s: float,
    contact_players: bool,
) -> str:
    if duration_s <= config.ace_max_duration_s and impact_in and not contact_players:
        return "ACE"
    if near_net and cos_inversion <= config.block_speed_inversion_ratio:
        return "POINT_BY_BLOCK"
    if max_speed_px >= config.spike_speed_thresh and acceleration > config.spike_speed_thresh and near_net:
        return "POINT_BY_SPIKE"
    if not impact_in:
        return "OPPONENT_ERROR"
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
    h, w, _ = frame.shape
    pad = 12
    panel_w = 220
    x0 = w - panel_w - pad
    y0 = pad
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 180), (0, 0, 0), -1)
    cv2.putText(frame, "Stats", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    lines = [
        f"Aces: {counts.get('ACE',0)}",
        f"Spikes: {counts.get('POINT_BY_SPIKE',0)}",
        f"Blocks: {counts.get('POINT_BY_BLOCK',0)}",
        f"Errors: {counts.get('OPPONENT_ERROR',0)}",
        f"Rallis: {rally_count}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0 + 10, y0 + 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------------- Motor Principal -----------------------------
class AnalyticsEngine:
    def __init__(self):
        self.rally_mgr = RallyManager()
        self.ocr = ScoreboardOCR()
        self.counts = {"ACE": 0, "POINT_BY_SPIKE": 0, "POINT_BY_BLOCK": 0, "OPPONENT_ERROR": 0}
        self.prev_score: Optional[Tuple[int, int, int, int]] = None
        self.pending_score_change: Optional[Tuple[int, int, int, int]] = None
        self.rally_counter: int = 0
        self.ball_history: Deque[Tuple[float, Tuple[int, int]]] = deque(maxlen=600)  # ~20 fps * 30s

    def process_frame(
        self,
        frame,
        frame_idx: int,
        timestamp_s: float,
        ball_state,
        players: List[Dict],
        tracker,  # VolleyballTracker
    ) -> Tuple[bool, Optional[str]]:
        """
        Atualiza estado do rali e retorna (rally_finalizado, ponto_label).
        Estatísticas só são registadas quando houver validação por OCR (mudança de score).
        """
        # guardar histórico de bola
        self.ball_history.append((timestamp_s, (int(ball_state.pixel[0]), int(ball_state.pixel[1]))))
        # manter apenas últimos 5s
        while self.ball_history and timestamp_s - self.ball_history[0][0] > 5.0:
            self.ball_history.popleft()
        # 1) leitura OCR a cada N frames
        if frame_idx % config.ocr_every_n_frames == 0:
            score = self.ocr.read(frame)
            if self.prev_score is None and score is not None:
                self.prev_score = score
            elif score is not None and self.prev_score is not None and score != self.prev_score:
                # prioridade: mudança de texto encerra rali
                self.pending_score_change = score

        rally_finished = False
        point_label = None

        # 2) contacto com jogadores para heurística
        min_dist = min_distance_ball_players(ball_state.pixel, players)
        contact_players = min_dist < 45

        # 3) Só encerra rali se houver mudança de score (OCR)
        if self.pending_score_change is not None:
            # garante que existe rally ativo; se não, cria um agora
            if self.rally_mgr.active is None:
                self.rally_mgr.start_if_needed(True, timestamp_s, frame_idx, tracker.trail_points())

            # olhar para histórico dos últimos 5s
            recent_points = [p for _, p in self.ball_history]
            impact_in = ball_inside_court(ball_state.court)
            duration = timestamp_s - self.rally_mgr.active.start_ts
            # calcular max speed nos últimos 5s
            max_speed_recent = 0.0
            if len(recent_points) >= 2:
                for i in range(len(recent_points) - 1):
                    dx = recent_points[i + 1][0] - recent_points[i][0]
                    dy = recent_points[i + 1][1] - recent_points[i][1]
                    max_speed_recent = max(max_speed_recent, float(np.hypot(dx, dy)))
            cos_inv_recent = tracker.horizontal_inversion()

            ptype = classify_point(
                max_speed_px=max_speed_recent,
                acceleration=tracker.acceleration(),
                cos_inversion=cos_inv_recent,
                near_net=tracker.ball_near_net(ball_state.pixel),
                impact_in=impact_in,
                duration_s=duration,
                contact_players=contact_players,
            )
            # decidir vencedor: compara pontos; se não mudar pontos, usa sets
            if self.prev_score:
                if self.pending_score_change[1] > self.prev_score[1]:
                    winner = "TeamA"
                elif self.pending_score_change[3] > self.prev_score[3]:
                    winner = "TeamB"
                elif self.pending_score_change[0] > self.prev_score[0]:
                    winner = "TeamA"
                elif self.pending_score_change[2] > self.prev_score[2]:
                    winner = "TeamB"
                else:
                    winner = "Unknown"
            else:
                winner = "Unknown"
            self.prev_score = self.pending_score_change
            self.pending_score_change = None

            # incrementa contador de ralis por mudança de score
            self.rally_counter += 1

            self.rally_mgr.end(
                ts=timestamp_s,
                frame_idx=frame_idx,
                winner=winner,
                ptype=ptype,
                max_speed=tracker._instant_speed(),
                impact=ball_state.court,
                reason="score_change",
            )
            if ptype in self.counts:
                self.counts[ptype] += 1
            rally_finished = True
            point_label = ptype

        return rally_finished, point_label
