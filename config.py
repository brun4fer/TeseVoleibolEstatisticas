from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _time_to_seconds(ts: str) -> float:
    """Convert timestamp 'HH:MM:SS' to total seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class Config:
    # Paths
    videos_dir: Path = Path(r"C:\Users\Utilizador\Desktop\Mestrado\Tese\VideosJogos")
    video_file: str = "video-2026-02-24T15-06-27.171Z.mp4"
    output_dir: Path = Path("outputs")
    calibration_dir: Path = Path("calibration")

    # Time window (absolute timestamps in the video)
    start_ts: str = "00:29:02"
    end_ts: str = "00:46:17"

    # Models
    yolo_model: str = "yolov8s.pt"  # modelo small para melhor detecção de bola
    score_reader_lang: Tuple[str, ...] = ("en",)

    # Detection thresholds
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45

    # Tracking / analytics params
    max_trail: int = 60
    spike_speed_thresh: float = 14.0  # pixels / frame
    block_speed_inversion_ratio: float = -0.6
    spike_dir_change_cos: float = -0.35  # inversao de direcao para spike
    ace_max_duration_s: float = 3.0
    net_band_height_px: int = 25  # tolerance when checking ball near net
    net_line_tolerance_px: int = 80  # distancia max ao segmento da rede para block
    zone_block_half_width_px: int = 100  # margem horizontal da caixa de impacto da rede (+40px de antecipacao)
    zone_block_below_px: int = 20  # ZONE_BLOCK começa 20px abaixo da fita da rede
    zone_block_above_px: int = 100  # ZONE_BLOCK estende 100px acima da fita da rede
    block_height_margin_m: float = 0.50  # faixa acima do topo da rede para validar block
    block_player_proximity_px: int = 110  # distancia max jogador defensor <-> impacto na rede
    block_min_vx_px: float = 1.5  # velocidade horizontal minima para validar inversao
    block_occlusion_max_frames: int = 12  # frames maximos para fechar evento por oclusao
    block_event_ttl_s: float = 3.0  # tempo maximo para associar evento da rede ao ponto OCR
    block_event_cooldown_frames: int = 6  # evita duplicar bloco em frames consecutivos
    ball_max_area_px: int = 2200  # area maxima aceitavel da bbox da bola
    ball_max_age_frames: int = 30  # manter ID da bola se desaparecer (frames)
    court_margin_m: float = 0.35  # tolerancia na conversao pixel->campo

    # Scoreboard ROI (x, y, w, h) in pixels relative to full frame
    score_roi: Tuple[int, int, int, int] = (25, 45, 340, 150)
    ocr_every_n_frames: int = 15

    # Kalman noise params
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1

    # UI / execução
    HEADLESS_MODE: bool = False

    def video_path(self) -> Path:
        return self.videos_dir / self.video_file

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def time_window_frames(self, fps: float) -> Tuple[int, int]:
        start_sec = _time_to_seconds(self.start_ts)
        end_sec = _time_to_seconds(self.end_ts)
        return int(start_sec * fps), int(end_sec * fps)


config = Config()
