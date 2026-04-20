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
    video_file: str = "VideoAcademica.mp4"
    output_dir: Path = Path("outputs")
    calibration_dir: Path = Path("calibration")
    event_preview_dir: Path = Path("outputs/event_previews")
    event_store_file: Path = Path("outputs/volleyball_events.json")

    # Time window (absolute timestamps in the video)
    start_ts: str = "00:37:02"
    end_ts: str = "00:46:17"

    # Models
    yolo_model: str = "yolov8s.pt"  # modelo small para melhor detecção de bola
    ball_yolo_model: str = "runs/detect/train5/weights/best.pt"
    score_reader_lang: Tuple[str, ...] = ("en",)

    # Detection thresholds
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    ball_conf_high: float = 0.35
    ball_conf_low: float = 0.15
    kalman_gate_px: float = 60.0

    # Tracking / analytics params
    max_trail: int = 60
    spike_speed_thresh: float = 14.0  # pixels / frame
    spike_speed_threshold_px: float = 8.0  # threshold for SPIKE vs FREEBALL
    block_speed_inversion_ratio: float = -0.6
    spike_dir_change_cos: float = -0.35  # inversao de direcao para spike
    ace_max_duration_s: float = 3.0
    net_band_height_px: int = 25  # tolerance when checking ball near net
    net_line_tolerance_px: int = 80  # distancia max ao segmento da rede para block
    net_buffer_px: int = 15  # neutral zone around net where side should not flip
    net_cross_confirm_frames: int = 3
    zone_block_half_width_px: int = 100  # margem horizontal da caixa de impacto da rede (+40px de antecipacao)
    zone_block_below_px: int = 20  # ZONE_BLOCK começa 20px abaixo da fita da rede
    zone_block_above_px: int = 100  # ZONE_BLOCK estende 100px acima da fita da rede
    block_height_margin_m: float = 0.50  # faixa acima do topo da rede para validar block
    block_player_proximity_px: int = 110  # distancia max jogador defensor <-> impacto na rede
    block_min_vx_px: float = 1.5  # velocidade horizontal minima para validar inversao
    block_occlusion_max_frames: int = 12  # frames maximos para fechar evento por oclusao
    block_event_ttl_s: float = 3.0  # tempo maximo para associar evento da rede ao ponto OCR
    block_event_cooldown_frames: int = 6  # evita duplicar bloco em frames consecutivos
    block_attack_min_speed_px: float = 6.0
    block_attack_min_approach_px: float = 10.0
    block_attack_window_points: int = 6
    block_net_contact_window_points: int = 8
    block_reversal_window_points: int = 6
    block_net_max_dwell_points: int = 4
    block_return_min_distance_px: float = 12.0
    block_post_contact_min_speed_px: float = 4.0
    block_live_window_points: int = 18
    ball_min_area_px: int = 10  # area minima aceitavel da bbox da bola
    ball_max_area_px: int = 2000  # area maxima aceitavel da bbox da bola
    ball_max_age_frames: int = 30  # manter ID da bola se desaparecer (frames)
    court_margin_m: float = 0.35  # tolerancia na conversao pixel->campo
    game_ball_out_of_bounds_margin_m: float = 0.75
    game_ball_out_of_bounds_penalty: float = 120.0

    # Scoreboard ROI (x, y, w, h) in pixels relative to full frame
    score_roi: Tuple[int, int, int, int] = (25, 45, 340, 150)
    ocr_every_n_frames: int = 15
    SCOREBOARD_DEBUG_VISUAL: bool = True
    SCOREBOARD_DEBUG_WINDOW: bool = False
    SCOREBOARD_DEBUG_CLEAN_ROI_WINDOW: bool = False
    SCOREBOARD_DEBUG_PREPROCESSED_WINDOW: bool = False
    SCOREBOARD_DEBUG_LOG: bool = False
    scoreboard_vote_window_reads: int = 5
    scoreboard_vote_min_hits: int = 3
    scoreboard_initial_confirm_reads: int = 2
    scoreboard_change_confirm_reads: int = 2
    scoreboard_valid_set_values: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    scoreboard_max_points_value: int = 60

    # Kalman noise params
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1

    # UI / execução
    HEADLESS_MODE: bool = False
    EVALUATION_MODE: bool = False
    BALL_DEBUG_VISUAL: bool = True
    BALL_DEBUG_LOG: bool = False
    SHOW_BALL_DEBUG: bool = False
    SHOW_SCOREBOARD_DEBUG: bool = False
    SHOW_RULES_DEBUG: bool = False
    SHOW_GEOMETRY_DEBUG: bool = False
    SHOW_STATS_PANEL: bool = True
    SHOW_SCOREBOARD_PANEL: bool = True
    SHOW_EVENT_PANEL: bool = True
    SHOW_PLAYER_DEBUG: bool = False
    SHOW_OCR_ROI_DEBUG: bool = False
    SHOW_TECHNICAL_INFO: bool = False
    SHOW_NET_ZONE_DEBUG: bool = False
    SHOW_BLOCK_DEBUG: bool = False
    EVENT_BANNER_HOLD_S: float = 4.0
    ball_core_conf_threshold: float = 0.15
    ball_core_resize_width: int = 1280
    ball_pixels_per_meter: float = 50.0
    ball_debug_trajectory_length: int = 40
    ball_debug_max_segment_px: float = 120.0

    # Volleyball game-intelligence layer
    GAME_RULES_ENABLED: bool = True
    GAME_RULES_VALIDATE_BALL: bool = True
    GAME_RULES_SUPPRESS_DUBIOUS_BALL_FOR_ANALYTICS: bool = True
    GAME_RULES_DEBUG_VISUAL: bool = True
    GAME_RULES_DEBUG_LOG: bool = False
    BLOCK_DEBUG_VISUAL: bool = True
    BLOCK_DEBUG_LOG: bool = False
    game_net_neutral_px: float = 15.0
    game_net_cross_tolerance_px: float = 80.0
    game_net_cross_confirm_frames: int = 3
    game_ball_max_step_px: float = 260.0
    game_ball_missing_step_allowance_px: float = 35.0
    game_ball_max_missing_grace_frames: int = 8
    game_ball_teleport_net_tolerance_px: float = 130.0
    game_ball_abrupt_turn_cos: float = -0.88
    game_ball_abrupt_turn_min_step_px: float = 45.0
    game_ball_abrupt_turn_net_grace_px: float = 140.0
    game_ball_stats_penalty_threshold: float = 45.0
    game_ball_reject_penalty_threshold: float = 100.0
    game_ball_large_step_near_net_penalty: float = 55.0
    game_ball_impossible_step_penalty: float = 120.0
    game_ball_side_teleport_penalty: float = 120.0
    game_ball_abrupt_turn_penalty: float = 45.0
    game_possession_confirm_frames: int = 3
    game_rally_lost_confirm_s: float = 1.0
    game_score_confirm_window_s: float = 2.0
    event_store_reset_on_start: bool = True
    event_preview_max_width: int = 420

    def video_path(self) -> Path:
        return self.videos_dir / self.video_file

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.event_preview_dir.mkdir(parents=True, exist_ok=True)

    def time_window_frames(self, fps: float) -> Tuple[int, int]:
        start_sec = _time_to_seconds(self.start_ts)
        end_sec = _time_to_seconds(self.end_ts)
        return int(start_sec * fps), int(end_sec * fps)


config = Config()
