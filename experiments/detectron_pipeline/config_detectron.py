"""Configuration for the isolated Detectron2 experiment pipeline."""

from pathlib import Path

VIDEO_PATH = r"C:\Users\Utilizador\Desktop\Mestrado\Tese\VideosJogos\video-2026-02-24T15-06-27.171Z.mp4"
START_TIME = "00:29:02"
END_TIME = "00:46:17"
CONFIDENCE_THRESHOLD = 0.35

# Detectron2 model configuration (COCO pretrained).
MODEL_CONFIG_PATH = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
BALL_CLASS_ID = 32  # COCO class id for "sports ball"

# Candidate filtering for "small round object" behavior.
MIN_BALL_AREA = 12
MAX_BALL_AREA_RATIO = 0.015
MAX_ASPECT_RATIO_DEVIATION = 0.65

# Simple centroid tracker behavior.
MAX_TRACKING_DISTANCE_PX = 140
MAX_MISSING_FRAMES = 18

# Output controls.
DEBUG_FRAME_INTERVAL = 120
DEBUG_MAX_IMAGES = 150
PROGRESS_EVERY_FRAMES = 30

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug_frames"
