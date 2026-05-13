# Volleyball Match Analytics for Thesis

Computer vision project for volleyball analysis from match video, developed in the context of an academic thesis on sports analytics.

This repository is no longer just a ball-tracking experiment. In its current state, it combines:

- ball detection and temporal tracking
- court calibration and homography
- scoreboard OCR with template matching
- rally segmentation and event classification
- event persistence to JSON with preview frames
- a small desktop dashboard for reviewing detected events

## Current status

The project already has a working end-to-end prototype for offline volleyball analysis.

Implemented today:

- full video processing pipeline through [`main.py`](main.py)
- YOLO-based player and ball detection
- temporal ball selection and recovery logic in [`ball_tracking_core.py`](ball_tracking_core.py)
- Kalman-assisted ball smoothing and short-gap interpolation in [`tracker.py`](tracker.py)
- manual court calibration and court-plane projection in [`calibration.py`](calibration.py)
- scoreboard OCR based on digit templates in [`scoreboard_template_reader.py`](scoreboard_template_reader.py)
- scoreboard sanity checks and stable voting in [`analytics.py`](analytics.py)
- rally closing and point-type classification such as spike, block, ace, error, freeball, and ball-on-net
- persistent event storage in [`event_store.py`](event_store.py)
- event review tools in [`stats_ui.py`](stats_ui.py) and [`stats_debug.py`](stats_debug.py)

Still not finished:

- fully automatic official match statistics extraction
- automatic calibration and automatic scoreboard ROI discovery
- quantitative benchmark evaluation
- packaging and reproducible dependency management

## What the pipeline does

At a high level, the main workflow is:

1. Open a selected video segment defined in [`config.py`](config.py).
2. Calibrate the court with manual clicks and estimate the homography.
3. Select the scoreboard ROI.
4. Detect players and ball candidates with YOLO.
5. Track the ball through temporal scoring, gating, recovery logic, and Kalman support.
6. Project ball motion into calibrated court geometry.
7. Read the scoreboard with template-based OCR and stabilize the reading over time.
8. Use rally logic plus scoreboard changes to confirm point endings.
9. Classify the rally outcome and save structured events.
10. Export CSV/JSON outputs and optional preview images.

## Main components

### Core pipeline

- [`main.py`](main.py)
  Entry point for the full analysis pipeline.

- [`tracker.py`](tracker.py)
  Runtime tracker for players and ball, with geometry-aware logic near the net.

- [`ball_tracking_core.py`](ball_tracking_core.py)
  Shared ball decision engine: candidate parsing, temporal scoring, foreground support, speed checks, and trajectory maintenance.

- [`analytics.py`](analytics.py)
  Rally manager, scoreboard stabilization, OCR validation, event classification, and statistics export.

- [`volleyball_rules.py`](volleyball_rules.py)
  Incremental game-intelligence layer used to reason about ball continuity, likely possession, net crossing, and rally lifecycle hints.

- [`block_detection.py`](block_detection.py)
  Trajectory-driven block detector on top of calibrated court geometry.

### Calibration and OCR

- [`calibration.py`](calibration.py)
  Manual court calibration and storage of homography, net line, and scoreboard ROI metadata.

- [`scoreboard_template_reader.py`](scoreboard_template_reader.py)
  Template-based OCR for scoreboard digits using preprocessing, component matching, and fallback strip matching.

- [`digit_templates/`](digit_templates)
  Digit templates used by the scoreboard reader.

### Event review and outputs

- [`event_store.py`](event_store.py)
  Persistent JSON event store with preview frame export.

- [`stats_ui.py`](stats_ui.py)
  Tkinter dashboard for browsing saved events.

- [`stats_debug.py`](stats_debug.py)
  CLI summary for auditing event categories, reasons, and recent detections.

### Training and dataset utilities

- [`train_vball.py`](train_vball.py)
  Helper script for YOLO fine-tuning on volleyball ball data.

- [`extract_frames.py`](extract_frames.py)
- [`ingest_dataset.py`](ingest_dataset.py)
- [`split_dataset.py`](split_dataset.py)
- [`rename_dataset.py`](rename_dataset.py)
- [`check_dataset.py`](check_dataset.py)

These scripts support dataset preparation and maintenance, but the main thesis workflow is currently the analysis pipeline in [`main.py`](main.py).

## Repository layout

```text
.
|-- analytics.py
|-- ball_tracking_core.py
|-- block_detection.py
|-- calibration.py
|-- config.py
|-- court_geometry.py
|-- event_store.py
|-- main.py
|-- overlay_renderer.py
|-- scoreboard_template_reader.py
|-- stats_debug.py
|-- stats_ui.py
|-- tracker.py
|-- volleyball_rules.py
|-- calibration/
|-- dataset/
|-- digit_templates/
|-- outputs/
|-- runs/
```

## Requirements

There is currently no `requirements.txt`, so setup is manual.

Recommended environment:

- Python 3.10 or newer
- Windows is the environment currently assumed by the repository layout
- CUDA-enabled PyTorch if GPU inference is desired

Main Python packages:

- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `pyyaml`

Optional notes:

- [`stats_ui.py`](stats_ui.py) uses Tkinter. On standard Windows Python installs this is usually already available.
- Some helper scripts may depend on your local video and dataset layout.

Example setup:

```bash
git clone https://github.com/brun4fer/TeseVoleibolEstatisticas.git
cd TeseVoleibolEstatisticas
python -m venv .venv
.venv\Scripts\activate
pip install torch ultralytics opencv-python numpy pandas pyyaml
```

If you want to verify GPU visibility:

```bash
python check_gpu.py
```

## Configuration

Before running the full pipeline, review [`config.py`](config.py).

The most important fields are:

- `videos_dir`
- `video_file`
- `start_ts`
- `end_ts`
- `ball_yolo_model`
- `score_roi`
- `ocr_every_n_frames`
- debug and visualization flags such as `HEADLESS_MODE`, `SHOW_STATS_PANEL`, and `SHOW_BLOCK_DEBUG`

Important practical note:

- the current config uses Windows-style local paths
- the pipeline is meant for offline analysis of a chosen video segment, not batch processing of a dataset folder

## How to run

### Full analytics pipeline

```bash
python main.py
```

Current startup behavior:

1. The program opens the selected video segment.
2. It asks for manual court calibration clicks.
3. It asks for manual scoreboard ROI selection.
4. It then processes the configured time window frame by frame.

This is the real current behavior of the repository. Calibration and scoreboard selection are not yet fully automatic.

### Evaluation mode

```bash
python main.py --eval
```

`--eval` runs the processing stage without the live overlay, but the current version still requires the startup calibration and ROI selection steps before frame processing begins.

### Scoreboard OCR tool

To test the scoreboard reader in isolation:

```bash
python scoreboard_template_reader.py --video "C:\path\to\video.mp4"
```

Useful optional arguments:

- `--start`
- `--end`
- `--frame-step`
- `--templates-dir`
- `--set-ratio`
- `--block-size`
- `--threshold-c`

### Event dashboard

```bash
python stats_ui.py
```

This opens the Tkinter dashboard for browsing the saved events from `outputs/volleyball_events.json`.

### Event debug summary

```bash
python stats_debug.py
```

Or with an explicit path:

```bash
python stats_debug.py outputs/volleyball_events.json
```

### Ball-tracking experiment

```bash
python test_ball_detection.py
```

This script is still useful for focused debugging of ball selection and trajectory behavior, but it is no longer the only important part of the repository.

### Training helper

```bash
python train_vball.py --data data.yaml
```

This script is a helper, not a polished training CLI. Review the file before long training runs because some training parameters are still hardcoded inside the script.

## Generated files

The main pipeline currently writes:

- `calibration/field_params.json`
  Saved homography, net line, and scoreboard ROI metadata.

- `outputs/tese_volleyball_stats.csv`
  CSV export of finished rallies.

- `outputs/volleyball_events.json`
  Structured event log.

- `outputs/event_previews/`
  Preview images associated with recorded events.

- `outputs/stats_summary.json`
  Summary file generated in evaluation mode.

## Current strengths

- The repository already goes beyond raw detection and includes match-level logic.
- Ball tracking uses several complementary signals instead of trusting YOLO confidence alone.
- Scoreboard OCR is integrated into the rally logic rather than treated as a standalone demo.
- Events are persisted in a structured way that is useful for later inspection.
- The project already contains practical debugging tools for reviewing results.

## Current limitations

- Full startup is still manual because court calibration and scoreboard ROI selection are interactive.
- The scoreboard OCR is tuned to the scoreboard style used in the current videos and is not yet a generic OCR solution.
- Event classification is heuristic and research-oriented, not yet benchmarked against annotated ground truth.
- The repository depends on local videos that are not included here.
- Configuration is still code-first through [`config.py`](config.py), with several environment-specific assumptions.
- There is no packaged dependency file or installation script yet.
- Some utility scripts are experimental and still reflect thesis iteration rather than production hardening.

## Thesis context

This project is part of a thesis-oriented effort toward automatic volleyball statistics from video.

The research direction includes:

- robust volleyball ball tracking in real match footage
- rally understanding from temporal and geometric cues
- scoreboard-assisted validation
- event detection for spike, block, ace, and error scenarios
- persistent outputs that can support later statistical analysis

In other words, this repository is currently best described as a working prototype for volleyball video analytics, not yet as a finished product for automatic official statistics.

## Author

Bruno

Email: `bigbf1130@gmail.com`

GitHub: [brun4fer](https://github.com/brun4fer)
