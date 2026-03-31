# Volleyball Ball Tracking System

Computer vision project for automatic volleyball analysis from video, developed in the context of an academic thesis on sports analytics.

## 2. Description / Overview
This project focuses on detecting the volleyball, estimating its trajectory over time, and extracting motion-related information from match footage. The current implementation combines object detection with temporal consistency rules and auxiliary motion cues to make ball tracking more stable in difficult broadcast-style scenes.

The core problem is not simply "find a white circle in the frame". In real matches, the system must deal with:

- small ball size relative to the full frame
- motion blur during fast attacks and serves
- occlusions by players and the net
- visually confusing static structures such as walls, banners, and court markings
- low contrast when the ball overlaps the wooden floor

This repository was created to support automatic volleyball analysis, including ball tracking, trajectory visualization, point analysis, and the broader goal of automating match statistics from video.

At the moment, the project contains:

- a focused experimental script for ball detection and trajectory validation: [`test_ball_detection.py`](test_ball_detection.py)
- a broader end-to-end match analysis pipeline: [`main.py`](main.py), [`tracker.py`](tracker.py), [`analytics.py`](analytics.py)
- training, calibration, OCR, and dataset utility scripts

## 3. Demonstration
This repository currently does not ship with embedded GIFs or screenshots, but the intended demonstration section should show:

- the original frame with the detected ball bounding box
- the estimated trajectory drawn over time
- the computed ball speed
- difficult cases such as serves, spikes, wall-like false positives, and ball-over-floor scenarios

Recommended demo assets for a future portfolio version:

- short GIF of `test_ball_detection.py` with trajectory overlay
- comparison clip showing raw YOLO detections vs. temporally filtered final detections
- screenshot of the full analytics pipeline with scoreboard and rally classification

## 4. Features
- Ball detection from volleyball video using YOLO.
- Candidate validation using bounding box size and plausible image regions.
- Temporal continuity filtering using recent ball positions.
- Simple motion prediction using the two latest accepted centers.
- Auxiliary foreground analysis via background subtraction.
- Trajectory drawing with protection against unrealistic long segments.
- Approximate ball speed estimation in km/h.
- Real-time visualization of detections, speed, and trajectory.
- Broader rally analytics pipeline with scoreboard reading and point classification.

## 5. System Architecture
The current ball detection experiment follows this pipeline:

1. Read frames from a selected time window of the input video.
2. Resize the frame to a working resolution suitable for YOLO inference.
3. Run YOLO on the current frame to detect candidate ball bounding boxes.
4. Parse all valid ball candidates rather than immediately trusting the highest-confidence box.
5. Apply geometric plausibility filters:
   - valid class
   - valid bounding box size
   - valid image region
6. Build a foreground mask with background subtraction.
7. Measure local motion evidence around each candidate:
   - `fg_active_pixels`
   - `fg_active_ratio`
8. Estimate trajectory continuity using:
   - `previous_center`
   - `older_center`
   - predicted next center
9. Reject physically implausible candidates using temporal hard gates:
   - maximum step distance
   - maximum prediction error
10. Score the remaining candidates using a combined cost based on:
   - temporal coherence
   - YOLO confidence
   - foreground support
11. Accept the best candidate and update:
   - speed history
   - trajectory deque
   - temporal state
12. Render the final visualization and debug output.

In the broader repository, the full analysis system extends this with:

1. court calibration and homography
2. player tracking
3. OCR-based scoreboard reading
4. rally segmentation
5. event classification such as spike, block, ace, and error

## 6. Tracking Logic
This is the most important part of the current ball selection strategy.

### `previous_center`
`previous_center` stores the last accepted ball center. It is the main temporal anchor for frame-to-frame continuity.

It is used to:

- measure displacement between consecutive accepted positions
- estimate instantaneous speed
- reject candidate jumps that are too large
- decide whether the current detection is consistent with the recent track

### `older_center`
`older_center` stores the penultimate accepted center. Together with `previous_center`, it provides a short motion history.

This lets the system estimate a simple velocity vector:

- `vx = previous_center.x - older_center.x`
- `vy = previous_center.y - older_center.y`

### Motion prediction
The project uses a lightweight prediction rule:

```text
predicted_center = previous_center + (previous_center - older_center)
```

This assumes short-term constant velocity. It is intentionally simple:

- easy to inspect
- easy to debug
- good enough to reject obviously incoherent detections
- lightweight for real-time experimentation

### Distance validation
Candidate detections are checked against temporal plausibility before final scoring:

- `MAX_STEP_PIXELS`
  Limits the distance between the last accepted center and the current candidate.

- `MAX_PREDICTION_ERROR_PIXELS`
  Limits how far a candidate can be from the predicted next position.

If a candidate violates these hard gates, it is rejected before scoring.

### Combined scoring
Once candidates pass the hard gates, the system computes a final score that combines:

- distance to the last accepted center
- prediction error relative to the estimated next center
- YOLO confidence
- foreground support measured in a local motion patch

This is important because no single signal is enough:

- YOLO confidence alone may prefer a static false positive
- foreground alone is noisy on the wooden court
- temporal continuity alone can fail after occlusion

The selected candidate is the one with the best combined score.

### Missed frame handling
The system does not immediately destroy temporal state after one failed frame.

- `missed_frames` counts consecutive frames with no valid accepted candidate
- while the count stays below `MAX_MISSED_FRAMES`, the temporal state is preserved
- if the count exceeds the threshold, the system resets:
  - `previous_center`
  - `older_center`
  - `speed_history_kmh`
  - `trajectory`

This allows short interruptions without fully losing the track.

### Outlier rejection
Even after temporal gating, accepted detections still go through speed sanity checks.

- `MAX_SPEED_KMH` rejects implausible jumps
- low-motion filters can suppress detections that remain almost stationary for too long

### Trajectory continuity
The visual trajectory is stored in a `deque` and updated only for accepted detections.

To avoid drawing misleading lines:

- the trajectory is cleared when the track is truly lost
- the trajectory is cleared after persistent invalid states
- `draw_trajectory()` skips unrealistic long segments

This prevents the overlay from connecting unrelated points across time.

## 7. Implemented Filters
The system uses multiple filters that complement each other.

### YOLO Detection
YOLO is the main detector and the first source of ball candidates.

Signals used:

- class label
- confidence score
- bounding box coordinates

Role:

- propose candidate ball locations
- provide the confidence prior used in final ranking

Limitations:

- can still fire on static circular or bright regions
- may confuse the ball with background details under challenging lighting

### Bounding Box and Position Validation
Before a candidate enters the tracking logic, it must satisfy:

- minimum width
- minimum height
- minimum area
- plausible x/y position inside the frame

This removes obviously invalid boxes and many detections near borders or implausible areas.

### Motion Gating
Motion gating is the main temporal filter.

It uses:

- `previous_center`
- `older_center`
- `predict_next_center()`

Hard checks:

- maximum step distance
- maximum prediction error

Soft ranking:

- lower distance to the recent trajectory is better
- lower prediction error is better
- higher confidence helps the candidate

This prevents the system from switching to a false positive that is spatially inconsistent with the ongoing ball path.

### Background Subtraction
Background subtraction is an auxiliary signal, not the primary detector.

The current implementation uses OpenCV MOG2 to build a foreground mask and then measures motion around each candidate center.

Per candidate, the system stores:

- `fg_active_pixels`
- `fg_active_ratio`
- `low_foreground`

Foreground is now used as a scoring cue rather than a hard binary rejection in most cases.

Why this matters:

- on white or high-contrast backgrounds, the ball often yields cleaner foreground support
- on the wooden court, foreground is noisier because of players, shadows, and scene dynamics
- treating foreground as a soft signal makes the system more tolerant in realistic gameplay

Limitations:

- background subtraction is sensitive to camera noise and overall scene motion
- players moving close to the ball region can inflate local foreground support
- the wooden court remains a difficult setting

### Trajectory Filtering
Trajectory filtering focuses on visual consistency.

Implemented safeguards:

- trajectory reset after prolonged track loss
- trajectory reset after invalid state accumulation
- no insertion of rejected detections into the trajectory
- segment-level skip for unrealistically long visual connections

This improves both interpretability and debugging.

## 8. Code Structure
The repository contains both the focused ball detection experiment and the broader match analytics pipeline.

### Main files
- [`test_ball_detection.py`](test_ball_detection.py)
  Standalone experimental script for ball detection, temporal candidate selection, speed estimation, background subtraction scoring, and trajectory visualization.

- [`main.py`](main.py)
  Entry point for the broader volleyball analytics workflow.

- [`tracker.py`](tracker.py)
  Full tracker used by the main pipeline. Includes player detection, ball tracking support, court logic, and geometric net-event logic.

- [`analytics.py`](analytics.py)
  Rally and point analysis engine. Uses motion heuristics, OCR consistency, and event rules to classify outcomes.

- [`calibration.py`](calibration.py)
  Court calibration, homography estimation, and pixel-to-court projection utilities.

- [`config.py`](config.py)
  Central configuration for paths, thresholds, and execution settings.

- [`scoreboard_template_reader.py`](scoreboard_template_reader.py)
  Template-based scoreboard reading used by the analytics pipeline.

- [`train_vball.py`](train_vball.py)
  Utility to fine-tune a YOLO model specifically for volleyball ball detection.

### Main functions in `test_ball_detection.py`
- `get_ball_candidates()`
  Parses YOLO outputs into candidate detections.

- `predict_next_center()`
  Predicts the next plausible ball position using the last two accepted centers.

- `score_ball_candidate()`
  Combines temporal distance, prediction error, YOLO confidence, and foreground support into a single selection score.

- `select_ball_candidate()`
  Applies temporal hard gates and chooses the best candidate.

- `calculate_speed()`
  Converts frame-to-frame displacement into estimated speed.

- `draw_trajectory()`
  Renders the accepted ball path while skipping unrealistic long segments.

- `build_foreground_mask()`
  Creates a binary foreground mask using background subtraction.

- `foreground_score_at_center()`
  Measures local motion evidence around a candidate center.

### Flow of `main()`
At a high level, `main()` in `test_ball_detection.py`:

1. loads the model and video
2. initializes temporal state
3. builds the foreground mask for each frame
4. gets YOLO candidates
5. selects the best candidate with temporal logic
6. computes speed if the detection is accepted
7. updates trajectory and debug state
8. renders overlays and logs

## 9. Installation
This repository currently does not include a `requirements.txt`, so the simplest setup is to install the main dependencies manually.

```bash
git clone https://github.com/brun4fer/TeseVoleibolEstatisticas.git
cd TeseVoleibolEstatisticas
python -m venv .venv
.venv\Scripts\activate
pip install ultralytics opencv-python numpy torch pandas pyyaml static-ffmpeg
```

Optional dependencies:

- `detectron2`
  Only needed for the experimental Detectron pipeline under [`experiments/detectron_pipeline`](experiments/detectron_pipeline).

Recommended environment notes:

- Python 3.10 or newer
- CUDA-enabled PyTorch if GPU inference is desired
- Windows works well with the current repository layout

## 10. How to Run
### Ball detection experiment
Run the standalone ball detection and trajectory visualization script:

```bash
python test_ball_detection.py
```

Before running it, review these constants in [`test_ball_detection.py`](test_ball_detection.py):

- `VIDEO_PATH`
- `START_TIME`
- `END_TIME`
- `MODEL_PATH`

These define:

- which video is processed
- which segment of the video is analyzed
- which YOLO weights are used

### Full analytics pipeline
To run the broader volleyball analysis system:

```bash
python main.py
```

Before running it, review [`config.py`](config.py):

- input video path
- start and end timestamps
- calibration paths
- execution mode and thresholds

### Training a ball-specific model
To fine-tune YOLO for volleyball ball detection:

```bash
python train_vball.py --data data.yaml --batch 16
```

## 11. Important Parameters
The most influential parameters are split between the experimental ball script and the main configuration.

### Ball detection experiment
| Parameter | Meaning | Practical impact |
| --- | --- | --- |
| `CONF_THRESHOLD` | Minimum YOLO confidence for raw detections | Higher values reduce noise but can miss small balls |
| `MAX_SPEED_KMH` | Maximum plausible speed after acceptance | Rejects extreme jumps and spurious switches |
| `MAX_STEP_PIXELS` | Maximum step from last accepted center | Strong temporal continuity gate |
| `MAX_PREDICTION_ERROR_PIXELS` | Maximum error relative to predicted center | Controls tolerance to trajectory deviation |
| `MAX_MISSED_FRAMES` | Number of misses before full state reset | Higher values preserve continuity longer |
| `MIN_MOVEMENT_PIXELS` | Threshold for very small motion | Helps detect stationary false positives |
| `MAX_STATIONARY_FRAMES` | Consecutive near-static frames before suppression | Higher values are more tolerant to slow play |
| `DISTANCE_WEIGHT` | Weight of step distance in final score | Larger values prioritize continuity more strongly |
| `PREDICTION_WEIGHT` | Weight of prediction error in final score | Larger values prioritize short-term motion prediction |
| `CONFIDENCE_WEIGHT` | Contribution of YOLO confidence | Larger values trust the detector more |
| `FOREGROUND_WEIGHT` | Contribution of foreground support | Larger values favor moving candidates |
| `LOW_FOREGROUND_PENALTY` | Penalty applied to weak foreground candidates | Useful to discourage static false positives without hard rejection |
| `FOREGROUND_PATCH_RADIUS` | ROI radius around candidate center | Controls local motion measurement size |
| `SHOW_FOREGROUND_DEBUG` | Draw foreground diagnostics near the bbox | Useful for tuning |
| `SHOW_FOREGROUND_MASK` | Show the MOG2 foreground mask window | Useful for diagnosing motion quality |

### Full pipeline configuration
| Parameter | Meaning | File |
| --- | --- | --- |
| `ball_conf_high` / `ball_conf_low` | Ball acceptance thresholds | [`config.py`](config.py) |
| `kalman_gate_px` | Kalman gating threshold in the full tracker | [`config.py`](config.py) |
| `ball_min_area_px` / `ball_max_area_px` | Minimum and maximum ball box area | [`config.py`](config.py) |
| `court_margin_m` | Margin for court geometry acceptance | [`config.py`](config.py) |
| `net_band_height_px` | Net tolerance used in event logic | [`config.py`](config.py) |
| `spike_speed_threshold_px` | Spike vs. freeball heuristic threshold | [`config.py`](config.py) |

## 12. Current Limitations
This project is functional, but it still faces several realistic limitations.

- The ball remains difficult to distinguish when it overlaps the wooden court.
- Background subtraction is informative but noisy in scenes with many moving players.
- Performance depends heavily on the quality of the YOLO weights and the diversity of the training data.
- Strong occlusions by players or the net can still break continuity.
- Speed estimation is approximate because it depends on pixel displacement and a scale factor, not a full 3D physical reconstruction.
- Broadcast camera changes, zoom, or motion can reduce foreground stability.
- The temporal model is intentionally lightweight, so it is less robust than a dedicated probabilistic tracker.

## 13. Future Improvements
- Expand and rebalance the volleyball ball dataset with more difficult floor-background cases.
- Train a more robust dedicated small-object detector.
- Add automatic calibration-aware filtering based on the actual court polygon.
- Replace the simple temporal predictor with a Kalman filter inside the standalone script.
- Integrate a more robust multi-object or re-identification tracker when needed.
- Improve ball segmentation or appearance modeling for hard ball-over-floor cases.
- Add benchmark evaluation metrics such as precision, recall, ID consistency, and tracking stability.
- Export richer analytics and trajectory summaries for coaches and performance staff.

## 14. Technologies Used
- Python
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- pandas
- YAML-based dataset configuration

## 15. Academic Context
This repository is part of an academic effort focused on automatic performance analysis in volleyball.

The broader objective is to support thesis-level research on:

- ball tracking in real match footage
- automatic rally understanding
- event classification
- scoreboard-based validation
- video-driven statistical analysis for sports performance

In that sense, this project is both:

- a technical computer vision prototype
- a research tool for volleyball analytics

## 16. Author
Bruno  
Email: bigbf1130@gmail.com  
GitHub: [brun4fer](https://github.com/brun4fer)
