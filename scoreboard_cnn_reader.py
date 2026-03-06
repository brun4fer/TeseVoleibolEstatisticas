"""
CNN-based volleyball scoreboard reader.

Modes:
1) collect: collect labeled digit samples into dataset/0..9
2) train: train a lightweight CNN and save digit_cnn.pth
3) infer: run real-time scoreboard digit inference from video
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


DIGIT_NAMES = ("sets_top", "points_top", "sets_bottom", "points_bottom")
MODEL_PATH = Path("digit_cnn.pth")


def ts_to_seconds(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def select_scoreboard_roi_from_frame(frame: np.ndarray) -> Tuple[int, int, int, int]:
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    cv2.namedWindow("Select Scoreboard ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select Scoreboard ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("frame")
    cv2.destroyWindow("Select Scoreboard ROI")
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled.")
    return x, y, w, h


def split_roi_cells(roi: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = roi.shape[:2]
    h2 = h // 2
    w2 = w // 2
    return {
        "sets_top": roi[0:h2, 0:w2],
        "points_top": roi[0:h2, w2:w],
        "sets_bottom": roi[h2:h, 0:w2],
        "points_bottom": roi[h2:h, w2:w],
    }


def split_roi_boxes(x: int, y: int, w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    h2 = h // 2
    w2 = w // 2
    return {
        "sets_top": (x, y, w2, h2),
        "points_top": (x + w2, y, w - w2, h2),
        "sets_bottom": (x, y + h2, w2, h - h2),
        "points_bottom": (x + w2, y + h2, w - w2, h - h2),
    }


def _normalize_to_canvas(binary_digit: np.ndarray, target_w: int = 28, target_h: int = 28) -> np.ndarray:
    if binary_digit is None or binary_digit.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    h, w = binary_digit.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(binary_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_digit(cell: np.ndarray) -> np.ndarray:
    if cell is None or cell.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    # Ensure foreground digit is white on black background.
    if cv2.countNonZero(th) > (th.size // 2):
        th = cv2.bitwise_not(th)

    coords = cv2.findNonZero(th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = th[y : y + h, x : x + w]
    else:
        digit = np.zeros((28, 28), dtype=np.uint8)

    return _normalize_to_canvas(digit, 28, 28)


def digit_to_tensor(digit: np.ndarray) -> torch.Tensor:
    arr = digit.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, 28, 28]


def ensure_dataset_dirs(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for d in range(10):
        (dataset_dir / str(d)).mkdir(parents=True, exist_ok=True)


def parse_labels(label_str: str) -> Dict[str, int]:
    parts = [p.strip() for p in label_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--labels must be: sets_top,points_top,sets_bottom,points_bottom (4 digits).")

    labels: Dict[str, int] = {}
    for name, raw in zip(DIGIT_NAMES, parts):
        value = int(raw)
        if value < 0 or value > 9:
            raise ValueError("Each label must be a single digit 0..9.")
        labels[name] = value
    return labels


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    return Path(args.dataset_dir)


class ScoreDigitDataset(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        self.samples = []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for d in range(10):
            cls = dataset_dir / str(d)
            if not cls.exists():
                continue
            for ext in exts:
                for path in cls.glob(ext):
                    self.samples.append((path, d))

        if not self.samples:
            raise RuntimeError(f"No digit samples found in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((28, 28), dtype=np.uint8)
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = digit_to_tensor(img)
        return tensor, int(label)


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_collect(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.labels:
        raise ValueError("--labels is required in collect mode.")

    dataset_dir = resolve_dataset_dir(args)
    ensure_dataset_dirs(dataset_dir)
    labels = parse_labels(args.labels)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS from video.")

    start_frame = int(ts_to_seconds(args.start) * fps)
    end_frame = int(ts_to_seconds(args.end) * fps) if args.end else None
    if end_frame is not None and end_frame < start_frame:
        cap.release()
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Could not read frame at start timestamp")

    roi_x, roi_y, roi_w, roi_h = select_scoreboard_roi_from_frame(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved = 0
    boxes = split_roi_boxes(roi_x, roi_y, roi_w, roi_h)

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        if roi.size != 0:
            cells = split_roi_cells(roi)
            for name in DIGIT_NAMES:
                digit = preprocess_digit(cells[name])
                out = dataset_dir / str(labels[name]) / f"{frame_idx}_{name}.png"
                cv2.imwrite(str(out), digit)
                cv2.imshow(name, cv2.resize(digit, (140, 140), interpolation=cv2.INTER_NEAREST))
                saved += 1

        view = frame.copy()
        for name, (bx, by, bw, bh) in boxes.items():
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
            cv2.putText(view, name, (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(view, f"saved={saved}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("scoreboard_collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} digit samples to {dataset_dir.resolve()}")


def run_train(args: argparse.Namespace) -> None:
    dataset_dir = resolve_dataset_dir(args)
    ds = ScoreDigitDataset(dataset_dir)

    total = len(ds)
    if total < 2:
        raise RuntimeError("Need at least 2 samples to train/validate.")

    val_count = max(1, int(total * 0.2))
    train_count = total - val_count
    train_ds, val_ds = random_split(ds, [train_count, val_count], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(xb.size(0))

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == yb).sum().item())
                val_total += int(xb.size(0))

        avg_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        print(
            f"Epoch {epoch}/5 | "
            f"loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH.resolve()}")


def predict_digit(model: DigitCNN, device: str, digit_img: np.ndarray) -> Tuple[int, float]:
    x = digit_to_tensor(digit_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())


def run_infer(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train mode first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS from video.")

    start_frame = int(ts_to_seconds(args.start) * fps)
    end_frame = int(ts_to_seconds(args.end) * fps) if args.end else None
    if end_frame is not None and end_frame < start_frame:
        cap.release()
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Could not read frame at start timestamp")

    roi_x, roi_y, roi_w, roi_h = select_scoreboard_roi_from_frame(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    boxes = split_roi_boxes(roi_x, roi_y, roi_w, roi_h)

    frame_idx = start_frame
    last_score = None

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        if roi.size == 0:
            frame_idx += 1
            continue

        cells = split_roi_cells(roi)
        pred_digits: Dict[str, int] = {}
        view = frame.copy()

        for name in DIGIT_NAMES:
            digit_img = preprocess_digit(cells[name])
            pred, conf = predict_digit(model, device, digit_img)
            pred_digits[name] = pred

            cv2.imshow(name, cv2.resize(digit_img, (140, 140), interpolation=cv2.INTER_NEAREST))
            bx, by, bw, bh = boxes[name]
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
            cv2.putText(
                view,
                f"{name}: {pred} ({conf:.2f})",
                (bx, max(18, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        sets_top = pred_digits["sets_top"]
        points_top = pred_digits["points_top"]
        sets_bottom = pred_digits["sets_bottom"]
        points_bottom = pred_digits["points_bottom"]
        score = (sets_top, sets_bottom, points_top, points_bottom)
        if score != last_score:
            print(f"SETS: {sets_top}-{sets_bottom}")
            print(f"POINTS: {points_top}-{points_bottom}")
            last_score = score

        cv2.imshow("scoreboard_cnn_reader", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Complete CNN-based volleyball scoreboard reader")
    p.add_argument("--mode", required=True, choices=["collect", "train", "infer"])
    p.add_argument("--video", default=None, help="Path to input video (required for collect/infer)")
    p.add_argument("--start", default="00:00:00", help="Start timestamp HH:MM:SS")
    p.add_argument("--end", default=None, help="End timestamp HH:MM:SS")
    p.add_argument("--labels", default=None, help="Collect labels: sets_top,points_top,sets_bottom,points_bottom")
    p.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    p.add_argument("--dataset", default=None, help="Alternative dataset root directory")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode in ("collect", "infer") and not args.video:
        raise ValueError("--video is required for collect/infer modes.")

    if args.mode == "collect":
        run_collect(args)
        return

    if args.mode == "train":
        run_train(args)
        return

    run_infer(args)


if __name__ == "__main__":
    main()
