from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#---- Defining All The Utilities ---
def get_logger(name: str = "traffic_ai", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent: calling twice returns the same logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger


def make_deterministic(seed: int = 0) -> None:
    """Seed Python, numpy, and torch RNGs. Also enables deterministic torch algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Defining the Model ---
class TrafficMultiTaskModel(nn.Module):
    """Deep Multi-Task model predicting waiting time (regression) and light color (classification)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Head for waiting time regression
        self.regression_head = nn.Linear(hidden_dim, 1)
        # Head for traffic light color classification (3 classes: 0=Red, 1=Yellow, 2=Green)
        self.classification_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        wait_time = self.regression_head(features)
        light_color = self.classification_head(features)
        return wait_time, light_color

# --- Loading and Preprocessing the dataset ---
def label_traffic_light_color(vehicle_count: float, lane_occupancy: float, waiting_time: float) -> int:
    """Label recommended traffic light state dynamically.

    0 = Red (low traffic, transition to close)
    1 = Yellow (moderate/transitional queue)
    2 = Green (high density / long wait, prioritize clearing)
    """
    if vehicle_count >= 70 or lane_occupancy >= 60.0 or waiting_time > 40.0:
        return 2  # Green
    elif vehicle_count < 35 or lane_occupancy < 25.0:
        return 0  # Red
    else:
        return 1  # Yellow


def load_and_preprocess_data(csv_path: str, seed: int = 42) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load traffic_dataset.csv, compute targets, split, and normalize features."""
    log = get_logger("data_loader")
    log.info("Loading dataset from %s", csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    log.info("Found %d records in dataset", len(rows))

    vehicle_counts = []
    average_speeds = []
    lane_occupancies = []
    flow_rates = []
    time_of_days = []
    waiting_times = []
    light_colors = []

    for idx, row in enumerate(rows):
        try:
            vc = float(row["vehicle_count"])
            avg_speed = float(row["average_speed"])
            occ = float(row["lane_occupancy"])
            fr = float(row["flow_rate"])
            tod = row["time_of_day"].strip().lower()
            wt = float(row["waiting_time"])

            # Compute classification label dynamically based on traffic parameters
            lc = label_traffic_light_color(vc, occ, wt)

            vehicle_counts.append(vc)
            average_speeds.append(avg_speed)
            lane_occupancies.append(occ)
            flow_rates.append(fr)
            time_of_days.append(tod)
            waiting_times.append(wt)
            light_colors.append(lc)
        except KeyError as e:
            raise KeyError(f"Missing expected column in row {idx}: {e}")
        except ValueError as e:
            raise ValueError(f"Value error parsing row {idx}: {e}")

    # Unique categorical categories mapping
    categories = sorted(list(set(time_of_days)))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Encode time_of_day
    time_encoded = []
    for tod in time_of_days:
        one_hot = [0.0] * len(categories)
        one_hot[cat_to_idx[tod]] = 1.0
        time_encoded.append(one_hot)

    # Convert to arrays
    num_features = np.stack(
        [
            np.array(vehicle_counts, dtype=np.float32),
            np.array(average_speeds, dtype=np.float32),
            np.array(lane_occupancies, dtype=np.float32),
            np.array(flow_rates, dtype=np.float32),
        ],
        axis=1,
    )
    cat_features = np.array(time_encoded, dtype=np.float32)
    y_wait = np.array(waiting_times, dtype=np.float32).reshape(-1, 1)
    y_light = np.array(light_colors, dtype=np.int64)

    # Train / Val split (80% / 20%)
    num_samples = len(y_wait)
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(num_samples * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Scale numerical features on train split to prevent data leakage
    train_num = num_features[train_idx]
    means = train_num.mean(axis=0)
    stds = train_num.std(axis=0)
    stds[stds == 0.0] = 1.0

    log.info("Scaling features with Train Means: %s", means)
    log.info("Scaling features with Train Stds: %s", stds)

    # Normalize entire dataset using training stats
    num_features_normalized = (num_features - means) / stds

    # Combine normalized numerical features and one-hot categorical features
    X = np.hstack([num_features_normalized, cat_features])

    X_train, y_wait_train, y_light_train = X[train_idx], y_wait[train_idx], y_light[train_idx]
    X_val, y_wait_val, y_light_val = X[val_idx], y_wait[val_idx], y_light[val_idx]

    return X_train, y_wait_train, y_light_train, X_val, y_wait_val, y_light_val, means, stds, categories

# ---Training Logic ---
def train_config(
    run_idx: int,
    lr: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    loaders: tuple[DataLoader, DataLoader],
    input_dim: int,
    device: torch.device,
    best_val_loss: float,
    save_path: str,
    means: np.ndarray,
    stds: np.ndarray,
    categories: list[str],
) -> tuple[float, float, bool]:
    """Train a single hyperparameter configuration. Returns val_mse, val_acc, and is_best."""
    log = get_logger(f"run_{run_idx}")
    log.info("Starting Run %d: lr=%s, hidden_dim=%d", run_idx, lr, hidden_dim)

    train_loader, val_loader = loaders

    model = TrafficMultiTaskModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Loss Functions
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    run_best_val_loss = float("inf")
    best_val_mse = float("inf")
    best_val_acc = 0.0
    improved_overall = False

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_sum = 0.0
        for batch_x, batch_wait, batch_light in train_loader:
            batch_x = batch_x.to(device)
            batch_wait = batch_wait.to(device)
            batch_light = batch_light.to(device)

            optimizer.zero_grad()
            pred_wait, pred_light = model(batch_x)

            loss_reg = criterion_reg(pred_wait, batch_wait)
            loss_clf = criterion_clf(pred_light, batch_light)

            # Combined loss
            loss = loss_reg + 10.0 * loss_clf

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        correct_light = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_wait, batch_light in val_loader:
                batch_x = batch_x.to(device)
                batch_wait = batch_wait.to(device)
                batch_light = batch_light.to(device)

                pred_wait, pred_light = model(batch_x)

                loss_reg = criterion_reg(pred_wait, batch_wait)
                loss_clf = criterion_clf(pred_light, batch_light)

                val_loss_sum += (loss_reg.item() + 10.0 * loss_clf.item()) * batch_x.size(0)
                val_mse_sum += loss_reg.item() * batch_x.size(0)

                _, predicted_classes = torch.max(pred_light, 1)
                correct_light += (predicted_classes == batch_light).sum().item()
                total_samples += batch_x.size(0)

        epoch_val_loss = val_loss_sum / total_samples
        epoch_val_mse = val_mse_sum / total_samples
        epoch_val_acc = correct_light / total_samples

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            log.info(
                "Epoch %02d/%02d | Combined Loss: %.4f | Wait MSE: %.4f | Light Acc: %.2f%%",
                epoch,
                epochs,
                epoch_val_loss,
                epoch_val_mse,
                epoch_val_acc * 100,
            )

        if epoch_val_loss < run_best_val_loss:
            run_best_val_loss = epoch_val_loss
            best_val_mse = epoch_val_mse
            best_val_acc = epoch_val_acc

            # Save if this is the best overall model across ALL runs
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                improved_overall = True
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "input_means": means.tolist(),
                    "input_stds": stds.tolist(),
                    "categories": categories,
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "val_loss": epoch_val_loss,
                    "val_mse": epoch_val_mse,
                    "val_acc": epoch_val_acc,
                }
                torch.save(checkpoint, save_path)

    log.info("Run %d Complete. Best Wait MSE: %.4f | Best Light Acc: %.2f%%", run_idx, best_val_mse, best_val_acc * 100)
    return best_val_mse, best_val_acc, improved_overall

# ---Perception Pipeline ---
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    cls: int


VEHICLE_CLS: set[int] = {2, 3, 5, 7}  # car, motorbike, bus, truck


class VehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.3) -> None:
        self.weights = weights
        self.conf = conf
        self._model: Any | None = None
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.weights)
        except ImportError as e:
            # Silent fallback if not installed, process_frame will handle simulated flow.
            pass

    def detect(self, frame, classes: set[int] | None = None) -> list[Detection]:
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("YOLO model not loaded (ultralytics package missing).")
        classes = classes or VEHICLE_CLS
        results = self._model.predict(frame, conf=self.conf, verbose=False)
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls.item())
                if cls not in classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(b.conf.item()), cls=cls))
        return out


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    age: int = 0
    last_seen: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    def update(self, bbox: tuple[float, float, float, float], frame_idx: int) -> None:
        x1, y1, x2, y2 = bbox
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.bbox = bbox
        self.age += 1
        self.last_seen = frame_idx
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)


class CentroidTracker:
    def __init__(self, max_assoc_dist: float = 80.0, max_age: int = 30) -> None:
        self.max_assoc_dist = max_assoc_dist
        self.max_age = max_age
        self._next_id = itertools.count(1)
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        used: set[int] = set()
        dets_sorted = sorted(detections, key=lambda d: -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        for d in dets_sorted:
            x1, y1, x2, y2 = d.bbox
            c = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            best_id, best_d = None, float("inf")
            for tid, t in self._tracks.items():
                if tid in used:
                    continue
                dx = t.centroid[0] - c[0]
                dy = t.centroid[1] - c[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_d and dist <= self.max_assoc_dist:
                    best_d = dist
                    best_id = tid
            if best_id is not None:
                self._tracks[best_id].update(d.bbox, frame_idx)
                used.add(best_id)
            else:
                tid = next(self._next_id)
                tr = Track(track_id=tid, bbox=d.bbox, centroid=c, age=1, last_seen=frame_idx, history=[c])
                self._tracks[tid] = tr
                used.add(tid)

        dead = [tid for tid, t in self._tracks.items() if frame_idx - t.last_seen > self.max_age]
        for tid in dead:
            del self._tracks[tid]
        return list(self._tracks.values())


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class LanePolygons:
    polygons: dict[str, list[tuple[float, float]]]
    metres_per_pixel: float = 0.05
    fps: float = 10.0


@dataclass
class LaneStats:
    queue_count: float = 0.0
    mean_speed_mps: float = 0.0
    arrival_rate_vps: float = 0.0


class QueueEstimator:
    def __init__(
        self,
        lanes: LanePolygons,
        stationary_px: float = 4.0,
        stationary_frames: int = 5,
        ema_alpha: float = 0.3,
    ) -> None:
        self.lanes = lanes
        self.stationary_px = stationary_px
        self.stationary_frames = stationary_frames
        self.ema_alpha = ema_alpha
        self._ema: dict[str, LaneStats] = {name: LaneStats() for name in lanes.polygons}
        self._prev_ids: set[int] = set()
        self._new_ids: set[int] = set()

    def update(self, tracks: Iterable[Track], frame_idx: int) -> dict[str, LaneStats]:
        tracks = list(tracks)
        per_lane: dict[str, LaneStats] = {name: LaneStats() for name in self.lanes.polygons}
        for t in tracks:
            cx, cy = t.centroid
            if len(t.history) >= 2:
                dx = t.history[-1][0] - t.history[-2][0]
                dy = t.history[-1][1] - t.history[-2][1]
                speed_pxpf = (dx * dx + dy * dy) ** 0.5
            else:
                speed_pxpf = 0.0
            is_stationary = speed_pxpf < self.stationary_px
            for name, poly in self.lanes.polygons.items():
                if not point_in_polygon(cx, cy, poly):
                    continue
                if is_stationary and t.age >= self.stationary_frames:
                    per_lane[name].queue_count += 1.0
                speed_mps = speed_pxpf * self.lanes.fps * self.lanes.metres_per_pixel
                per_lane[name].mean_speed_mps += speed_mps
            if t.track_id not in self._prev_ids:
                self._new_ids.add(t.track_id)
        self._prev_ids = {t.track_id for t in tracks}

        new_count = len(self._new_ids)
        self._new_ids.clear()
        for name, st in per_lane.items():
            prev = self._ema[name]
            prev.queue_count = (1 - self.ema_alpha) * prev.queue_count + self.ema_alpha * st.queue_count
            prev.mean_speed_mps = (1 - self.ema_alpha) * prev.mean_speed_mps + self.ema_alpha * st.mean_speed_mps
            prev.arrival_rate_vps = (1 - self.ema_alpha) * prev.arrival_rate_vps + self.ema_alpha * (
                new_count / max(1.0, self.lanes.fps)
            )
            self._ema[name] = prev
        return {name: LaneStats(**vars(s)) for name, s in self._ema.items()}

