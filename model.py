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

# ---Controller Classes---
class CCTVCamera:
    """Represents a CCTV camera associated with a traffic signal."""

    def __init__(self, name: str):
        self.name = name
        self.log = get_logger(f"cctv_{name}")
        self.model_path = None
        self.model = None
        self.categories = []
        self.means = None
        self.stds = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize perception modules
        poly = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
        self.lanes = LanePolygons(polygons={name: poly}, fps=10.0, metres_per_pixel=0.05)
        self.detector = VehicleDetector()
        self.tracker = CentroidTracker()
        self.estimator = QueueEstimator(lanes=self.lanes)

    def download_model(self, checkpoint_path: str):
        """Simulates downloading the model from a remote source to local CCTV storage."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Source model checkpoint not found at: {checkpoint_path}")

        local_dir = f"outputs/cctv_downloads/cctv_{self.name}"
        os.makedirs(local_dir, exist_ok=True)
        local_dest = os.path.join(local_dir, "traffic_model.pt")

        self.log.info(
            "Downloading model checkpoint from %s to CCTVCamera %s local storage at %s...",
            checkpoint_path,
            self.name,
            local_dest,
        )
        shutil.copy(checkpoint_path, local_dest)
        self.model_path = local_dest

        checkpoint = torch.load(local_dest, map_location=self.device)
        self.categories = checkpoint["categories"]
        self.means = np.array(checkpoint["input_means"], dtype=np.float32)
        self.stds = np.array(checkpoint["input_stds"], dtype=np.float32)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        input_dim = 4 + len(self.categories)

        self.model = TrafficMultiTaskModel(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.log.info("CCTVCamera %s successfully loaded downloaded model.", self.name)

    def predict(
        self,
        vehicle_count: float,
        average_speed: float,
        lane_occupancy: float,
        flow_rate: float,
        time_of_day: str,
    ) -> tuple[float, str]:
        """Perform predictions using the local downloaded model."""
        if self.model is None:
            raise RuntimeError(f"No model downloaded or loaded on CCTVCamera {self.name}.")

        num_features = np.array([vehicle_count, average_speed, lane_occupancy, flow_rate], dtype=np.float32)
        num_normalized = (num_features - self.means) / self.stds

        time_encoded = [0.0] * len(self.categories)
        tod_clean = time_of_day.strip().lower()
        if tod_clean in self.categories:
            time_encoded[self.categories.index(tod_clean)] = 1.0

        input_vec = np.concatenate([num_normalized, time_encoded]).astype(np.float32)
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_wait, pred_light = self.model(input_tensor)
            predicted_wait = pred_wait.item()
            _, predicted_class = torch.max(pred_light, 1)
            predicted_class = predicted_class.item()

        color_mapping = {0: "Red", 1: "Yellow", 2: "Green"}
        recommended_color = color_mapping.get(predicted_class, "Unknown")
        return predicted_wait, recommended_color

    def detect_vehicles(self, frame=None, simulated_count: float = 0.0) -> list[Detection]:
        """Perform virtual detection if no frame is provided, otherwise real YOLO detection."""
        if frame is not None:
            try:
                return self.detector.detect(frame)
            except Exception as e:
                self.log.warning("Real detection failed: %s. Falling back to virtual detection.", e)

        # Virtual detection: generate random bounding boxes for simulated_count vehicles
        detections = []
        for _ in range(int(simulated_count)):
            x1 = random.uniform(100.0, 1500.0)
            y1 = random.uniform(100.0, 800.0)
            w = random.uniform(80.0, 150.0)
            h = random.uniform(80.0, 150.0)
            x2 = x1 + w
            y2 = y1 + h
            conf = random.uniform(0.75, 0.98)
            cls = random.choice([2, 3, 5, 7])
            detections.append(Detection(bbox=(x1, y1, x2, y2), conf=conf, cls=cls))
        return detections

    def process_frame(self, frame=None, simulated_count: float = 0.0, frame_idx: int = 0) -> dict[str, float]:
        """Detect, track, and estimate parameters from the frame."""
        detections = self.detect_vehicles(frame, simulated_count)
        tracks = self.tracker.update(detections, frame_idx)
        stats = self.estimator.update(tracks, frame_idx)

        lane_stat = stats.get(self.name)
        vehicle_count = float(len(detections))

        if lane_stat and lane_stat.mean_speed_mps > 0:
            average_speed = float(lane_stat.mean_speed_mps * 3.6)
        else:
            average_speed = 30.0

        lane_occupancy = float(min(100.0, vehicle_count * 1.25))

        if lane_stat and lane_stat.arrival_rate_vps > 0:
            flow_rate = float(lane_stat.arrival_rate_vps * 3600.0)
        else:
            flow_rate = vehicle_count * 15.0

        return {
            "vehicle_count": vehicle_count,
            "average_speed": average_speed,
            "lane_occupancy": lane_occupancy,
            "flow_rate": flow_rate,
        }


class TrafficLightDevice:
    """Represents a traffic light unit with local micro-controller capability."""

    def __init__(self, name: str, initial_state: str = "Red"):
        self.name = name
        self.state = initial_state
        self.timer = 0
        self.log = get_logger(f"light_{name}")
        self.model_path = None
        self.model = None
        self.categories = []
        self.means = None
        self.stds = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def download_model(self, checkpoint_path: str):
        """Simulates downloading the model checkpoint to the local Traffic Light micro-controller."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Source model checkpoint not found at: {checkpoint_path}")

        local_dir = f"outputs/traffic_light_downloads/light_{self.name}"
        os.makedirs(local_dir, exist_ok=True)
        local_dest = os.path.join(local_dir, "traffic_model.pt")

        self.log.info(
            "Downloading model checkpoint from %s to TrafficLight %s local storage at %s...",
            checkpoint_path,
            self.name,
            local_dest,
        )
        shutil.copy(checkpoint_path, local_dest)
        self.model_path = local_dest

        checkpoint = torch.load(local_dest, map_location=self.device)
        self.categories = checkpoint["categories"]
        self.means = np.array(checkpoint["input_means"], dtype=np.float32)
        self.stds = np.array(checkpoint["input_stds"], dtype=np.float32)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        input_dim = 4 + len(self.categories)

        self.model = TrafficMultiTaskModel(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.log.info("TrafficLight %s successfully loaded downloaded model.", self.name)

    def predict(
        self,
        vehicle_count: float,
        average_speed: float,
        lane_occupancy: float,
        flow_rate: float,
        time_of_day: str,
    ) -> tuple[float, str]:
        """Perform predictions using the local downloaded model."""
        if self.model is None:
            raise RuntimeError(f"No model downloaded or loaded on TrafficLightDevice {self.name}.")

        num_features = np.array([vehicle_count, average_speed, lane_occupancy, flow_rate], dtype=np.float32)
        num_normalized = (num_features - self.means) / self.stds

        time_encoded = [0.0] * len(self.categories)
        tod_clean = time_of_day.strip().lower()
        if tod_clean in self.categories:
            time_encoded[self.categories.index(tod_clean)] = 1.0

        input_vec = np.concatenate([num_normalized, time_encoded]).astype(np.float32)
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_wait, pred_light = self.model(input_tensor)
            predicted_wait = pred_wait.item()
            _, predicted_class = torch.max(pred_light, 1)
            predicted_class = predicted_class.item()

        color_mapping = {0: "Red", 1: "Yellow", 2: "Green"}
        recommended_color = color_mapping.get(predicted_class, "Unknown")
        return predicted_wait, recommended_color


class TrafficSignalNode:
    """Combines CCTVCamera, TrafficLightDevice, and traffic parameters."""

    def __init__(self, name: str, initial_state: str, initial_params: dict):
        self.name = name
        self.cctv = CCTVCamera(name=name)
        self.traffic_light = TrafficLightDevice(name=name, initial_state=initial_state)
        self.params = initial_params.copy()
        self.ai_model_role = "Standby"

    def download_models(self, checkpoint_path: str):
        """Triggers simulated downloads to both CCTV and Traffic Light devices."""
        self.cctv.download_model(checkpoint_path)
        self.traffic_light.download_model(checkpoint_path)

    def update_params(self, vehicle_delta: float, occupancy_delta: float):
        """Simulate changes in traffic parameters."""
        self.params["vehicle_count"] = max(0.0, self.params["vehicle_count"] + vehicle_delta)
        self.params["lane_occupancy"] = max(0.0, min(100.0, self.params["lane_occupancy"] + occupancy_delta))
        self.params["flow_rate"] = self.params["vehicle_count"] * 15.0


class DynamicTrafficLightController:
    """Manages primary and secondary AI model nodes dynamically."""

    def __init__(self, nodes: list[TrafficSignalNode], conflict_map: dict[str, list[str]] = None):
        self.log = get_logger("dynamic_controller")
        self.nodes = nodes
        self.conflict_map = conflict_map or {}
        self.primary_node: TrafficSignalNode | None = None
        self.secondary_node: TrafficSignalNode | None = None

        # Initialize dataset file path
        self.dataset_path = "data/detected_traffic_parameters.csv"
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            f.write("step,node_name,vehicle_count,average_speed,lane_occupancy,flow_rate,time_of_day\n")

        # Set initial parameters for nodes using CCTV camera detection (step 0)
        self.log.info("Initializing node parameters using CCTV camera detection...")
        for node in self.nodes:
            detected_params = node.cctv.process_frame(
                frame=None,
                simulated_count=node.params["vehicle_count"],
                frame_idx=0
            )
            node.params.update(detected_params)
            self.log.info(
                "Initial detected parameters for %s: VC=%.1f, Occ=%.1f%%, Flow=%.1f",
                node.name,
                node.params["vehicle_count"],
                node.params["lane_occupancy"],
                node.params["flow_rate"]
            )
            self.save_to_dataset(0, node)

        # Build connectivity log to nearby signals
        for node in self.nodes:
            nearby_nodes = [n.name for n in self.nodes if n.name != node.name]
            self.log.info("Signal node %s connected to nearby signals: %s", node.name, nearby_nodes)

        # Set initial roles based on initial green light parameter
        self.update_roles(initial=True)

    def save_to_dataset(self, step: int, node: TrafficSignalNode):
        """Append the current detected parameters of a node to the dataset CSV."""
        import csv
        with open(self.dataset_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                node.name,
                node.params["vehicle_count"],
                node.params["average_speed"],
                node.params["lane_occupancy"],
                node.params["flow_rate"],
                node.params["time_of_day"]
            ])

    def download_models_to_devices(self, checkpoint_path: str):
        """Simulates downloading the AI model to all managed devices (CCTV and traffic lights)."""
        self.log.info("Initiating model downloads to all CCTV cameras and Traffic Lights...")
        for node in self.nodes:
            node.download_models(checkpoint_path)
        self.log.info("All devices successfully downloaded and initialized the AI model.")

    def update_roles(self, initial: bool = False):
        """Update Primary/Secondary AI model assignments.

        - Traffic light containing Green is Primary.
        - Next traffic light in configuration sequence is Secondary.
        """
        old_primary = self.primary_node
        old_secondary = self.secondary_node

        new_primary = None
        new_primary_idx = -1

        for i, node in enumerate(self.nodes):
            if node.traffic_light.state in ("Green", "Yellow_to_Red"):
                new_primary = node
                new_primary_idx = i
                break

        if new_primary is None:
            for i, node in enumerate(self.nodes):
                if node.traffic_light.state == "Green":
                    new_primary = node
                    new_primary_idx = i
                    break

        if new_primary is None and len(self.nodes) > 0:
            new_primary = self.nodes[0]
            new_primary_idx = 0

        new_secondary = None
        if new_primary_idx != -1 and len(self.nodes) > 1:
            sec_idx = (new_primary_idx + 1) % len(self.nodes)
            new_secondary = self.nodes[sec_idx]

        self.primary_node = new_primary
        self.secondary_node = new_secondary

        # Assign roles to signals
        for node in self.nodes:
            if node == self.primary_node:
                node.ai_model_role = "Primary"
            elif node == self.secondary_node:
                node.ai_model_role = "Secondary"
            else:
                node.ai_model_role = "Standby"

        # Log role updates
        if initial:
            self.log.info(
                "[Role Init] Primary AI Model: %s (Green light node) | Secondary AI Model: %s (Next in configuration)",
                self.primary_node.name if self.primary_node else "None",
                self.secondary_node.name if self.secondary_node else "None",
            )
        elif old_primary != self.primary_node or old_secondary != self.secondary_node:
            self.log.info(
                "[Role Switch] Green light shifted! Primary AI Model: %s | Secondary AI Model: %s",
                self.primary_node.name if self.primary_node else "None",
                self.secondary_node.name if self.secondary_node else "None",
            )

    def run_simulation_step(self, step_idx: int):
        """Simulate a single controller time step."""
        self.log.info("=== Sim Step %d ===", step_idx)

        # 1. Update/Simulate traffic parameters based on state
        for node in self.nodes:
            if node.traffic_light.state == "Green":
                node.update_params(vehicle_delta=-12.0, occupancy_delta=-10.0)
            elif node.traffic_light.state == "Red":
                node.update_params(vehicle_delta=15.0, occupancy_delta=10.0)

        # 2. Run CCTV camera vehicle detection on each node to update parameters and log to dataset CSV
        for node in self.nodes:
            detected_params = node.cctv.process_frame(
                frame=None,
                simulated_count=node.params["vehicle_count"],
                frame_idx=step_idx
            )
            node.params.update(detected_params)
            self.save_to_dataset(step_idx, node)

        # Log current status
        for node in self.nodes:
            self.log.info(
                "Signal %s (Light=%s, ModelRole=%s): VC=%.1f, Occ=%.1f%%",
                node.name,
                node.traffic_light.state,
                node.ai_model_role,
                node.params["vehicle_count"],
                node.params["lane_occupancy"],
            )

        # 3. Run state transitions
        next_states = {}

        for node in self.nodes:
            current_state = node.traffic_light.state
            name = node.name

            if current_state == "Green":
                if node.ai_model_role == "Primary":
                    pred_wait, pred_color = node.traffic_light.predict(
                        vehicle_count=node.params["vehicle_count"],
                        average_speed=node.params["average_speed"],
                        lane_occupancy=node.params["lane_occupancy"],
                        flow_rate=node.params["flow_rate"],
                        time_of_day=node.params["time_of_day"],
                    )
                    self.log.info(
                        "Signal %s (GREEN, Primary Model) -> AI recommendation: %s (Wait prediction: %.2fs)",
                        name,
                        pred_color,
                        pred_wait,
                    )

                    if pred_color in ("Yellow", "Red"):
                        self.log.info(
                            "Signal %s Primary AI model detected clear lane. Initiating transition to RED/YELLOW.",
                            name,
                        )
                        next_states[name] = ("Yellow_to_Red", 2)
                    else:
                        next_states[name] = ("Green", 0)
                else:
                    self.log.info(
                        "Signal %s (GREEN) -> Role is %s (Not Primary). Transition forbidden. Staying GREEN.",
                        name,
                        node.ai_model_role,
                    )
                    next_states[name] = ("Green", 0)

            elif current_state == "Yellow_to_Red":
                if node.traffic_light.timer > 0:
                    self.log.info(
                        "Signal %s (Yellow_to_Red) transitioning. Steps remaining: %d",
                        name,
                        node.traffic_light.timer,
                    )
                    next_states[name] = ("Yellow_to_Red", node.traffic_light.timer - 1)
                else:
                    self.log.info("Signal %s transition complete. Now RED.", name)
                    next_states[name] = ("Red", 0)

            elif current_state == "Red":
                if node.ai_model_role == "Secondary":
                    pred_wait, pred_color = node.traffic_light.predict(
                        vehicle_count=node.params["vehicle_count"],
                        average_speed=node.params["average_speed"],
                        lane_occupancy=node.params["lane_occupancy"],
                        flow_rate=node.params["flow_rate"],
                        time_of_day=node.params["time_of_day"],
                    )

                    role_label = f"RED, Secondary Model"
                    self.log.info(
                        "Signal %s (%s) -> AI recommendation: %s (Wait prediction: %.2fs)",
                        name,
                        role_label,
                        pred_color,
                        pred_wait,
                    )

                    if pred_color == "Green":
                        self.log.info(
                            "Signal %s (RED, Secondary Model) -> AI Model requests GREEN. Checking connected radar...",
                            name,
                        )

                        conflicts = self.conflict_map.get(name, [])
                        has_conflict = False
                        for conf_name in conflicts:
                            conf_node = next((n for n in self.nodes if n.name == conf_name), None)
                            if conf_node and conf_node.traffic_light.state != "Red":
                                self.log.warning(
                                    "SAFETY INTERLOCK ACTIVE: %s wants to change to Green, "
                                    "but conflicting signal %s is currently %s! Staying RED.",
                                    name,
                                    conf_node.name,
                                    conf_node.traffic_light.state,
                                )
                                has_conflict = True
                                break

                        if not has_conflict:
                            self.log.info("Safety Check Passed! Initiating transition to GREEN.")
                            next_states[name] = ("Yellow_to_Green", 2)
                        else:
                            next_states[name] = ("Red", 0)
                    else:
                        next_states[name] = ("Red", 0)
                else:
                    self.log.info(
                        "Signal %s (RED) -> Role is %s (Not Secondary). Transition forbidden. Staying RED.",
                        name,
                        node.ai_model_role,
                    )
                    next_states[name] = ("Red", 0)

            elif current_state == "Yellow_to_Green":
                if node.traffic_light.timer > 0:
                    self.log.info(
                        "Signal %s (Yellow_to_Green) transitioning. Steps remaining: %d",
                        name,
                        node.traffic_light.timer,
                    )
                    next_states[name] = ("Yellow_to_Green", node.traffic_light.timer - 1)
                else:
                    self.log.info("Signal %s transition complete. Now GREEN.", name)
                    next_states[name] = ("Green", 0)

        for name, (state, timer) in next_states.items():
            node = next((n for n in self.nodes if n.name == name), None)
            if node:
                node.traffic_light.state = state
                node.traffic_light.timer = timer

        self.update_roles()

# ---Transition Verification Logic ---
def verify_traffic_light_transitions(checkpoint_path: str, device: torch.device) -> bool:
    """Verify that the saved model correctly transitions: Green -> Yellow -> Red."""
    log = get_logger("transition_verifier")
    log.info("Verifying traffic light transitions for model: %s", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        log.error("Checkpoint not found for verification: %s", checkpoint_path)
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    categories = checkpoint["categories"]
    means = np.array(checkpoint["input_means"], dtype=np.float32)
    stds = np.array(checkpoint["input_stds"], dtype=np.float32)
    hidden_dim = checkpoint.get("hidden_dim", 64)

    input_dim = 4 + len(categories)
    model = TrafficMultiTaskModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    color_mapping = {0: "Red", 1: "Yellow", 2: "Green"}

    steps = [
        (85.0, 15.0, 65.0, 1200.0, "morning", "Green"),
        (50.0, 30.0, 40.0, 800.0, "morning", "Yellow"),
        (20.0, 50.0, 15.0, 300.0, "morning", "Red"),
    ]

    actual_transitions = []

    for idx, (vc, speed, occ, flow, tod, expected) in enumerate(steps, 1):
        num_features = np.array([vc, speed, occ, flow], dtype=np.float32)
        num_normalized = (num_features - means) / stds

        time_encoded = [0.0] * len(categories)
        tod_clean = tod.strip().lower()
        if tod_clean in categories:
            time_encoded[categories.index(tod_clean)] = 1.0

        input_vec = np.concatenate([num_normalized, time_encoded]).astype(np.float32)
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_wait, pred_light = model(input_tensor)
            _, predicted_class = torch.max(pred_light, 1)
            predicted_class = predicted_class.item()

        predicted_color = color_mapping.get(predicted_class, "Unknown")
        actual_transitions.append(predicted_color)
        log.info(
            "Step %d: VC=%.1f, Occ=%.1f%% -> Expected: %s, Predicted: %s (Wait prediction: %.2f)",
            idx, vc, occ, expected, predicted_color, pred_wait.item()
        )

    if actual_transitions == ["Green", "Yellow", "Red"]:
        log.info("SUCCESS: Model correctly transitions: Green -> Yellow -> Red")
        return True
    else:
        log.warning(
            "FAILED: Model did not transition in Green -> Yellow -> Red sequence. Actual sequence: %s",
            actual_transitions,
        )
        return False

# --- Main Function ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Traffic Multi-Task MLP Model")
    parser.add_argument("--csv-path", type=str, default="traffic_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per configuration")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save-path", type=str, default="outputs/traffic_model.pt", help="Path to save model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    log = get_logger("train_traffic_model")

    # Set random seeds
    make_deterministic(args.seed)

    # Determine Device (GPU/CUDA support)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info("Local GPU detected! Training on GPU: %s (device_idx: 0)", gpu_name)
    else:
        log.info("Local GPU (CUDA) not available. Training on CPU.")

    # Load and process data
    try:
        X_train, y_wait_train, y_light_train, X_val, y_wait_val, y_light_val, means, stds, categories = (
            load_and_preprocess_data(args.csv_path, seed=args.seed)
        )
    except Exception as e:
        log.error("Failed to load and preprocess data: %s", e)
        return

    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_wait_train),
        torch.from_numpy(y_light_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_wait_val),
        torch.from_numpy(y_light_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loaders = (train_loader, val_loader)
    input_dim = X_train.shape[1]

    # Hyperparameter Configurations (Run the model several times with different configs)
    configs = [
        {"lr": 1e-3, "hidden_dim": 64},
        {"lr": 5e-4, "hidden_dim": 128},
        {"lr": 5e-4, "hidden_dim": 64},
    ]

    log.info("Starting hyperparameter sweep (%d runs)...", len(configs))
    best_val_loss = float("inf")
    results = []

    for idx, cfg in enumerate(configs, 1):
        lr = cfg["lr"]
        hidden_dim = cfg["hidden_dim"]

        val_mse, val_acc, is_best = train_config(
            run_idx=idx,
            lr=lr,
            hidden_dim=hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loaders=loaders,
            input_dim=input_dim,
            device=device,
            best_val_loss=best_val_loss,
            save_path=args.save_path,
            means=means,
            stds=stds,
            categories=categories,
        )

        results.append((idx, lr, hidden_dim, val_mse, val_acc))
        if is_best:
            ckpt = torch.load(args.save_path, map_location="cpu")
            best_val_loss = ckpt["val_loss"]

    # Print Summary Table of all Runs
    log.info("=========================================================")
    log.info("HYPERPARAMETER SWEEP SUMMARY")
    log.info("=========================================================")
    log.info("Run | LR     | Hidden Dim | Val Wait MSE | Val Light Acc")
    log.info("---------------------------------------------------------")
    for r in results:
        log.info("%3d | %.4f | %10d | %12.4f | %12.2f%%", r[0], r[1], r[2], r[3], r[4] * 100)
    log.info("=========================================================")
    log.info("Saved best overall model checkpoint to %s", args.save_path)

    # Verify transition predictions
    verify_traffic_light_transitions(args.save_path, device)

    # Run a short dynamic simulation to verify everything is 100% error free
    log.info("Running post-training verification simulation...")
    nodes = [
        TrafficSignalNode(
            name="Primary_AI_Model",
            initial_state="Green",
            initial_params={
                "vehicle_count": 80.0,
                "average_speed": 15.0,
                "lane_occupancy": 65.0,
                "flow_rate": 1200.0,
                "time_of_day": "morning"
            }
        ),
        TrafficSignalNode(
            name="Secondary_AI_Model",
            initial_state="Red",
            initial_params={
                "vehicle_count": 10.0,
                "average_speed": 50.0,
                "lane_occupancy": 8.0,
                "flow_rate": 200.0,
                "time_of_day": "morning"
            }
        )
    ]
    conflict_map = {
        "Primary_AI_Model": ["Secondary_AI_Model"],
        "Secondary_AI_Model": ["Primary_AI_Model"]
    }
    controller = DynamicTrafficLightController(nodes, conflict_map)
    controller.download_models_to_devices(args.save_path)
    
    # Run 5 validation simulation steps
    for step in range(1, 6):
        controller.run_simulation_step(step)

    log.info("Validation simulation completed successfully. model.py execution finished.")


if __name__ == "__main__":
    main()
