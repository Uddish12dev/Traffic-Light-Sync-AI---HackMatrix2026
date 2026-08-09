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

