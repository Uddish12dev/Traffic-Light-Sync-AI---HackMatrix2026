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
