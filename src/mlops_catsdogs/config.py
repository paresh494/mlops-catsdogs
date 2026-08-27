"""Central configuration. Values can be overridden via environment variables or params.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three parents up from this file (src/mlops_catsdogs/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224
CHANNELS = 3
CLASS_NAMES = ["cat", "dog"]  # index 0 = cat, index 1 = dog
# ImageNet-style normalisation constants (standard for 224x224 RGB CNNs)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def _env_path(name: str, default: Path) -> Path:
    val = os.getenv(name)
    return Path(val) if val else default


@dataclass
class Paths:
    repo_root: Path = REPO_ROOT
    raw_data: Path = field(default_factory=lambda: _env_path("RAW_DATA_DIR", REPO_ROOT / "data" / "raw"))
    processed_data: Path = field(default_factory=lambda: _env_path("PROCESSED_DATA_DIR", REPO_ROOT / "data" / "processed"))
    models: Path = field(default_factory=lambda: _env_path("MODELS_DIR", REPO_ROOT / "models"))
    model_file: Path = field(default_factory=lambda: _env_path("MODEL_PATH", REPO_ROOT / "models" / "model.pt"))
    metrics: Path = field(default_factory=lambda: _env_path("METRICS_DIR", REPO_ROOT / "metrics"))


@dataclass
class TrainConfig:
    epochs: int = int(os.getenv("EPOCHS", "5"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "1e-3"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    seed: int = int(os.getenv("SEED", "42"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    # data split ratios (train / val / test) -- must sum to 1.0
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)


PATHS = Paths()
TRAIN = TrainConfig()

# MLflow >= 3 puts the file store in maintenance mode; default to a local SQLite
# backend (artifacts land in ./mlartifacts). Override with MLFLOW_TRACKING_URI to
# point at a remote tracking server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "cats-vs-dogs")
