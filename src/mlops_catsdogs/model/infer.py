"""Inference utilities shared by the training code and the REST service.

Kept free of FastAPI imports so it can be unit-tested in isolation
(see ``tests/test_infer.py``).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mlops_catsdogs.config import CLASS_NAMES, PATHS
from mlops_catsdogs.data.augment import to_normalised_tensor
from mlops_catsdogs.model.cnn import SimpleCNN, build_model


def softmax(logits) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def save_checkpoint(model: SimpleCNN, path: Path | str, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "classes": CLASS_NAMES, "arch": "SimpleCNN"}
    if extra:
        payload["meta"] = extra
    torch.save(payload, path)
    return path


def load_model(path: Path | str | None = None, device: str = "cpu") -> SimpleCNN:
    """Load a trained :class:`SimpleCNN`. If *path* is missing, returns a randomly
    initialised model (useful for tests / first boot) and does not raise."""
    path = Path(path or PATHS.model_file)
    model = build_model()
    if path.exists():
        ckpt = torch.load(path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def _predict_tensor(model: SimpleCNN, tensor: torch.Tensor, device: str = "cpu") -> dict[str, Any]:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor.to(device)).cpu().numpy()[0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "label_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
    }


def predict_image(model: SimpleCNN, image: Image.Image, device: str = "cpu") -> dict[str, Any]:
    return _predict_tensor(model, to_normalised_tensor(image), device)


def predict_array(model: SimpleCNN, array: np.ndarray, device: str = "cpu") -> dict[str, Any]:
    """*array* is HWC uint8 (or float) RGB."""
    img = Image.fromarray(np.asarray(array).astype("uint8"), mode="RGB")
    return predict_image(model, img, device)


def predict_bytes(model: SimpleCNN, raw: bytes, device: str = "cpu") -> dict[str, Any]:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return predict_image(model, img, device)


class InferenceEngine:
    """Loads the model once and serves predictions."""

    def __init__(self, model_path: Path | str | None = None, device: str = "cpu"):
        self.model_path = Path(model_path or PATHS.model_file)
        self.device = device
        self.model = load_model(self.model_path, device)
        self.model_loaded = self.model_path.exists()

    def predict(self, raw: bytes) -> dict[str, Any]:
        return predict_bytes(self.model, raw, self.device)

    def reload(self) -> None:
        self.model = load_model(self.model_path, self.device)
        self.model_loaded = self.model_path.exists()
