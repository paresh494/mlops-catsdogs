"""Data pre-processing: discover raw images, resize to 224x224 RGB, split 80/10/10.

The functions here are deliberately small and pure so they can be unit-tested (see
``tests/test_preprocess.py``).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from mlops_catsdogs.config import CLASS_NAMES, IMAGE_SIZE

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_to_square(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Return an ``RGB`` copy of *img* resized to ``size x size`` pixels."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def image_to_array(img: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Convert a PIL image to a ``uint8`` array of shape ``(size, size, 3)``."""
    arr = np.asarray(resize_to_square(img, size), dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError(f"unexpected array shape {arr.shape}, expected {(size, size, 3)}")
    return arr


def label_from_path(path: Path) -> int:
    """Infer the class index from a file path.

    A file is a *dog* if the token ``dog`` appears in the file name or its parent
    directory name, otherwise it is a *cat*. Matches the Kaggle "Dogs vs Cats"
    layout (``cat.123.jpg`` / ``dog.123.jpg`` or ``cats/`` & ``dogs/`` folders).
    """
    haystack = f"{path.parent.name}/{path.name}".lower()
    return 1 if "dog" in haystack else 0


def discover_images(raw_dir: Path | str) -> list[tuple[Path, int]]:
    """Walk *raw_dir* and return a sorted list of ``(path, label)`` pairs."""
    raw_dir = Path(raw_dir)
    items: list[tuple[Path, int]] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            items.append((p, label_from_path(p)))
    return items


def split_dataset(
    items: Sequence,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list, list, list]:
    """Shuffle *items* deterministically and split into train / val / test.

    The three returned lists are disjoint and their union is a permutation of
    *items*. ``ratios`` must sum to (approximately) 1.0.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def _save_split(split_name: str, pairs: Iterable[tuple[Path, int]], out_dir: Path, size: int) -> list[dict]:
    rows: list[dict] = []
    for src, label in pairs:
        cls = CLASS_NAMES[label]
        dst_dir = out_dir / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{src.stem}.jpg"
        with Image.open(src) as im:
            resize_to_square(im, size).save(dst, format="JPEG", quality=95)
        rows.append({"split": split_name, "class": cls, "label": label, "path": str(dst.relative_to(out_dir))})
    return rows


def preprocess_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    size: int = IMAGE_SIZE,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    """End-to-end pre-processing. Returns a summary dict and writes a manifest.csv."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    items = discover_images(raw_dir)
    if not items:
        raise FileNotFoundError(
            f"no images found under {raw_dir}. Download the Kaggle Dogs-vs-Cats dataset "
            f"into it, or run scripts/generate_synthetic_data.py for a smoke dataset."
        )
    train, val, test = split_dataset(items, ratios, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    manifest += _save_split("train", train, out_dir, size)
    manifest += _save_split("val", val, out_dir, size)
    manifest += _save_split("test", test, out_dir, size)

    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "label", "path"])
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "n_total": len(items),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "image_size": size,
        "classes": CLASS_NAMES,
    }
    return summary


if __name__ == "__main__":  # pragma: no cover
    import json

    from mlops_catsdogs.config import PATHS, TRAIN

    print(json.dumps(preprocess_dataset(PATHS.raw_data, PATHS.processed_data, IMAGE_SIZE, TRAIN.split_ratios, TRAIN.seed), indent=2))
