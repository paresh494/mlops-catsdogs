"""Generate a tiny synthetic 'cats vs dogs' dataset so the whole pipeline
(preprocess -> train -> serve -> deploy -> smoke test) can run end-to-end without
downloading the ~800 MB Kaggle dataset.

The two classes are linearly separable-ish (warm/red-biased vs cool/blue-biased
textures with different shapes) so the baseline CNN reaches well above 50%.

Usage:
    python scripts/generate_synthetic_data.py --per-class 120 --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _make_image(is_dog: bool, rng: np.random.Generator, size: int = 256) -> Image.Image:
    base = rng.normal(loc=128, scale=35, size=(size, size, 3))
    if is_dog:
        base[..., 2] += 45  # bluer / cooler
        base[..., 0] -= 20
    else:
        base[..., 0] += 45  # redder / warmer
        base[..., 2] -= 20
    arr = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = rng.integers(60, size - 60, size=2)
    r = int(rng.integers(25, 55))
    if is_dog:  # rectangle
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=(20, 20, 200), width=6)
    else:       # ellipse
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(200, 20, 20), width=6)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for cls, is_dog in (("cats", False), ("dogs", True)):
        d = args.out / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            _make_image(is_dog, rng).save(d / f"{cls[:-1]}.{i:04d}.jpg", quality=90)
        print(f"wrote {args.per_class} images -> {d}")


if __name__ == "__main__":
    main()
