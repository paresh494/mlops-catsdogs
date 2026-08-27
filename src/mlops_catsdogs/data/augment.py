"""Lightweight data augmentation + tensor conversion (implemented with PIL/NumPy so
that torchvision is not a hard dependency).

``build_transform(train=True)`` returns a callable ``PIL.Image -> torch.FloatTensor``
of shape ``(3, 224, 224)`` normalised with ImageNet statistics. When ``train`` is
True the callable also applies random horizontal flip, small rotation and
brightness/contrast jitter for better generalisation.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance

from mlops_catsdogs.config import IMAGE_SIZE, NORM_MEAN, NORM_STD


def random_horizontal_flip(img: Image.Image, p: float = 0.5) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img


def random_rotation(img: Image.Image, degrees: float = 15.0) -> Image.Image:
    angle = random.uniform(-degrees, degrees)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def random_color_jitter(img: Image.Image, amount: float = 0.2) -> Image.Image:
    for Enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-amount, amount)
        img = Enhancer(img).enhance(factor)
    return img


def to_normalised_tensor(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL image -> normalised CHW float tensor."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0            # HWC in [0, 1]
    arr = (arr - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)
    chw = np.transpose(arr, (2, 0, 1)).copy()                  # CHW
    return torch.from_numpy(chw)


class Transform:
    """Picklable transform callable (a closure would not pickle for DataLoader workers)."""

    def __init__(self, train: bool, size: int = IMAGE_SIZE):
        self.train = train
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.train:
            img = random_horizontal_flip(img)
            img = random_rotation(img)
            img = random_color_jitter(img)
        return to_normalised_tensor(img, self.size)


def build_transform(train: bool = True, size: int = IMAGE_SIZE) -> Transform:
    return Transform(train=train, size=size)
