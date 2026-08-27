"""Unit tests for data pre-processing functions (M3 task 1)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlops_catsdogs.data.preprocess import (
    image_to_array,
    label_from_path,
    resize_to_square,
    split_dataset,
)


def test_resize_to_square_shape_and_mode(rgb_image):
    out = resize_to_square(rgb_image, size=224)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_resize_to_square_converts_grayscale():
    gray = Image.new("L", (50, 70), color=128)
    out = resize_to_square(gray, size=224)
    assert out.mode == "RGB"
    assert out.size == (224, 224)


def test_image_to_array_dtype_and_shape(rgb_image):
    arr = image_to_array(rgb_image, size=224)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8
    assert arr.min() >= 0 and arr.max() <= 255


@pytest.mark.parametrize(
    "path,expected",
    [
        (Path("data/raw/dog.101.jpg"), 1),
        (Path("data/raw/cat.7.jpg"), 0),
        (Path("data/raw/dogs/xyz.png"), 1),
        (Path("data/raw/cats/abc.png"), 0),
    ],
)
def test_label_from_path(path, expected):
    assert label_from_path(path) == expected


def test_split_dataset_ratios_and_disjoint():
    items = list(range(1000))
    train, val, test = split_dataset(items, ratios=(0.8, 0.1, 0.1), seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    # disjoint + complete
    assert set(train) | set(val) | set(test) == set(items)
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))


def test_split_dataset_is_deterministic():
    items = list(range(200))
    a = split_dataset(items, seed=7)
    b = split_dataset(items, seed=7)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_split_dataset_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_dataset(list(range(10)), ratios=(0.5, 0.3, 0.1))
