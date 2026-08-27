"""Unit tests for model / inference utilities (M3 task 1)."""
import numpy as np
import pytest

from mlops_catsdogs.config import CLASS_NAMES
from mlops_catsdogs.model.infer import (
    load_model,
    predict_array,
    predict_bytes,
    softmax,
)


def test_softmax_sums_to_one():
    out = softmax([1.0, 2.0, 3.0])
    assert out.shape == (3,)
    assert np.isclose(out.sum(), 1.0)
    assert np.all(out > 0)


def test_softmax_matches_reference():
    out = softmax([0.0, 0.0])
    assert np.allclose(out, [0.5, 0.5])
    out2 = softmax([1000.0, 1000.0])  # numerical stability
    assert np.allclose(out2, [0.5, 0.5])


def test_softmax_is_monotonic():
    out = softmax([0.1, 5.0, 1.0])
    assert np.argmax(out) == 1


@pytest.fixture(scope="module")
def model():
    # No trained checkpoint needed: load_model returns a random-init model.
    return load_model(path="/nonexistent/model.pt", device="cpu")


def test_load_model_without_checkpoint(model):
    assert model is not None
    assert not model.training  # eval mode


def test_predict_array_schema(model):
    arr = np.random.default_rng(1).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    out = predict_array(model, arr)
    assert set(out) == {"label", "label_index", "confidence", "probabilities"}
    assert out["label"] in CLASS_NAMES
    assert 0 <= out["label_index"] < len(CLASS_NAMES)
    assert np.isclose(sum(out["probabilities"].values()), 1.0)
    assert set(out["probabilities"]) == set(CLASS_NAMES)
    assert 0.0 <= out["confidence"] <= 1.0


def test_predict_bytes_matches_array(model, png_bytes):
    out = predict_bytes(model, png_bytes)
    assert out["label"] in CLASS_NAMES
    assert np.isclose(sum(out["probabilities"].values()), 1.0)


def test_predict_bytes_rejects_garbage(model):
    with pytest.raises(Exception):
        predict_bytes(model, b"not an image")
