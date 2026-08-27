"""API smoke tests using FastAPI's TestClient (exercises M2 endpoints)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _image_bytes():
    arr = np.random.default_rng(2).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict(client):
    r = client.post("/predict", files={"file": ("x.jpg", _image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("cat", "dog")
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
    assert "inference_ms" in body


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_metrics_endpoint(client):
    assert client.get("/metrics").status_code == 200
    s = client.get("/metrics/summary").json()
    assert s["request_count"] >= 1
