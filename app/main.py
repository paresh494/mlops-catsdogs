"""FastAPI inference service for Cats vs Dogs.

Endpoints
    GET  /            - service metadata
    GET  /health      - liveness/readiness probe (M2, M4 smoke test)
    POST /predict     - multipart image upload -> {label, probabilities, ...}
    GET  /metrics     - Prometheus exposition (M5 monitoring)
    GET  /metrics/summary - human-readable counters (request count, latency)

Request/response metadata is logged for every call (M5). Image bytes are never
logged.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas import HealthResponse, MetricsSummary, PredictionResponse
from mlops_catsdogs import __version__
from mlops_catsdogs.model.infer import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("catsdogs.api")

# ---- Prometheus metrics -------------------------------------------------------
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
ERRORS = Counter("prediction_errors_total", "Failed predictions")
LATENCY = Histogram("request_latency_seconds", "Request latency (s)", ["path"])

# ---- lightweight in-app counters (M5, no external infra required) ------------
_STATE = {"request_count": 0, "prediction_count": 0, "error_count": 0}
_LATENCIES: deque[float] = deque(maxlen=1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = InferenceEngine()
    logger.info(
        "model loaded=%s path=%s", app.state.engine.model_loaded, app.state.engine.model_path
    )
    yield


app = FastAPI(title="Cats vs Dogs Inference API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    _STATE["request_count"] += 1
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        _STATE["error_count"] += 1
        logger.exception("unhandled error request_id=%s path=%s", rid, request.url.path)
        raise
    finally:
        elapsed = time.perf_counter() - start
        _LATENCIES.append(elapsed)
        path = request.url.path
        LATENCY.labels(path=path).observe(elapsed)
        REQUESTS.labels(method=request.method, path=path, status=str(locals().get("status", 500))).inc()
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.1f client=%s",
            rid, request.method, path, locals().get("status", 500), elapsed * 1000,
            request.client.host if request.client else "-",
        )
    response.headers["x-request-id"] = rid
    return response


@app.get("/", tags=["meta"])
def root():
    return {"service": "cats-vs-dogs", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request):
    eng: InferenceEngine = request.app.state.engine
    return HealthResponse(
        status="ok",
        model_loaded=eng.model_loaded,
        model_path=str(eng.model_path),
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: Request, file: UploadFile = File(...)):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"expected an image, got {file.content_type}")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    eng: InferenceEngine = request.app.state.engine
    t0 = time.perf_counter()
    try:
        result = eng.predict(raw)
    except Exception as exc:  # noqa: BLE001
        ERRORS.inc()
        _STATE["error_count"] += 1
        logger.exception("prediction failed request_id=%s", rid)
        raise HTTPException(status_code=422, detail=f"could not process image: {exc}") from exc
    dt_ms = (time.perf_counter() - t0) * 1000
    _STATE["prediction_count"] += 1
    PREDICTIONS.labels(label=result["label"]).inc()
    logger.info(
        "prediction request_id=%s filename=%s label=%s confidence=%.3f inference_ms=%.1f",
        rid, file.filename, result["label"], result["confidence"], dt_ms,
    )
    return PredictionResponse(**result, inference_ms=round(dt_ms, 2), request_id=rid)


@app.get("/metrics", tags=["monitoring"])
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary", response_model=MetricsSummary, tags=["monitoring"])
def metrics_summary():
    lat = sorted(_LATENCIES)
    avg = (sum(lat) / len(lat) * 1000) if lat else 0.0
    p95 = (lat[int(len(lat) * 0.95) - 1] * 1000) if lat else 0.0
    return MetricsSummary(
        request_count=_STATE["request_count"],
        prediction_count=_STATE["prediction_count"],
        error_count=_STATE["error_count"],
        avg_latency_ms=round(avg, 2),
        p95_latency_ms=round(p95, 2),
    )


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
