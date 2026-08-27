from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_path: str
    version: str


class PredictionResponse(BaseModel):
    label: str = Field(examples=["dog"])
    label_index: int = Field(examples=[1])
    confidence: float = Field(examples=[0.87])
    probabilities: dict[str, float] = Field(examples=[{"cat": 0.13, "dog": 0.87}])
    inference_ms: float
    request_id: str


class MetricsSummary(BaseModel):
    request_count: int
    prediction_count: int
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
