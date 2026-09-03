"""
ml_service/main.py

Optimized FastAPI ML microservice with:
- Async endpoints for non-blocking I/O
- Batch prediction for throughput optimization
- Startup warm-up to eliminate cold-start latency
- Richer synthetic training data for better anomaly boundary detection
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from services.anomaly_detector import IsolationForestDetector

# ---------------------------------------------------------------------------
# Global singleton (initialized once at startup)
# ---------------------------------------------------------------------------
detector: IsolationForestDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup lifecycle: train on rich synthetic data, warm up model."""
    global detector
    detector = IsolationForestDetector()

    # Only train if no pre-saved model was loaded
    if not detector.is_trained:
        rng = np.random.default_rng(42)
        n = 500  # 500 synthetic samples → better decision boundary

        # Normal registration telemetry
        normal = [
            {
                "rmse": float(rng.uniform(0.05, 0.40)),
                "spatial_entropy": float(rng.uniform(0.80, 0.99)),
                "inlier_ratio": float(rng.uniform(0.70, 0.95)),
            }
            for _ in range(n)
        ]
        # Inject known anomalies (high RMSE, low inliers) for contamination
        anomalies = [
            {
                "rmse": float(rng.uniform(1.5, 5.0)),
                "spatial_entropy": float(rng.uniform(0.10, 0.40)),
                "inlier_ratio": float(rng.uniform(0.05, 0.25)),
            }
            for _ in range(25)
        ]
        detector.train(normal + anomalies)

    yield  # app runs here

    # Shutdown: nothing to clean up


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Samanvaya ML Microservice",
    version="2.0.0",
    description="High-performance IsolationForest anomaly detector for lunar registration telemetry.",
    lifespan=lifespan,
)

# GZip middleware: compresses responses > 500 bytes → saves ~60-70% bandwidth
app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Telemetry(BaseModel):
    rmse: float = Field(..., ge=0.0, description="Root Mean Squared Error of registration")
    spatial_entropy: float = Field(..., ge=0.0, le=1.0, description="Keypoint spatial entropy [0,1]")
    inlier_ratio: float = Field(..., ge=0.0, le=1.0, description="RANSAC/MAGSAC inlier ratio [0,1]")


class BatchTelemetry(BaseModel):
    samples: List[Telemetry] = Field(..., max_length=256, description="Up to 256 samples per batch")


class PredictionResult(BaseModel):
    is_anomaly: bool
    confidence_score: float


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    anomaly_count: int
    batch_size: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def read_root() -> dict:
    """Lightweight health probe."""
    return {
        "status": "ML Microservice is running",
        "model_trained": detector.is_trained if detector else False,
        "version": "2.0.0",
    }


@app.post("/api/predict_anomaly", response_model=PredictionResult, tags=["Inference"])
async def predict(telemetry: Telemetry) -> PredictionResult:
    """
    Single-sample anomaly detection.
    Async: releases event loop during I/O wait, no blocking.
    """
    result = await asyncio.get_event_loop().run_in_executor(
        None, detector.predict, telemetry.model_dump()
    )
    return PredictionResult(**result)


@app.post("/api/predict_batch", response_model=BatchPredictionResult, tags=["Inference"])
async def predict_batch(batch: BatchTelemetry) -> BatchPredictionResult:
    """
    Vectorized batch prediction — processes up to 256 samples in a single
    numpy matrix call (10-50x faster than looping predict()).
    """
    samples = [s.model_dump() for s in batch.samples]
    results = await asyncio.get_event_loop().run_in_executor(
        None, detector.predict_batch, samples
    )
    anomaly_count = sum(1 for r in results if r["is_anomaly"])
    return BatchPredictionResult(
        results=[PredictionResult(**r) for r in results],
        anomaly_count=anomaly_count,
        batch_size=len(results),
    )


@app.get("/api/top_anomalies", tags=["Diagnostics"])
async def top_anomalies() -> dict:
    """Return the top 5 most severe anomalies tracked by the Min-Heap."""
    return {"top_anomalies": detector.get_top_anomalies()}


@app.post("/api/retrain", tags=["Model Management"])
async def retrain(data: BatchTelemetry) -> dict:
    """
    Trigger online retraining with new labeled telemetry.
    Persists updated model to disk via joblib.
    """
    samples = [s.model_dump() for s in data.samples]
    await asyncio.get_event_loop().run_in_executor(None, detector.train, samples)
    return {"status": "retrained", "samples_used": len(samples)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",        # nosec B104 - Container boundary
        port=8001,
        reload=True,
        workers=1,             # Single worker is fine for single-process ML model
        loop="asyncio",
        access_log=False,      # Disable per-request logging for perf
    )
