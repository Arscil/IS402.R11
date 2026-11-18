"""
Credit-card fraud prediction API (FastAPI + NumPy).

The model (trained with federated learning, exported by export_model.py) is a small Dense network,
so inference is plain NumPy: no TensorFlow in the image, small image, fast cold start.

Endpoints
    GET  /health       liveness check
    GET  /model-info   model name, threshold, test metrics, region
    POST /predict      one or many raw transactions -> fraud probability and label
Docs: /docs (Swagger UI)

Environment
    REGION           Azure region shown in responses (set by the deployment)
    CACHE_MODEL      "true" (default): load the model once at startup.
                     "false": reload model.json on every request (used to measure the gain of caching).
    ALLOWED_ORIGINS  comma-separated CORS origins for the web page (default "*")
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.json"
REGION = os.environ.get("REGION", "local")
CACHE_MODEL = os.environ.get("CACHE_MODEL", "true").lower() != "false"
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
MAX_BATCH = 1000


class FraudModel:
    def __init__(self, path: Path):
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.name = spec["model_name"]
        self.columns: List[str] = spec["feature_columns"]
        self.threshold = float(spec["threshold"])
        self.metrics = spec["metrics_test"]
        self.exported_at = spec["exported_at"]
        self.layers = [(np.asarray(l["W"]), np.asarray(l["b"]), l["activation"]) for l in spec["layers"]]
        sc = spec["scaler"]
        self.scale_idx = [self.columns.index(c) for c in spec["scaled_columns"]]
        if sc["type"] == "MinMaxScaler":
            self._scale = lambda x: x * np.asarray(sc["scale"]) + np.asarray(sc["min"])
        else:  # StandardScaler
            self._scale = lambda x: (x - np.asarray(sc["mean"])) / np.asarray(sc["scale"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64).copy()
        X[:, self.scale_idx] = self._scale(X[:, self.scale_idx])
        h = X
        for W, b, act in self.layers:
            h = h @ W + b
            h = np.maximum(h, 0.0) if act == "relu" else 1.0 / (1.0 + np.exp(-h))
        return h.ravel()


_cached = FraudModel(MODEL_PATH)


def get_model() -> FraudModel:
    return _cached if CACHE_MODEL else FraudModel(MODEL_PATH)


# Request schema: one field per feature, in the training column order (Time, V1..V28, Amount)
Transaction = create_model(
    "Transaction",
    **{c: (float, Field(..., description="Giây kể từ giao dịch đầu tiên" if c == "Time"
                        else "Số tiền giao dịch" if c == "Amount" else "Thành phần PCA ẩn danh"))
       for c in _cached.columns},
)


class PredictRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)  # type: ignore[valid-type]
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0,
                                       description="Bỏ trống để dùng ngưỡng tối ưu F1 của mô hình")


app = FastAPI(title="Fraud Detection API (Federated Learning)",
              description="Phát hiện giao dịch thẻ tín dụng gian lận, mô hình huấn luyện bằng học liên kết "
                          "(TensorFlow Federated, FedAvg) trên Azure.",
              version="1.0.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "fraud-detection-api", "region": REGION, "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "region": REGION, "started_at": STARTED_AT}


@app.get("/model-info")
def model_info() -> Dict[str, object]:
    m = get_model()
    return {"model": m.name, "threshold": m.threshold, "metrics_test": m.metrics,
            "feature_columns": m.columns, "exported_at": m.exported_at,
            "region": REGION, "cache_model": CACHE_MODEL, "started_at": STARTED_AT}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, object]:
    t0 = time.perf_counter()
    m = get_model()
    X = np.array([[getattr(t, c) for c in m.columns] for t in req.transactions], dtype=np.float64)
    if not np.isfinite(X).all():
        raise HTTPException(status_code=422, detail="Giá trị đầu vào phải là số hữu hạn")
    threshold = m.threshold if req.threshold is None else req.threshold
    prob = m.predict_proba(X)
    results = [{"fraud_probability": round(float(p), 6), "is_fraud": bool(p >= threshold),
                "label": "Gian lận" if p >= threshold else "Hợp lệ"} for p in prob]
    return {"model": m.name, "threshold": threshold, "region": REGION, "count": len(results),
            "fraud_count": sum(r["is_fraud"] for r in results),
            "server_time_ms": round((time.perf_counter() - t0) * 1000, 3), "results": results}
