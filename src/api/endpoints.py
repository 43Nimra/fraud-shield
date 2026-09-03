import sys
sys.path.insert(0, '.')

import os
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.data.database import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FraudShield API",
    description="Real-time transaction fraud detection",
    version="1.0.0"
)

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    product_type: str = Field(..., description="Product code: W/C/H/R/S")
    card_type: Optional[str] = Field(None, description="Card type")
    email_domain: Optional[str] = Field(None, description="Email domain")

    @field_validator('amount')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 50000:
            raise ValueError("Amount suspiciously high")
        return round(v, 2)

    @field_validator('product_type')
    @classmethod
    def valid_product_type(cls, v):
        allowed = ['W', 'C', 'H', 'R', 'S']
        if v.upper() not in allowed:
            raise ValueError(f"product_type must be one of {allowed}")
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN123456",
                "amount": 299.99,
                "product_type": "W",
                "card_type": "visa",
                "email_domain": "gmail.com"
            }
        }
    }


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_predictions: int
    version: str


class ModelService:
    def __init__(self):
        self.model = None
        self.model_version = "not_loaded"
        self.threshold = 0.6

    def load_model(self):
        try:
            model_path = os.getenv("MODEL_PATH", "src/api/fraud_model.ubj")
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            self.model_version = "xgboost-v1"
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    def predict(self, features: np.ndarray) -> tuple:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        prob = float(self.model.predict_proba(features)[0][1])
        is_fraud = prob >= self.threshold
        return prob, is_fraud


model_service = ModelService()


def get_risk_level(probability: float) -> str:
    if probability < 0.3:   return "LOW"
    elif probability < 0.6: return "MEDIUM"
    else:                   return "HIGH"


def prepare_features(request: TransactionRequest) -> np.ndarray:
    product_map = {'W': 0, 'C': 1, 'H': 2, 'R': 3, 'S': 4}
    feature_array = np.zeros(386)
    feature_array[0] = request.amount
    feature_array[1] = np.log1p(request.amount)
    feature_array[2] = (request.amount - 150) / 200
    feature_array[3] = 1 if request.amount > 500 else 0
    feature_array[4] = min(4, int(request.amount / 200))
    feature_array[5] = product_map.get(request.product_type, -1)
    return feature_array.reshape(1, -1)


def log_prediction_to_db(transaction_id, probability, is_fraud, latency_ms):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
                (transaction_id, fraud_probability, is_fraud_predicted, model_version, latency_ms)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (transaction_id, probability, 1 if is_fraud else 0, model_service.model_version, latency_ms))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"DB log failed: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info("FraudShield API starting...")
    success = model_service.load_model()
    if not success:
        logger.warning("Model load failed — /predict will return 503")


@app.get("/")
async def root():
    return {"name": "FraudShield API", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except:
        total = 0
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        total_predictions=total,
        version="1.0.0"
    )


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(request: TransactionRequest):
    start_time = time.perf_counter()
    try:
        features = prepare_features(request)
        probability, is_fraud = model_service.predict(features)
        latency_ms = (time.perf_counter() - start_time) * 1000
        log_prediction_to_db(request.transaction_id, probability, is_fraud, latency_ms)
        logger.info(f"TXN={request.transaction_id} | Amount=${request.amount} | Prob={probability:.3f} | Fraud={is_fraud} | Latency={latency_ms:.1f}ms")
        return FraudPredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(probability, 4),
            risk_level=get_risk_level(probability),
            model_version=model_service.model_version,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*), SUM(is_fraud_predicted),
               ROUND(AVG(fraud_probability)::numeric, 4),
               ROUND(AVG(latency_ms)::numeric, 2)
        FROM predictions
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return {
        "total_predictions": row[0],
        "total_fraud_flagged": row[1],
        "avg_fraud_probability": float(row[2]) if row[2] else 0,
        "avg_latency_ms": float(row[3]) if row[3] else 0,
        "model_version": model_service.model_version,
        "fraud_rate": round(row[1] / row[0] * 100, 2) if row[0] > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
