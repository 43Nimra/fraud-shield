import sys
sys.path.insert(0, '.')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import logging
import time
from datetime import datetime

from src.data.database import get_connection
from configs.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(
    title="FraudShield API",
    description="Real-time transaction fraud detection",
    version="1.0.0"
)

# ─────────────────────────────────────────
# Pydantic Models — Request/Response
# ─────────────────────────────────────────
class TransactionRequest(BaseModel):
    """
    API request schema — Pydantic validate karta hai
    WHY: Wrong data type aaye → automatic 422 error
    """
    transaction_id: str = Field(..., description="Unique transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    product_type: str = Field(..., description="Product code: W/C/H/R/S")
    card_type: Optional[str] = Field(None, description="Card type")
    email_domain: Optional[str] = Field(None, description="Email domain")

    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 50000:
            raise ValueError("Amount suspiciously high")
        return round(v, 2)

    @validator('product_type')
    def valid_product_type(cls, v):
        allowed = ['W', 'C', 'H', 'R', 'S']
        if v.upper() not in allowed:
            raise ValueError(f"product_type must be one of {allowed}")
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN123456",
                "amount": 299.99,
                "product_type": "W",
                "card_type": "visa",
                "email_domain": "gmail.com"
            }
        }


class FraudPredictionResponse(BaseModel):
    """API response schema"""
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


# ─────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────
class ModelService:
    """
    WHY class: Model ek baar load karo — har request pe nahi
    Singleton pattern — ek hi instance
    """
    def __init__(self):
        self.model = None
        self.model_version = "not_loaded"
        self.threshold = 0.6  # Day 5 mein best threshold mila

    def load_model(self):
        """MLflow se best model load karo"""
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            client = mlflow.tracking.MlflowClient()

            # Best XGBoost run find karo
            runs = client.search_runs(
                experiment_ids=["1"],
                order_by=["metrics.auc_roc DESC"],
                max_results=1
            )

            if not runs:
                raise Exception("Koi trained model nahi mila")

            best_run = runs[0]
            run_id = best_run.info.run_id
            auc = best_run.data.metrics.get("auc_roc", 0)

            self.model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            self.model_version = f"xgboost-{run_id[:8]}"

            logger.info(f"Model loaded: {self.model_version} (AUC={auc})")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    def predict(self, features: np.ndarray) -> tuple:
        """Prediction karo — probability aur binary return"""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prob = float(self.model.predict_proba(features)[0][1])
        is_fraud = prob >= self.threshold
        return prob, is_fraud


# Singleton instance
model_service = ModelService()


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability < 0.3:   return "LOW"
    elif probability < 0.6: return "MEDIUM"
    else:                   return "HIGH"


def prepare_features(request: TransactionRequest) -> np.ndarray:
    """Request → feature array"""
    product_map = {'W': 0, 'C': 1, 'H': 2, 'R': 3, 'S': 4}

    features = {
        'TransactionAmt': request.amount,
        'amt_log': np.log1p(request.amount),
        'amt_zscore': (request.amount - 150) / 200,
        'is_high_value': 1 if request.amount > 500 else 0,
        'amt_bucket': min(4, int(request.amount / 200)),
        'ProductCD_encoded': product_map.get(request.product_type, -1),
    }

    # 400 features chahiye model ko — baaki 0 se fill
    feature_array = np.zeros(386)
    feature_array[0] = features['TransactionAmt']
    feature_array[1] = features['amt_log']
    feature_array[2] = features['amt_zscore']
    feature_array[3] = features['is_high_value']
    feature_array[4] = features['amt_bucket']
    feature_array[5] = features['ProductCD_encoded']

    return feature_array.reshape(1, -1)


def log_prediction_to_db(
    transaction_id: str,
    probability: float,
    is_fraud: bool,
    latency_ms: float
):
    """Prediction PostgreSQL mein log karo"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
                (transaction_id, fraud_probability, is_fraud_predicted, model_version, latency_ms)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            transaction_id,
            probability,
            1 if is_fraud else 0,
            model_service.model_version,
            latency_ms
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"DB log failed: {e}")


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Server start hone pe model load karo"""
    logger.info("FraudShield API starting...")
    success = model_service.load_model()
    if not success:
        logger.warning("Model load failed — /predict will return 503")


@app.get("/", response_model=dict)
async def root():
    return {
        "name": "FraudShield API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/predict", "/health", "/stats", "/docs"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — monitoring ke liye"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        total_predictions=total,
        version="1.0.0"
    )


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(request: TransactionRequest):
    """
    Main endpoint — transaction fraud probability return karo
    """
    start_time = time.perf_counter()

    try:
        # Features prepare karo
        features = prepare_features(request)

        # Prediction
        probability, is_fraud = model_service.predict(features)

        # Latency calculate karo
        latency_ms = (time.perf_counter() - start_time) * 1000

        # DB mein log karo
        log_prediction_to_db(
            request.transaction_id,
            probability,
            is_fraud,
            latency_ms
        )

        logger.info(
            f"TXN={request.transaction_id} | "
            f"Amount=${request.amount} | "
            f"Prob={probability:.3f} | "
            f"Fraud={is_fraud} | "
            f"Latency={latency_ms:.1f}ms"
        )

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
    """Prediction statistics — dashboard ke liye"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(is_fraud_predicted) as total_fraud,
            ROUND(AVG(fraud_probability)::numeric, 4) as avg_probability,
            ROUND(AVG(latency_ms)::numeric, 2) as avg_latency_ms
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
