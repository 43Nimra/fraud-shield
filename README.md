# 🛡️ FraudShield — Real-Time Transaction Fraud Detection API

> Detects fraudulent transactions in **<4ms** using XGBoost trained on 590K real transactions. Deployed on GCP Cloud Run.

[![Live API](https://img.shields.io/badge/API-Live%20on%20GCP-success)](https://fraud-shield-590785312612.us-central1.run.app/docs)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.92-blue)]()
[![Recall](https://img.shields.io/badge/Recall-92%25-green)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)

---

## 🚀 Live Demo

```bash
curl -X POST https://fraud-shield-590785312612.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TXN001", "amount": 4500.00, "product_type": "C"}'
```

**Response in <4ms:**
```json
{
  "transaction_id": "TXN001",
  "is_fraud": true,
  "fraud_probability": 0.8392,
  "risk_level": "HIGH",
  "latency_ms": 3.39
}
```

---

## 📊 The Problem

Financial fraud costs the global economy **$5.13 trillion annually**.
Traditional rule-based systems miss **40%+** of fraud cases.

**FraudShield** uses ML to detect fraud patterns that rules cannot.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.917** |
| Recall | **92.2%** — catches 922/1000 fraud cases |
| Precision | **75.6%** |
| F1 Score | **0.831** |
| Latency | **<4ms** per prediction |
| Dataset | **590,540** real transactions (IEEE-CIS) |

**Key Finding:** XGBoost outperformed PyTorch Neural Network (AUC 0.917 vs 0.844) on tabular fraud data — confirming that tree-based models excel at structured financial data.

---


## 🏗️ Architecture

**Request Flow:**

`Client` → `FastAPI (GCP Cloud Run)` → `Feature Engineering` → `XGBoost Model (AUC 0.917)` → `Response (<4ms)` → `PostgreSQL (log)`

---

## 🔬 ML Pipeline
IEEE-CIS Dataset (590,540 transactions)
│
▼
Exploratory Analysis (SQL + PostgreSQL)
• Product C: 77.59% fraud rate
• High value (>$500): distinct pattern
• Identity missing = fraud signal
│
▼
Feature Engineering (sklearn-style pipeline)
• Log transform — skewed amount normalize
• Z-score — outlier detection
• Amount bucketing — range patterns
• Missing value flags — fraud signal
• Categorical encoding
│
▼
SMOTE — Imbalanced Data Fix
• Original: 3.5% fraud / 96.5% legit
• After: balanced training set
│
▼
Model Training + MLflow Tracking
• XGBoost baseline → AUC 0.912
• XGBoost high-recall → AUC 0.912, Recall 0.949
• XGBoost deep-trees → AUC 0.917 ✓ BEST
• PyTorch Neural Net → AUC 0.844
│
▼
Threshold Tuning
• Default 0.5 → F1: 0.831
• Optimal 0.6 → F1: 0.847 ✓
│
▼
FastAPI → Docker → GCP Cloud Run

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, PyTorch, Scikit-learn |
| API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Cloud | GCP Cloud Run |
| Language | Python 3.13 |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Real-time fraud detection |
| `/health` | GET | Service health check |
| `/stats` | GET | Prediction statistics |
| `/docs` | GET | Interactive API documentation |

**Full API Docs:** https://fraud-shield-590785312612.us-central1.run.app/docs

---

## 🔑 Key Learnings

1. **XGBoost > Neural Networks** on tabular fraud data — empirically proven
2. **Imbalanced data** (3.5% fraud) requires SMOTE + class weights
3. **Threshold tuning** (0.5→0.6) improved F1 from 0.831 to 0.847
4. **Feature V70** most important — anonymous transaction velocity feature
5. **Product C** has 77.59% fraud rate — strongest categorical signal

---


## 📁 Project Structure

| Path | Description |
|------|-------------|
| `src/data/loader.py` | Generator-based data loading (590K rows) |
| `src/data/database.py` | PostgreSQL connection |
| `src/data/schema.py` | FraudTransaction dataclass + validation |
| `src/features/engineer.py` | sklearn-style feature pipeline |
| `src/models/train.py` | XGBoost training + MLflow tracking |
| `src/models/tune.py` | Hyperparameter tuning (3 experiments) |
| `src/models/neural_net.py` | PyTorch neural network |
| `src/api/endpoints.py` | FastAPI — /predict, /health, /stats |
| `src/utils/decorators.py` | @timer, @validate_dataframe, @log_step |
| `configs/config.py` | Dataclass-based config management |
| `Dockerfile` | Container definition (linux/amd64) |

---

## 👩‍💻 Author

**Nimra** — CS Final Year Student → ML Engineer

Building production ML systems while completing CS degree.

[![GitHub](https://img.shields.io/badge/GitHub-43Nimra-black)](https://github.com/43Nimra)
