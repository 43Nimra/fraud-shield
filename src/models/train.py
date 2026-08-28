import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import logging

from src.data.loader import load_full_dataset
from src.features.engineer import FraudFeatureEngineer, prepare_train_test
from configs.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("FraudShield-XGBoost")


def prepare_data(sample_size: int = 10000) -> tuple:
    """
    Data load, feature engineer, train/test split.
    sample_size: kitni rows use karni hain training ke liye
    """
    logger.info("Step 1: Data loading...")
    df_t, df_i = load_full_dataset()

    # Sample — pehle run pe chota sample (fast iteration)
    fraud = df_t[df_t['isFraud'] == 1].sample(
        n=min(sample_size // 2, len(df_t[df_t['isFraud'] == 1])),
        random_state=42
    )
    legit = df_t[df_t['isFraud'] == 0].sample(
        n=sample_size // 2,
        random_state=42
    )
    sample = pd.concat([fraud, legit]).reset_index(drop=True)
    logger.info(f"Sample: {len(sample):,} rows | Fraud: {fraud.shape[0]:,} | Legit: {legit.shape[0]:,}")

    logger.info("Step 2: Feature engineering...")
    engineer = FraudFeatureEngineer()
    df_featured = engineer.fit_transform(sample)

    logger.info("Step 3: Train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test(df_featured)

    logger.info("Step 4: SMOTE — imbalanced data fix...")
    # Numeric columns only
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_num = X_train[numeric_cols].fillna(0)
    X_test_num = X_test[numeric_cols].fillna(0)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_num, y_train)
    logger.info(f"After SMOTE — Train: {len(X_train_balanced):,} | Fraud: {y_train_balanced.sum():,}")

    return X_train_balanced, X_test_num, y_train_balanced, y_test, numeric_cols


def evaluate_model(model, X_test, y_test) -> dict:
    """Model evaluation — sahi metrics se"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["true_negatives"]  = int(cm[0][0])
    metrics["false_positives"] = int(cm[0][1])
    metrics["false_negatives"] = int(cm[1][0])
    metrics["true_positives"]  = int(cm[1][1])

    return metrics


def train_xgboost(params: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    XGBoost train karo + MLflow mein log karo
    """
    with mlflow.start_run(run_name=f"xgboost-baseline"):

        # Parameters log karo
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(**params, eval_metric='auc', verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Metrics MLflow mein log karo
        mlflow.log_metrics({
            "auc_roc":   metrics["auc_roc"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1_score":  metrics["f1_score"],
        })

        # Model save karo
        mlflow.xgboost.log_model(model, "xgboost_model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        return metrics, model, run_id


if __name__ == "__main__":
    # Data prepare karo
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(sample_size=10000)

    # XGBoost parameters
    params = {
        "n_estimators":     200,
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "random_state":     42,
        "n_jobs":           -1,
    }

    # Train + MLflow log
    metrics, model, run_id = train_xgboost(
        params, X_train, X_test, y_train, y_test
    )

    # Results print karo
    print("\n" + "="*50)
    print("FRAUDSHIELD — XGBoost Baseline Results")
    print("="*50)
    print(f"AUC-ROC   : {metrics['auc_roc']}")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1 Score  : {metrics['f1_score']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correct legit) : {metrics['true_negatives']:,}")
    print(f"  False Positives (wrong fraud)   : {metrics['false_positives']:,}")
    print(f"  False Negatives (missed fraud)  : {metrics['false_negatives']:,}")
    print(f"  True Positives  (caught fraud)  : {metrics['true_positives']:,}")
    print(f"\nMLflow Run ID: {run_id}")
    print(f"MLflow UI: http://127.0.0.1:5000")
    print("="*50)
