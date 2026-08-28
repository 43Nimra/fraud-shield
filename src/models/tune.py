import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import logging

from src.models.train import prepare_data, evaluate_model
from configs.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("FraudShield-XGBoost")


def train_single_run(params: dict, run_name: str, X_train, X_test, y_train, y_test) -> dict:
    """Ek run train karo aur MLflow mein log karo"""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params, eval_metric='auc', verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({
            "auc_roc":   metrics["auc_roc"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1_score":  metrics["f1_score"],
        })

        mlflow.xgboost.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

        logger.info(f"{run_name}: AUC={metrics['auc_roc']} | Recall={metrics['recall']} | F1={metrics['f1_score']}")
        return metrics, model, run_id


def threshold_tuning(model, X_test, y_test):
    """
    Threshold tuning — default 0.5 ki jagah best threshold find karo
    WHY: Same model, different threshold = different Recall/Precision tradeoff
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nThreshold Tuning Results:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Fraud Caught':>12}")
    print("-" * 57)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= threshold).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        caught = y_pred[y_test == 1].sum()

        print(f"{threshold:>10.1f} {p:>10.3f} {r:>10.3f} {f:>10.3f} {caught:>12,}")

        if f > best_f1:
            best_f1 = f
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.3f})")
    return best_threshold


def feature_importance(model, feature_names: list, top_n: int = 15):
    """Top N important features dikhao"""
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    print(f"\nTop {top_n} Important Features:")
    print("-" * 40)
    for _, row in feat_imp.iterrows():
        bar = "█" * int(row['importance'] * 200)
        print(f"{row['feature'][:25]:<25} {row['importance']:.4f} {bar}")

    return feat_imp


if __name__ == "__main__":
    logger.info("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(sample_size=10000)

    # Multiple parameter combinations try karo
    experiments = [
        {
            "name": "high-recall",
            "params": {
                "n_estimators": 200, "max_depth": 6,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "scale_pos_weight": 10,
                "random_state": 42, "n_jobs": -1,
            }
        },
        {
            "name": "deep-trees",
            "params": {
                "n_estimators": 300, "max_depth": 8,
                "learning_rate": 0.05, "subsample": 0.7,
                "colsample_bytree": 0.7, "scale_pos_weight": 5,
                "random_state": 42, "n_jobs": -1,
            }
        },
        {
            "name": "fast-learner",
            "params": {
                "n_estimators": 150, "max_depth": 4,
                "learning_rate": 0.2, "subsample": 0.9,
                "colsample_bytree": 0.9, "scale_pos_weight": 3,
                "random_state": 42, "n_jobs": -1,
            }
        },
    ]

    # Sab runs train karo
    results = []
    for exp in experiments:
        metrics, model, run_id = train_single_run(
            exp["params"], exp["name"],
            X_train, X_test, y_train, y_test
        )
        results.append({
            "name": exp["name"],
            "run_id": run_id,
            "model": model,
            "metrics": metrics
        })

    # Best model find karo — AUC pe
    best = max(results, key=lambda x: x["metrics"]["auc_roc"])

    print("\n" + "="*57)
    print("ALL RUNS COMPARISON")
    print("="*57)
    print(f"{'Run':<15} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 57)
    for r in results:
        m = r["metrics"]
        marker = " ← BEST" if r["name"] == best["name"] else ""
        print(f"{r['name']:<15} {m['auc_roc']:>8} {m['precision']:>10} {m['recall']:>8} {m['f1_score']:>8}{marker}")

    # Best model pe threshold tuning
    print(f"\nBest Model: {best['name']} (AUC={best['metrics']['auc_roc']})")
    best_threshold = threshold_tuning(best["model"], X_test, y_test)

    # Feature importance
    feat_imp = feature_importance(
        best["model"],
        list(feature_cols),
        top_n=15
    )

    print(f"\nMLflow UI: http://127.0.0.1:5000")
    print("Experiments tab pe jaao — 3 runs compare karo!")
