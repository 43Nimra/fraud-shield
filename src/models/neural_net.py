import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import logging

from src.models.train import prepare_data, evaluate_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("FraudShield-PyTorch")


# ─────────────────────────────────────────
# 1. Custom Dataset
# ─────────────────────────────────────────
class FraudDataset(Dataset):
    """
    PyTorch Dataset — DataLoader ke saath kaam karta hai.
    WHY: DataLoader batches, shuffling, parallel loading
    automatically handle karta hai.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# 2. Neural Network Architecture
# ─────────────────────────────────────────
class FraudDetectorNN(nn.Module):
    """
    FraudShield Neural Network.

    Architecture:
    Input → Dense(256) → BatchNorm → ReLU → Dropout
          → Dense(128) → BatchNorm → ReLU → Dropout
          → Dense(64)  → BatchNorm → ReLU → Dropout
          → Dense(1)   → Sigmoid
          → Fraud Probability
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(FraudDetectorNN, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ─────────────────────────────────────────
# 3. Training Loop
# ─────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Ek epoch train karo — loss return karo"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()          # gradients reset
        predictions = model(X_batch)   # forward pass
        loss = criterion(predictions, y_batch)  # loss calculate
        loss.backward()                # backpropagation
        
        # Gradient clipping — exploding gradients rokta hai
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()               # weights update
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validation — loss aur AUC return karo"""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            probs = model(X_batch)
            loss = criterion(probs, y_batch)
            total_loss += loss.item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc


# ─────────────────────────────────────────
# 4. Main Training Function
# ─────────────────────────────────────────
def train_neural_network(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3
) -> dict:

    # Device — GPU hai toh GPU, warna CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Datasets + DataLoaders
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset  = FraudDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Model
    input_dim = X_train.shape[1]
    model = FraudDetectorNN(input_dim, dropout_rate).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + Optimizer
    # Fraud imbalanced — pos_weight se handle karo
    pos_weight = torch.tensor([y_train.sum() / (len(y_train) - y_train.sum())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    criterion_sigmoid = nn.BCELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # L2 regularization
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    with mlflow.start_run(run_name="pytorch-fraud-detector"):
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "input_dim": input_dim,
            "architecture": "256-128-64-1",
            "optimizer": "AdamW",
            "device": str(device)
        })

        best_auc = 0
        best_model_state = None

        logger.info(f"Training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion_sigmoid, device)
            val_loss, val_auc = evaluate_epoch(model, test_loader, criterion_sigmoid, device)

            scheduler.step(val_auc)

            # MLflow mein har epoch log karo
            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "val_loss":   round(val_loss, 4),
                "val_auc":    round(val_auc, 4),
            }, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")

        # Best model load karo
        model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_prob = model(X_test_tensor).cpu().numpy()

        y_pred = (y_prob >= 0.5).astype(int)

        final_metrics = {
            "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
        }

        mlflow.log_metrics(final_metrics)
        sample_input = torch.FloatTensor(X_test_np[:5]).to(device)
        mlflow.pytorch.log_model(model, "pytorch_model", input_example=sample_input.cpu().numpy())

        run_id = mlflow.active_run().info.run_id

        return final_metrics, model, run_id


if __name__ == "__main__":
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(sample_size=10000)

    # numpy arrays chahiye PyTorch ke liye
    X_train_np = X_train.values.astype(np.float32)
    X_test_np  = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    y_test_np  = y_test.values.astype(np.float32)

    metrics, model, run_id = train_neural_network(
        X_train_np, X_test_np,
        y_train_np, y_test_np,
        epochs=30,
        batch_size=256,
        learning_rate=0.001,
        dropout_rate=0.3
    )

    print("\n" + "="*50)
    print("FRAUDSHIELD — PyTorch Neural Network Results")
    print("="*50)
    print(f"AUC-ROC   : {metrics['auc_roc']}")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1 Score  : {metrics['f1_score']}")
    print(f"\nMLflow Run ID: {run_id}")
    print(f"MLflow UI: http://127.0.0.1:5000")
    print("="*50)

    # XGBoost vs PyTorch comparison
    print("\nXGBoost vs PyTorch:")
    print(f"XGBoost AUC  : 0.9169 (deep-trees run)")
    print(f"PyTorch AUC  : {metrics['auc_roc']}")
    print(f"Winner: {'PyTorch' if metrics['auc_roc'] > 0.9169 else 'XGBoost'}")
