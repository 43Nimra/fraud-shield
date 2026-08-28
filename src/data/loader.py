import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Tuple
import logging

from configs.config import config

# Logging setup — print() nahi use karte production mein
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom Exception — WHY: generic Exception se better,
# caller ko pata chalta hai exact kya galat hua
class DataLoadError(Exception):
    """Raised when data loading fails"""
    pass


@dataclass
class DatasetInfo:
    """Dataset ka summary — load hone ke baad"""
    transaction_rows: int
    identity_rows: int
    fraud_count: int
    legitimate_count: int
    fraud_percentage: float

    def __str__(self) -> str:
        return (
            f"\nDataset Summary:"
            f"\n  Transactions : {self.transaction_rows:,}"
            f"\n  Identity rows: {self.identity_rows:,}"
            f"\n  Fraud        : {self.fraud_count:,} ({self.fraud_percentage:.2f}%)"
            f"\n  Legitimate   : {self.legitimate_count:,}"
        )


def load_transactions(chunksize: int = 10000) -> Generator[pd.DataFrame, None, None]:
    """
    Generator function — 652MB file ek saath RAM mein load nahi karte.
    Chunk by chunk yield karta hai.

    WHY Generator:
    - 652MB CSV = ~4GB RAM ek saath
    - Generator = sirf 10K rows at a time
    - Memory efficient
    """
    filepath = config.data.raw_data_dir / "train_transaction.csv"

    if not filepath.exists():
        raise DataLoadError(
            f"Transaction file nahi mili: {filepath}\n"
            f"Kaggle se download karo aur data/raw/ mein rakho"
        )

    logger.info(f"Loading transactions from {filepath}")

    try:
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            yield chunk
    except Exception as e:
        raise DataLoadError(f"File load karne mein error: {e}") from e


def load_full_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Poora dataset load karta hai — EDA ke liye.
    Production mein generator use karo.
    """
    transaction_path = config.data.raw_data_dir / "train_transaction.csv"
    identity_path = config.data.raw_data_dir / "train_identity.csv"

    if not transaction_path.exists():
        raise DataLoadError(f"File nahi mili: {transaction_path}")

    if not identity_path.exists():
        raise DataLoadError(f"File nahi mili: {identity_path}")

    logger.info("Loading full transaction dataset...")
    df_transaction = pd.read_csv(transaction_path)

    logger.info("Loading identity dataset...")
    df_identity = pd.read_csv(identity_path)

    logger.info("Dataset load complete!")
    return df_transaction, df_identity


def get_dataset_info(df_transaction: pd.DataFrame, df_identity: pd.DataFrame) -> DatasetInfo:
    """Dataset ka summary banata hai"""
    fraud_count = int(df_transaction['isFraud'].sum())
    total = len(df_transaction)

    return DatasetInfo(
        transaction_rows=total,
        identity_rows=len(df_identity),
        fraud_count=fraud_count,
        legitimate_count=total - fraud_count,
        fraud_percentage=(fraud_count / total) * 100
    )


if __name__ == "__main__":
    logger.info("Testing data loader...")

    # Test 1 — Generator test
    logger.info("Test 1: Generator — pehle 3 chunks")
    chunk_count = 0
    for chunk in load_transactions(chunksize=10000):
        chunk_count += 1
        logger.info(f"  Chunk {chunk_count}: {len(chunk):,} rows, columns: {len(chunk.columns)}")
        if chunk_count == 3:
            break

    # Test 2 — Full load
    logger.info("Test 2: Full dataset load")
    df_t, df_i = load_full_dataset()

    # Test 3 — Dataset info
    info = get_dataset_info(df_t, df_i)
    print(info)
