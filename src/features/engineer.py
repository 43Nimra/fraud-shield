import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Tuple

from src.utils.decorators import timer, validate_dataframe, log_step

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature engineering settings"""
    high_value_threshold: float = 500.0
    log_transform_cols: list = field(default_factory=lambda: ['TransactionAmt'])
    categorical_cols: list = field(default_factory=lambda: [
        'ProductCD', 'card4', 'card6', 'P_emaildomain'
    ])
    drop_cols: list = field(default_factory=lambda: [
        'TransactionID', 'TransactionDT'
    ])


class FraudFeatureEngineer:
    """
    WHY class: Feature engineering ek pipeline hai —
    fit() training data pe, transform() new data pe.
    Sklearn jaisa pattern — industry standard.

    fit()       → statistics calculate karo training data se
    transform() → wahi statistics apply karo new data pe
    fit_transform() → dono ek saath
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_stats = {}  # training se statistics store hongi
        self.is_fitted = False
        self.feature_names = []

    @log_step
    @timer
    @validate_dataframe(['TransactionAmt', 'isFraud'])
    def fit(self, df: pd.DataFrame) -> 'FraudFeatureEngineer':
        """
        Training data se statistics seekho.
        WHY fit/transform pattern:
        Test data pe KABHI statistics calculate nahi karte —
        data leakage hoti hai.
        """
        logger.info(f"Fitting on {len(df):,} rows")

        # Amount statistics — transform mein use hongi
        self.feature_stats['amt_mean'] = df['TransactionAmt'].mean()
        self.feature_stats['amt_std'] = df['TransactionAmt'].std()
        self.feature_stats['amt_median'] = df['TransactionAmt'].median()

        # Categorical — top categories store karo
        for col in self.config.categorical_cols:
            if col in df.columns:
                top_cats = df[col].value_counts().head(10).index.tolist()
                self.feature_stats[f'{col}_top'] = top_cats

        self.is_fitted = True
        logger.info(f"Stats calculated: {list(self.feature_stats.keys())}")
        return self

    @log_step
    @timer
    @validate_dataframe(['TransactionAmt'])
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features banao — fit se sikhi statistics use karo.
        """
        if not self.is_fitted:
            raise RuntimeError("Pehle fit() karo")

        df = df.copy()  # original data mat badlo

        # Feature 1 — Log transform (amount skewed hoti hai)
        # WHY: $10 aur $10,000 ka diff bahut bada hai —
        # log se normalize hota hai
        df['amt_log'] = np.log1p(df['TransactionAmt'])

        # Feature 2 — Amount zscore
        # WHY: mean se kitna door hai — outlier detection
        df['amt_zscore'] = (
            (df['TransactionAmt'] - self.feature_stats['amt_mean']) /
            (self.feature_stats['amt_std'] + 1e-8)
        )

        # Feature 3 — High value flag
        df['is_high_value'] = (
            df['TransactionAmt'] > self.config.high_value_threshold
        ).astype(int)

        # Feature 4 — Amount buckets
        # WHY: model ko ranges mein sochne mein help karta hai
        df['amt_bucket'] = pd.cut(
            df['TransactionAmt'],
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # Feature 5 — Categorical encoding
        # WHY: ML models numbers chahte hain, strings nahi
        for col in self.config.categorical_cols:
            if col in df.columns:
                top_cats = self.feature_stats.get(f'{col}_top', [])
                # Top categories ko number banao, baaki -1
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: top_cats.index(x) if x in top_cats else -1
                )

        # Feature 6 — Missing value flags
        # WHY: missingness itself ek signal hai fraud mein
        df['has_identity'] = df.get('id_01', pd.Series([np.nan]*len(df))).notna().astype(int)

        # Drop unnecessary columns
        cols_to_drop = [c for c in self.config.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Track feature names
        self.feature_names = [
            c for c in df.columns
            if c != 'isFraud'
        ]

        logger.info(f"Features created: {len(self.feature_names)}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience — fit aur transform ek saath"""
        return self.fit(df).transform(df)


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split — stratified (imbalanced data ke liye zaroori)
    WHY stratified: agar random split karo toh test mein
    fraud cases bahut kam ho sakte hain
    """
    from sklearn.model_selection import train_test_split

    # Target alag karo
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # Stratified split — fraud ratio preserve karta hai
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # YE IMPORTANT HAI
    )

    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    logger.info(f"Train fraud rate: {y_train.mean():.3f}")
    logger.info(f"Test fraud rate:  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data.loader import load_full_dataset

    logger.info("Loading dataset...")
    df_t, df_i = load_full_dataset()

    # Sample for testing
    sample = pd.concat([
        df_t[df_t['isFraud'] == 1].sample(n=2000, random_state=42),
        df_t[df_t['isFraud'] == 0].sample(n=2000, random_state=42)
    ]).reset_index(drop=True)

    logger.info(f"Sample shape: {sample.shape}")

    # Feature engineering
    engineer = FraudFeatureEngineer()
    df_featured = engineer.fit_transform(sample)

    logger.info(f"Features shape: {df_featured.shape}")
    logger.info(f"New features: amt_log, amt_zscore, is_high_value, amt_bucket")

    # Train test split
    X_train, X_test, y_train, y_test = prepare_train_test(df_featured)

    print(f"\nFeature Engineering Complete!")
    print(f"Total features: {len(engineer.feature_names)}")
    print(f"Train size: {len(X_train):,}")
    print(f"Test size:  {len(X_test):,}")
    print(f"Fraud rate preserved: {y_train.mean():.3f} train | {y_test.mean():.3f} test")
