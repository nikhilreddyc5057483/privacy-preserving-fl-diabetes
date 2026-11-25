"""Data loading and preprocessing for the Pima Indians Diabetes Database.

Steps (all leakage-safe):
  1. load the CSV (768 rows, 8 features + Outcome)
  2. stratified 80/20 train/test split with a fixed seed
  3. zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI -> missing
  4. fill missing values with medians computed on the TRAINING rows only
  5. standardise features with a scaler fitted on the TRAINING rows only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima(csv_path: str | Path) -> pd.DataFrame:
    """Read the Pima CSV and check it has the expected shape and columns."""
    df = pd.read_csv(csv_path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[FEATURES + [TARGET]]


class MedianZeroImputer:
    """Treat implausible zeros as missing and fill them with training-set medians."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = list(columns) if columns is not None else list(ZERO_AS_MISSING)
        self.medians_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame) -> "MedianZeroImputer":
        for col in self.columns:
            values = X[col].replace(0, np.nan)
            self.medians_[col] = float(values.median())   # median of non-zero training values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].replace(0, np.nan).fillna(self.medians_[col])
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    imputer: MedianZeroImputer
    scaler: StandardScaler
    train_index: np.ndarray
    test_index: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
                 zero_cols: list[str] | None = None) -> PreparedData:
    """Split first, then fit imputer + scaler on the training rows only."""
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    imputer = MedianZeroImputer(zero_cols).fit(X_tr)          # medians from train only
    X_tr_imp, X_te_imp = imputer.transform(X_tr), imputer.transform(X_te)

    scaler = StandardScaler().fit(X_tr_imp.to_numpy())        # mean/std from train only
    return PreparedData(
        X_train=scaler.transform(X_tr_imp.to_numpy()),
        X_test=scaler.transform(X_te_imp.to_numpy()),
        y_train=y_tr, y_test=y_te, imputer=imputer, scaler=scaler,
        train_index=X_tr.index.to_numpy(), test_index=X_te.index.to_numpy())
