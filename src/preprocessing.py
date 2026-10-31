"""Data loading and preprocessing for the Pima Indians Diabetes Dataset.

In this dataset a value of 0 for Glucose, BloodPressure, SkinThickness,
Insulin or BMI is physiologically impossible, so it really means "missing".
We convert those zeros to NaN and fill them with the median of the training
data. All of this lives inside a scikit-learn Pipeline so the exact same
steps are applied at training time and at prediction time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
# Columns where 0 means "not measured"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in the given columns so the imputer can fill them."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES).copy()
        cols = self.columns if self.columns is not None else ZERO_AS_MISSING
        X[cols] = X[cols].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/download_data.py"
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def split_data(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the same class ratio."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor() -> Pipeline:
    """Zero -> NaN -> median imputation -> standard scaling."""
    return Pipeline(steps=[
        ("zero_to_nan", ZeroToNaN(columns=ZERO_AS_MISSING)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
