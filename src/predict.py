"""Load the saved model and make predictions.

Usage:  python -m src.predict
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.preprocessing import FEATURES, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# Plausible input ranges, used to reject obviously wrong values
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120),
}


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.train")
    return joblib.load(path)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


def validate_input(data: dict) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    for f in FEATURES:
        try:
            value = float(data[f])
        except (TypeError, ValueError):
            raise ValueError(f"{f} must be a number, got {data[f]!r}")
        lo, hi = VALID_RANGES[f]
        if not lo <= value <= hi:
            raise ValueError(f"{f}={value} is outside the valid range [{lo}, {hi}]")
    return pd.DataFrame([{f: float(data[f]) for f in FEATURES}], columns=FEATURES)


def predict(data: dict) -> dict:
    """Return {'prediction': 0/1, 'label': str, 'probability': P(diabetes)}."""
    X = validate_input(data)
    model = load_model()
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return {
        "prediction": pred,
        "label": "Diabetes Positive" if pred == 1 else "Diabetes Negative",
        "probability": round(proba, 4),
    }


if __name__ == "__main__":
    samples = {
        "high-risk example": {"Pregnancies": 6, "Glucose": 168, "BloodPressure": 72, "SkinThickness": 35,
                              "Insulin": 0, "BMI": 38.0, "DiabetesPedigreeFunction": 0.627, "Age": 50},
        "low-risk example": {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
                             "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
    }
    for name, s in samples.items():
        print(f"{name}: {predict(s)}")
