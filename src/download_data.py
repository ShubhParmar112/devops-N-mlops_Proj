"""Download the Pima Indians Diabetes Dataset into data/diabetes.csv.

Usage:  python src/download_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"

EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def download(force: bool = False) -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"Dataset already present: {DATA_PATH}")
    else:
        print(f"Downloading dataset from {URL}")
        urlretrieve(URL, DATA_PATH)

    # Verify the file is what we expect before anything else uses it
    df = pd.read_csv(DATA_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, f"Unexpected columns: {list(df.columns)}"
    assert df.shape == (768, 9), f"Unexpected shape: {df.shape}"
    print(f"OK: {df.shape[0]} rows, {df.shape[1]} columns")
    return DATA_PATH


if __name__ == "__main__":
    download()
