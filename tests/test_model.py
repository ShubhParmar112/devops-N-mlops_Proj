"""Tests for the saved model and the prediction pipeline.

These double as a quality gate in CI: if a retrained model gets worse than
the minimum thresholds below, the pipeline fails before a Docker image is built.
"""
import pytest
from sklearn.metrics import recall_score, roc_auc_score

from src.predict import MODEL_PATH, load_metadata, load_model, predict
from src.preprocessing import FEATURES, load_data, split_data

MIN_ROC_AUC = 0.75
MIN_RECALL = 0.60

HIGH_RISK = {"Pregnancies": 8, "Glucose": 190, "BloodPressure": 80, "SkinThickness": 40,
             "Insulin": 200, "BMI": 42.0, "DiabetesPedigreeFunction": 1.2, "Age": 55}
LOW_RISK = {"Pregnancies": 0, "Glucose": 80, "BloodPressure": 65, "SkinThickness": 20,
            "Insulin": 60, "BMI": 21.0, "DiabetesPedigreeFunction": 0.15, "Age": 22}


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), "Model not found - run `python -m src.train` first"
    return load_model()


def test_model_is_full_pipeline(model):
    # Preprocessing must be saved together with the classifier
    assert list(model.named_steps) == ["preprocessor", "model"]
    assert hasattr(model, "predict_proba")


def test_predict_output_format(model):
    result = predict(HIGH_RISK)
    assert set(result) == {"prediction", "label", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in ("Diabetes Positive", "Diabetes Negative")


def test_high_and_low_risk_cases(model):
    assert predict(HIGH_RISK)["prediction"] == 1
    assert predict(LOW_RISK)["prediction"] == 0
    assert predict(HIGH_RISK)["probability"] > predict(LOW_RISK)["probability"]


def test_zero_values_are_handled(model):
    # Zeros meaning "not measured" must not crash the pipeline
    sample = {**LOW_RISK, "Insulin": 0, "SkinThickness": 0}
    assert predict(sample)["prediction"] in (0, 1)


def test_missing_feature_rejected(model):
    bad = {k: v for k, v in HIGH_RISK.items() if k != "Glucose"}
    with pytest.raises(ValueError, match="Missing"):
        predict(bad)


@pytest.mark.parametrize("feature,value", [("Glucose", -5), ("Age", 500), ("BMI", 200)])
def test_out_of_range_rejected(model, feature, value):
    with pytest.raises(ValueError, match="outside the valid range"):
        predict({**HIGH_RISK, feature: value})


def test_non_numeric_rejected(model):
    with pytest.raises(ValueError, match="must be a number"):
        predict({**HIGH_RISK, "BMI": "abc"})


def test_model_meets_quality_thresholds(model):
    _, X_test, _, y_test = split_data(load_data())
    proba = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) >= MIN_ROC_AUC
    assert recall_score(y_test, model.predict(X_test)) >= MIN_RECALL


def test_metadata_matches_model():
    meta = load_metadata()
    assert meta["features"] == FEATURES
    assert meta["model_name"] in ("LogisticRegression", "RandomForest")
