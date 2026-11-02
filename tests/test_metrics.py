"""Tests that predictions are recorded in the Prometheus metrics."""
import pytest
from prometheus_client import REGISTRY

from src.predict import predict

VALID = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 25,
         "Insulin": 80, "BMI": 30.0, "DiabetesPedigreeFunction": 0.4, "Age": 35}


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_successful_prediction_is_counted():
    before_req = value("diabetes_prediction_requests_total", source="test")
    before_lat = value("diabetes_prediction_latency_seconds_count", source="test")
    result = predict(VALID, source="test")
    outcome = "positive" if result["prediction"] == 1 else "negative"

    assert value("diabetes_prediction_requests_total", source="test") == before_req + 1
    assert value("diabetes_prediction_success_total", source="test", result=outcome) >= 1
    assert value("diabetes_prediction_latency_seconds_count", source="test") == before_lat + 1


def test_failed_prediction_is_counted():
    before = value("diabetes_prediction_failures_total", source="test", reason="validation_error")
    with pytest.raises(ValueError):
        predict({**VALID, "Glucose": -1}, source="test")
    assert value("diabetes_prediction_failures_total", source="test",
                 reason="validation_error") == before + 1
