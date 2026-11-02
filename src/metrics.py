"""Prometheus metrics for the Diabetes Prediction System.

The metrics are defined once per Python process (module-level objects), so
Streamlit re-running the app script on every interaction does not
re-register them. `start_metrics_server()` serves them at
http://<host>:METRICS_PORT/metrics for Prometheus to scrape.

Besides the custom metrics below, prometheus_client automatically exposes
process_cpu_seconds_total and process_resident_memory_bytes (on Linux, i.e.
inside Docker/Kubernetes), which the Grafana dashboard uses for CPU/memory.
"""
import os
import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

PREDICTION_REQUESTS = Counter(
    "diabetes_prediction_requests_total",
    "Total prediction requests received",
    ["source"],  # form | batch | csv
)
PREDICTION_SUCCESS = Counter(
    "diabetes_prediction_success_total",
    "Prediction requests that returned a result",
    ["source", "result"],  # result: positive | negative
)
PREDICTION_FAILURES = Counter(
    "diabetes_prediction_failures_total",
    "Prediction requests that failed",
    ["source", "reason"],  # reason: validation_error | model_error
)
PREDICTION_LATENCY = Histogram(
    "diabetes_prediction_latency_seconds",
    "Time taken to validate the input and run the model",
    ["source"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
APP_START_TIME = Gauge(
    "diabetes_app_start_time_seconds",
    "Unix time when the application process started",
)
MODEL_INFO = Info("diabetes_model", "Information about the deployed model")

APP_START_TIME.set(time.time())
_server_started = False


def set_model_info(metadata: dict) -> None:
    MODEL_INFO.info({
        "name": str(metadata.get("model_name", "unknown")),
        "registry_version": str(metadata.get("registered_version", "unknown")),
        "trained_at": str(metadata.get("trained_at", "unknown")),
    })


def start_metrics_server() -> bool:
    """Start the /metrics HTTP endpoint once per process. Returns True if running."""
    global _server_started
    if _server_started:
        return True
    if not METRICS_ENABLED:
        return False
    try:
        start_http_server(METRICS_PORT)
    except OSError:
        # Port already taken (e.g. a second app process on the same machine) - the app still works
        return False
    _server_started = True
    return True


@contextmanager
def track_prediction(source: str):
    """Count a request and time it. The caller reports success via the yielded dict."""
    PREDICTION_REQUESTS.labels(source=source).inc()
    outcome = {"result": None}
    start = time.perf_counter()
    try:
        yield outcome
    except ValueError:
        PREDICTION_FAILURES.labels(source=source, reason="validation_error").inc()
        raise
    except Exception:
        PREDICTION_FAILURES.labels(source=source, reason="model_error").inc()
        raise
    else:
        PREDICTION_SUCCESS.labels(source=source, result=outcome["result"] or "unknown").inc()
    finally:
        PREDICTION_LATENCY.labels(source=source).observe(time.perf_counter() - start)
