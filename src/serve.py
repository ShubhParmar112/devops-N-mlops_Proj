"""Container entrypoint: start the Prometheus /metrics server, then run Streamlit.

Streamlit only executes the app script when a browser session connects, so
starting the metrics server from the app alone would leave /metrics down
until someone opens the page. Starting it here, in the same process, makes
it available as soon as the container starts.

Usage:  python -m src.serve        (same as `streamlit run app/streamlit_app.py`)
"""
from streamlit.web import bootstrap

from src.metrics import METRICS_PORT, set_model_info, start_metrics_server
from src.predict import load_metadata
from src.preprocessing import PROJECT_ROOT

APP_PATH = str(PROJECT_ROOT / "app" / "streamlit_app.py")
FLAGS = {"server.port": 8501, "server.address": "0.0.0.0", "server.headless": True}


def main():
    set_model_info(load_metadata())
    if start_metrics_server():
        print(f"Prometheus metrics available on :{METRICS_PORT}/metrics", flush=True)
    bootstrap.load_config_options(flag_options=FLAGS)
    bootstrap.run(APP_PATH, False, [], FLAGS)


if __name__ == "__main__":
    main()
