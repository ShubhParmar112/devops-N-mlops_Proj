"""Smoke test: run the Streamlit app headlessly and click Predict."""
from streamlit.testing.v1 import AppTest

APP = "app/streamlit_app.py"


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Diabetes Prediction System" in at.title[0].value
    assert len(at.number_input) == 8


def test_predict_button_shows_result():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.number_input[1].set_value(190)  # Glucose
    at.button[0].click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Diabetes Positive" in page or "Diabetes Negative" in page
