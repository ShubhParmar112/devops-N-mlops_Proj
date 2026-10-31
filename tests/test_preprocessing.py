"""Tests for data loading and the preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURES, TARGET, ZERO_AS_MISSING, ZeroToNaN, build_preprocessor, load_data, split_data,
)


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_dataset_shape_and_columns(df):
    assert df.shape == (768, 9)
    assert list(df.columns) == FEATURES + [TARGET]


def test_target_is_binary(df):
    assert set(df[TARGET].unique()) == {0, 1}


def test_zero_to_nan_only_touches_selected_columns():
    row = {f: 0 for f in FEATURES}
    out = ZeroToNaN(columns=ZERO_AS_MISSING).transform(pd.DataFrame([row]))
    for col in ZERO_AS_MISSING:
        assert np.isnan(out[col].iloc[0])
    # A zero is a valid value for these, so it must stay 0
    for col in ["Pregnancies", "DiabetesPedigreeFunction", "Age"]:
        assert out[col].iloc[0] == 0


def test_preprocessor_output_has_no_missing_values(df):
    X = df[FEATURES]
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()


def test_split_is_stratified(df):
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 614 and len(X_test) == 154
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_data(tmp_path / "nope.csv")
