import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.data_prep import FEATURES, MedianZeroImputer, ZERO_AS_MISSING, load_pima, prepare_data


@pytest.fixture(scope="module")
def df():
    return load_pima(ROOT / "data" / "pima_diabetes.csv")


def test_dataset_shape_and_classes(df):
    assert df.shape == (768, 9)
    assert df["Outcome"].value_counts().to_dict() == {0: 500, 1: 268}


def test_imputation_replaces_zeros_with_train_median():
    X = pd.DataFrame({c: [0, 10, 20, 30] for c in FEATURES})
    imp = MedianZeroImputer().fit(X)
    out = imp.transform(X)
    for col in ZERO_AS_MISSING:
        assert imp.medians_[col] == 20.0            # median of non-zero values 10, 20, 30
        assert out[col].iloc[0] == 20.0
        assert (out[col] != 0).all()
    assert out["Pregnancies"].iloc[0] == 0          # zero pregnancies is valid, not imputed


def test_no_implausible_zeros_after_prep(df):
    d = prepare_data(df)
    raw_train = d.imputer.transform(df.loc[d.train_index, FEATURES])
    assert (raw_train[ZERO_AS_MISSING] != 0).all().all()


def test_imputer_and_scaler_fitted_on_train_only(df):
    d = prepare_data(df, seed=42)
    train_raw = df.loc[d.train_index, FEATURES]
    expected_medians = {c: train_raw[c].replace(0, np.nan).median() for c in ZERO_AS_MISSING}
    assert d.imputer.medians_ == pytest.approx(expected_medians)
    train_imp = d.imputer.transform(train_raw).to_numpy()
    np.testing.assert_allclose(d.scaler.mean_, train_imp.mean(axis=0))
    np.testing.assert_allclose(d.X_train.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(d.X_train.std(axis=0), 1, atol=1e-10)
    # the test set is transformed with train statistics, so it is NOT exactly standardised
    assert not np.allclose(d.X_test.mean(axis=0), 0, atol=1e-6)


def test_stratified_split(df):
    d = prepare_data(df, test_size=0.2, seed=42)
    assert len(d.y_train) == 614 and len(d.y_test) == 154
    assert set(d.train_index).isdisjoint(d.test_index)
    full_rate = df["Outcome"].mean()
    assert abs(d.y_train.mean() - full_rate) < 0.01
    assert abs(d.y_test.mean() - full_rate) < 0.01
    assert int(d.y_test.sum()) == 54


def test_split_is_reproducible(df):
    a, b = prepare_data(df, seed=42), prepare_data(df, seed=42)
    np.testing.assert_array_equal(a.test_index, b.test_index)
