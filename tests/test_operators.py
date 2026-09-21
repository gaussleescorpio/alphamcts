import numpy as np
import pandas as pd
import pytest

from alphamcts.expression.operators import OPERATORS, get_operator


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 4)), columns=list("ABCD"))


@pytest.fixture
def y():
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(60, 4)), columns=list("ABCD"))


def test_registry_covers_paper_table():
    expected = {
        "Neg", "Abs", "Square", "Inverse", "Sign", "Sin", "Cos", "Tanh", "Log",
        "Delay", "Diff", "Pct", "Ma", "Med", "Sum", "Std", "Max", "Min", "Rank",
        "Skew", "Kurt", "Vari", "Autocorr", "Zscore",
        "Add", "Sub", "Mul", "Div", "Greater", "Less", "Cov", "Corr",
        # extensions beyond the paper's Table 2
        "TsRegress",
    }
    assert expected == set(OPERATORS)


def test_alias_lookup():
    assert get_operator("mean") is OPERATORS["Ma"]
    assert get_operator("CORR") is OPERATORS["Corr"]
    assert get_operator("nope") is None


def test_delay_and_diff(x):
    d = OPERATORS["Delay"](x, 3)
    assert d.iloc[3, 0] == x.iloc[0, 0]
    assert d.iloc[:3].isna().all().all()
    diff = OPERATORS["Diff"](x, 3)
    assert np.isclose(diff.iloc[5, 1], x.iloc[5, 1] - x.iloc[2, 1])


def test_ma_of_constant():
    c = pd.DataFrame(np.ones((30, 2)) * 5.0)
    out = OPERATORS["Ma"](c, 10)
    assert np.allclose(out.iloc[10:], 5.0)
    assert out.iloc[:9].isna().all().all()


def test_zscore_of_constant_is_nan():
    c = pd.DataFrame(np.ones((30, 2)))
    out = OPERATORS["Zscore"](c, 10)
    assert out.iloc[15:].isna().all().all()  # zero std -> inf -> cleaned to NaN


def test_corr_of_identical_series(x):
    out = OPERATORS["Corr"](x, x, 20)
    valid = out.iloc[25:]
    assert np.allclose(valid, 1.0)


def test_autocorr_shape_and_range(x):
    out = OPERATORS["Autocorr"](x, 20, 2)
    assert out.shape == x.shape
    valid = out.stack().dropna()
    assert ((valid >= -1.000001) & (valid <= 1.000001)).all()


def test_rank_within_window(x):
    out = OPERATORS["Rank"](x, 10)
    valid = out.stack().dropna()
    assert ((valid > 0) & (valid <= 1)).all()


def test_div_by_zero_cleaned(x):
    z = x.copy()
    z.iloc[:] = 0.0
    out = OPERATORS["Div"](x, z)
    assert out.isna().all().all()


def test_log_nonpositive_is_nan():
    df = pd.DataFrame({"A": [-1.0, 0.0, np.e]})
    out = OPERATORS["Log"](df)
    assert np.isnan(out.iloc[0, 0]) and np.isnan(out.iloc[1, 0])
    assert np.isclose(out.iloc[2, 0], 1.0)


def test_greater_less(x, y):
    g = OPERATORS["Greater"](x, y)
    l = OPERATORS["Less"](x, y)
    assert set(g.stack().unique()) <= {0.0, 1.0}
    assert np.allclose(g + l, 1.0)  # no ties in continuous data


def test_ts_regress_recovers_exact_slope(x):
    y = 2.5 * x + 7.0  # exact linear relation -> slope 2.5 in every window
    out = OPERATORS["TsRegress"](x, y, 20)
    assert out.iloc[:19].isna().all().all()  # warm-up
    valid = out.iloc[20:]
    assert np.allclose(valid, 2.5)


def test_ts_regress_matches_cov_over_var(x, y):
    out = OPERATORS["TsRegress"](x, y, 15)
    manual = x.rolling(15, min_periods=15).cov(y) / x.rolling(15, min_periods=15).var()
    pd.testing.assert_frame_equal(out, manual.replace([np.inf, -np.inf], np.nan))


def test_ts_regress_alias_lookup():
    assert get_operator("ts_regress") is OPERATORS["TsRegress"]
    assert get_operator("beta") is OPERATORS["TsRegress"]


def test_ts_regress_constant_x_is_nan():
    const = pd.DataFrame(np.ones((40, 3)))
    rng = np.random.default_rng(5)
    y = pd.DataFrame(rng.normal(size=(40, 3)))
    out = OPERATORS["TsRegress"](const, y, 10)
    assert out.isna().all().all()  # zero variance in x -> undefined slope


def test_vari_matches_definition(x):
    v = OPERATORS["Vari"](x, 15)
    manual = x.rolling(15, min_periods=15).std() / x.rolling(15, min_periods=15).mean()
    pd.testing.assert_frame_equal(v, manual.replace([np.inf, -np.inf], np.nan))
