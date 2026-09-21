"""Tests for cross-sectional industry + size neutralization."""

import numpy as np
import pandas as pd
import pytest

from alphamcts.data.neutralize import industry_demean, neutralize_panel


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    cols = [f"s{i}" for i in range(40)]
    industry = pd.DataFrame(
        [["IND_A"] * 20 + ["IND_B"] * 20] * len(dates), index=dates, columns=cols
    )
    log_mv = pd.DataFrame(
        10 + rng.standard_normal((len(dates), len(cols))), index=dates, columns=cols
    )
    noise = pd.DataFrame(rng.standard_normal((len(dates), len(cols))), index=dates, columns=cols)
    return dates, cols, industry, log_mv, noise


def test_industry_demean_removes_industry_means(setup):
    dates, cols, industry, _, noise = setup
    df = noise.copy()
    df.iloc[:, :20] += 5.0  # strong industry-A effect
    out = industry_demean(df, industry)
    for block in (out.iloc[:, :20], out.iloc[:, 20:]):
        assert block.mean(axis=1).abs().max() < 1e-10


def test_neutralize_removes_size_effect(setup):
    dates, cols, industry, log_mv, noise = setup
    df = 3.0 * log_mv + noise  # strong size effect
    out = neutralize_panel(df, industry=industry, log_mv=log_mv)
    # per-day correlation with log size must collapse
    corr = out.corrwith(log_mv, axis=1).abs()
    assert corr.mean() < 0.15


def test_neutralize_preserves_orthogonal_signal(setup):
    dates, cols, industry, log_mv, noise = setup
    df = noise  # signal independent of industry and size
    out = neutralize_panel(df, industry=industry, log_mv=log_mv)
    corr = out.corrwith(noise, axis=1)
    assert corr.mean() > 0.9  # bulk of the signal survives


def test_neutralize_handles_nan_and_unknown_industry(setup):
    dates, cols, industry, log_mv, noise = setup
    df = noise.copy()
    df.iloc[:, 0] = np.nan  # dead column
    industry = industry.copy()
    industry.iloc[:, 5] = ""  # unknown industry bucket
    out = neutralize_panel(df, industry=industry, log_mv=log_mv)
    assert out.iloc[:, 0].isna().all()      # NaN stays NaN
    assert out.iloc[:, 1:].notna().all().all()


def test_neutralize_without_exposures_is_identity(setup):
    _, _, _, _, noise = setup
    pd.testing.assert_frame_equal(neutralize_panel(noise), noise)
