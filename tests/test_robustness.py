"""Tests for the anti-overfitting admission statistics and zoo gates."""

import numpy as np
import pandas as pd
import pytest

from alphamcts.evaluation.metrics import AlphaMetrics
from alphamcts.evaluation.robustness import (
    asset_mismatch_ic,
    block_positive_fraction,
    newey_west_mean_test,
    reconstruction_r2,
    yearly_positive_fraction,
)
from alphamcts.zoo import AlphaZoo


def _metrics(**overrides) -> AlphaMetrics:
    base = dict(ic=0.03, ic_ir=0.4, rank_ic=0.03, rank_ir=0.4, turnover=0.5,
                max_corr=0.1, coverage=0.9)
    base.update(overrides)
    return AlphaMetrics(**base)


# ---------------------------------------------------------------------------
# Newey-West mean test
# ---------------------------------------------------------------------------

def test_newey_west_significant_for_persistent_signal():
    rng = np.random.default_rng(0)
    s = pd.Series(0.03 + 0.05 * rng.standard_normal(1000))
    t, p = newey_west_mean_test(s, lag=20)
    assert t > 3
    assert p < 0.01


def test_newey_west_insignificant_for_noise():
    rng = np.random.default_rng(1)
    s = pd.Series(0.002 + 0.1 * rng.standard_normal(300))
    _, p = newey_west_mean_test(s, lag=20)
    assert p > 0.05


def test_newey_west_penalizes_autocorrelation():
    """An autocorrelated series must look less significant under HAC than plain t."""
    rng = np.random.default_rng(2)
    eps = rng.standard_normal(800)
    ar = np.zeros(800)
    for i in range(1, 800):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    s = pd.Series(0.02 + 0.05 * ar)
    t_hac, _ = newey_west_mean_test(s, lag=20)
    t_iid, _ = newey_west_mean_test(s, lag=0)
    assert abs(t_hac) < abs(t_iid)


def test_newey_west_short_series_is_inconclusive():
    _, p = newey_west_mean_test(pd.Series([0.1, 0.2]), lag=5)
    assert p == 1.0


# ---------------------------------------------------------------------------
# Sub-period stability
# ---------------------------------------------------------------------------

def test_block_positive_fraction_stable_series():
    s = pd.Series(np.full(400, 0.01))
    assert block_positive_fraction(s) == 1.0


def test_block_positive_fraction_half_flipped():
    s = pd.Series(np.concatenate([np.full(200, 0.02), np.full(200, -0.02)]))
    assert block_positive_fraction(s, n_blocks=4) == pytest.approx(0.5)


def test_yearly_positive_fraction():
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="B")
    values = np.where(dates.year == 2019, -0.01, 0.01)
    s = pd.Series(values, index=dates)
    assert yearly_positive_fraction(s) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------

def test_asset_mismatch_collapses_genuine_signal():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    cols = [f"s{i}" for i in range(50)]
    signal = pd.DataFrame(rng.standard_normal((250, 50)), index=dates, columns=cols)
    fwd = signal * 0.5 + pd.DataFrame(
        rng.standard_normal((250, 50)), index=dates, columns=cols
    ) * 0.5
    control = asset_mismatch_ic(signal, fwd)
    assert control < 0.05  # mismatched signal must carry ~no information


# ---------------------------------------------------------------------------
# Reconstruction R^2
# ---------------------------------------------------------------------------

def _panel(rng, dates, cols):
    return pd.DataFrame(rng.standard_normal((len(dates), len(cols))), index=dates, columns=cols)


def test_reconstruction_r2_high_for_linear_combo():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = [f"s{i}" for i in range(40)]
    a, b = _panel(rng, dates, cols), _panel(rng, dates, cols)
    combo = 0.6 * a + 0.4 * b
    assert reconstruction_r2(combo, [a, b]) > 0.8


def test_reconstruction_r2_low_for_independent_factor():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = [f"s{i}" for i in range(40)]
    a, b, c = _panel(rng, dates, cols), _panel(rng, dates, cols), _panel(rng, dates, cols)
    assert reconstruction_r2(c, [a, b]) < 0.2


def test_reconstruction_r2_empty_pool_is_nan():
    rng = np.random.default_rng(6)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    a = _panel(rng, dates, ["s0", "s1", "s2"])
    assert reconstruction_r2(a, []) != reconstruction_r2(a, [])  # NaN


# ---------------------------------------------------------------------------
# Zoo admission gates
# ---------------------------------------------------------------------------

def test_zoo_rejects_insignificant_hac():
    zoo = AlphaZoo({"min_rank_ir": 0.1})
    ok, reasons = zoo.check_effective(_metrics(hac_p=0.2))
    assert not ok
    assert any("HAC" in r for r in reasons)


def test_zoo_rejects_unstable_blocks_and_years():
    zoo = AlphaZoo({"min_rank_ir": 0.1})
    ok, reasons = zoo.check_effective(_metrics(block_pos_frac=0.25, year_pos_frac=0.4))
    assert not ok
    assert any("time-block" in r for r in reasons)
    assert any("positive-year" in r for r in reasons)


def test_zoo_rejects_failed_negative_control_and_redundancy():
    zoo = AlphaZoo({"min_rank_ir": 0.1})
    ok, reasons = zoo.check_effective(_metrics(control_ic=0.05, recon_r2=0.95))
    assert not ok
    assert any("negative-control" in r for r in reasons)
    assert any("reconstruction" in r for r in reasons)


def test_zoo_accepts_robust_alpha_and_nan_stats_are_skipped():
    zoo = AlphaZoo({"min_rank_ir": 0.1})
    ok, _ = zoo.check_effective(
        _metrics(hac_p=0.001, block_pos_frac=1.0, year_pos_frac=1.0,
                 control_ic=0.005, recon_r2=0.1)
    )
    assert ok
    # NaN statistics (e.g. legacy zoo files) must not trip the gates
    ok_nan, _ = zoo.check_effective(_metrics())
    assert ok_nan
