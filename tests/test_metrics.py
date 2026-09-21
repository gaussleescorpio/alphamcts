import numpy as np
import pandas as pd

from alphamcts.evaluation.metrics import (
    compute_alpha_metrics,
    cs_pearson,
    daily_turnover,
    factor_correlation,
    rank_ic_series,
    top_portfolio_weights,
)


def test_cs_pearson_perfect_and_inverse():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, 20)))
    assert np.allclose(cs_pearson(x, x).dropna(), 1.0)
    assert np.allclose(cs_pearson(x, -x).dropna(), -1.0)


def test_rank_ic_of_perfect_predictor():
    rng = np.random.default_rng(1)
    fwd = pd.DataFrame(rng.normal(size=(50, 20)))
    factor = fwd * 3.0 + 1.0  # monotone transform
    assert np.allclose(rank_ic_series(factor, fwd).dropna(), 1.0)


def test_top_weights_sum_to_one():
    rng = np.random.default_rng(2)
    v = pd.DataFrame(rng.normal(size=(30, 40)))
    w = top_portfolio_weights(v, 0.1)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 4).all()  # top 10% of 40


def test_turnover_bounds_and_constant_factor():
    rng = np.random.default_rng(3)
    v = pd.DataFrame(rng.normal(size=(100, 30)))
    t = daily_turnover(v, 0.1)
    assert 0.0 <= t <= 2.0

    persistent = pd.DataFrame(np.tile(np.arange(30.0), (100, 1)))
    assert daily_turnover(persistent, 0.1) == 0.0


def test_factor_correlation_self_and_independent():
    rng = np.random.default_rng(4)
    a = pd.DataFrame(rng.normal(size=(80, 50)))
    b = pd.DataFrame(rng.normal(size=(80, 50)))
    assert np.isclose(factor_correlation(a, a), 1.0)
    assert abs(factor_correlation(a, b)) < 0.1


def test_target_returns_override(panel):
    from alphamcts.data.base import MarketPanel

    external = panel.fields["close"] * 0.0 + 0.01
    p2 = MarketPanel(fields=panel.fields, target_returns=external)
    fwd = p2.forward_returns(20)
    assert np.allclose(fwd.fillna(0.01), 0.01)
    assert fwd.index.equals(panel.dates) and fwd.columns.equals(panel.instruments)


def test_compute_alpha_metrics_fields(panel):
    fwd = panel.forward_returns(10)
    factor = -panel.fields["close"].pct_change(5)
    m = compute_alpha_metrics(factor, fwd, top_frac=0.1, max_corr=0.2)
    assert m.rank_ic == m.rank_ic
    assert m.turnover == m.turnover
    assert m.max_corr == 0.2
    assert 0.0 < m.coverage <= 1.0
