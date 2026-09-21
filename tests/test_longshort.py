import numpy as np
import pandas as pd
import pytest

from alphamcts.evaluation.longshort import (
    annualized_sharpe,
    long_short_returns,
    long_short_sharpe,
    monotonic_fraction,
    monotonicity_score,
    quantile_labels,
    quantile_mean_returns,
)
from alphamcts.evaluation.metrics import AlphaMetrics
from alphamcts.zoo import AlphaZoo


def make_panel(n_days=120, n_stocks=25, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"s{i}" for i in range(n_stocks)]
    signal = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    noise = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=idx, columns=cols)
    # forward return strongly driven by the signal
    fwd = 0.02 * signal + 0.002 * noise
    return signal, fwd


def test_quantile_labels_cover_all_buckets():
    signal, _ = make_panel()
    q = quantile_labels(signal, n=5)
    assert set(np.unique(q.dropna(axis=1).to_numpy())) == {1.0, 2.0, 3.0, 4.0, 5.0}
    # roughly balanced buckets each day
    counts = (q == 1).sum(axis=1)
    assert (counts == 5).all()


def test_quantile_means_monotone_for_true_signal():
    signal, fwd = make_panel()
    qmeans = quantile_mean_returns(signal, fwd)
    assert monotonic_fraction(qmeans) == 1.0
    assert qmeans[5] - qmeans[1] > 0
    # reversed signal -> reversed buckets
    qmeans_rev = quantile_mean_returns(-signal, fwd)
    assert qmeans_rev[5] - qmeans_rev[1] < 0
    assert monotonic_fraction(qmeans_rev) == 0.0


def test_long_short_sharpe_sign_follows_signal():
    signal, fwd = make_panel()
    assert long_short_sharpe(signal, fwd) > 2.0
    assert long_short_sharpe(-signal, fwd) < -2.0


def test_long_short_returns_are_daily_series():
    signal, fwd = make_panel()
    ls = long_short_returns(signal, fwd)
    assert isinstance(ls, pd.Series)
    assert len(ls) == len(signal.index)
    assert ls.mean() > 0


def test_annualized_sharpe_basics():
    idx = pd.bdate_range("2020-01-01", periods=252)
    steady = pd.Series(0.001 + 0.0001 * np.random.default_rng(0).normal(size=252), index=idx)
    assert annualized_sharpe(steady) > 10
    assert np.isnan(annualized_sharpe(steady.iloc[:5]))  # too short
    assert np.isnan(annualized_sharpe(pd.Series(0.0, index=idx)))  # zero variance


def base_metrics(**kw) -> AlphaMetrics:
    m = AlphaMetrics(ic=0.03, ic_ir=0.4, rank_ic=0.03, rank_ir=0.4,
                     turnover=0.5, max_corr=0.1, coverage=0.9)
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_zoo_gates_on_longshort_stats():
    zoo = AlphaZoo({"min_rank_ic": 0.0, "min_rank_ir": 0.0,
                    "min_q_spread": 0.0, "min_ls_sharpe": 1.0, "min_q_monotonic": 0.75})
    ok, reasons = zoo.check_effective(
        base_metrics(q_spread=0.01, ls_sharpe=1.5, q_monotonic=1.0))
    assert ok, reasons

    ok, reasons = zoo.check_effective(
        base_metrics(q_spread=-0.001, ls_sharpe=1.5, q_monotonic=1.0))
    assert not ok and any("quantile spread" in r for r in reasons)

    ok, reasons = zoo.check_effective(
        base_metrics(q_spread=0.01, ls_sharpe=0.4, q_monotonic=1.0))
    assert not ok and any("long-short Sharpe" in r for r in reasons)

    ok, reasons = zoo.check_effective(
        base_metrics(q_spread=0.01, ls_sharpe=1.5, q_monotonic=0.5))
    assert not ok and any("monotonicity" in r for r in reasons)

    # NaN statistics are skipped (old zoos stay loadable)
    ok, _ = zoo.check_effective(base_metrics())
    assert ok


def test_monotonicity_score_composite():
    # wrong direction -> hard zero, regardless of shape/sharpe
    assert monotonicity_score(-0.001, 1.0, 3.0) == 0.0
    assert monotonicity_score(float("nan"), 1.0, 3.0) == 0.0
    # perfect shape + saturated sharpe -> full score
    assert monotonicity_score(0.01, 1.0, 2.0) == pytest.approx(1.0)
    assert monotonicity_score(0.01, 1.0, 5.0) == pytest.approx(1.0)  # sharpe capped
    # decomposition: 0.4 * mono + 0.6 * sharpe/2
    assert monotonicity_score(0.01, 0.5, 1.0) == pytest.approx(0.4 * 0.5 + 0.6 * 0.5)
    # negative sharpe contributes nothing but mono part remains
    assert monotonicity_score(0.01, 0.75, -1.0) == pytest.approx(0.3)
    # NaN components are treated as zero
    assert monotonicity_score(0.01, float("nan"), 1.0) == pytest.approx(0.3)


def test_evaluator_emits_monotonicity_dimension():
    from alphamcts.evaluation.evaluator import DIMENSIONS, DIMENSION_DESCRIPTIONS

    assert "monotonicity" in DIMENSIONS
    assert "monotonicity" in DIMENSION_DESCRIPTIONS


def test_zoo_exemplars_for_monotonicity_pick_high_ls_sharpe():
    from alphamcts.zoo import AlphaRecord

    zoo = AlphaZoo({"min_rank_ic": 0.0, "min_rank_ir": 0.0})
    signal, fwd = make_panel()
    for i, ls in enumerate([0.5, 2.5, 1.0]):
        rec = AlphaRecord(
            name=f"a{i}", description="", expression=f"expr{i}", formula=None,
            args={}, metrics=base_metrics(ls_sharpe=ls), scores={}, overall=0.5,
            overfit_reason="", values=signal + i,
        )
        zoo.records.append(rec)
    picks = zoo.exemplars("monotonicity", None, k=1, eta=0.0)
    assert picks and picks[0].expression == "expr1"  # the ls_sharpe=2.5 record
