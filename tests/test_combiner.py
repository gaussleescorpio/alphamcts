import numpy as np
import pytest

from alphamcts.data.base import FIELDS
from alphamcts.evaluation.evaluator import MultiDimEvaluator
from alphamcts.expression.formula import AlphaFormula
from alphamcts.model import CombinerPipeline
from alphamcts.zoo import AlphaRecord, AlphaZoo

WINDOW_RANGE = (2, 60)

FORMULAS = [
    (
        "reversal",
        [
            {"name": "Pct", "param": ["w"], "input": ["close"], "output": "r"},
            {"name": "Neg", "param": [], "input": ["r"], "output": "a"},
        ],
        {"w": 5},
    ),
    (
        "pressure",
        [
            {"name": "Sub", "param": [], "input": ["close", "vwap"], "output": "p"},
            {"name": "Ma", "param": ["w1"], "input": ["p"], "output": "t"},
            {"name": "Zscore", "param": ["w2"], "input": ["t"], "output": "a"},
        ],
        {"w1": 10, "w2": 20},
    ),
    (
        "pv_corr",
        [
            {"name": "Corr", "param": ["w"], "input": ["close", "volume"], "output": "c"},
            {"name": "Neg", "param": [], "input": ["c"], "output": "a"},
        ],
        {"w": 20},
    ),
]


@pytest.fixture
def records(panel):
    zoo = AlphaZoo()
    evaluator = MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)
    out = []
    for name, steps, args in FORMULAS:
        f = AlphaFormula.from_json(
            {"name": name, "description": "", "formula": steps, "arguments": [args]},
            FIELDS,
            WINDOW_RANGE,
        )
        r = evaluator.evaluate(f, lambda e, h: (8.0, "ok"))
        out.append(
            AlphaRecord(name, "", r.expression, f, r.chosen_args, r.metrics, values=r.values)
        )
    return out


def test_lightgbm_combiner_and_backtest(panel, records):
    pipe = CombinerPipeline(panel, horizon=10, train_ratio=0.7)
    report = pipe.run_model("lightgbm", records)
    assert report.n_alphas == 3
    assert report.ic == report.ic and report.rank_ic == report.rank_ic
    # reversal is genuinely predictive in the synthetic market
    assert report.rank_ic > 0.0
    assert report.aer == report.aer and report.ir == report.ir
    assert len(report.daily_excess) > 30


def test_mlp_combiner_runs(panel, records):
    pipe = CombinerPipeline(
        panel,
        horizon=10,
        train_ratio=0.7,
        combiner_cfg={"mlp": {"hidden": [32, 16], "max_epochs": 10, "patience": 3}},
    )
    report = pipe.run_model("mlp", records)
    assert report.ic == report.ic
    assert np.isfinite(report.aer)


def test_equal_weight_combiner(panel, records):
    pipe = CombinerPipeline(panel, horizon=10, train_ratio=0.7)
    report = pipe.run_model("equal", records)
    assert report.ic == report.ic and np.isfinite(report.aer)


def test_ewma_combiner_runs_and_weights_are_valid(panel, records):
    pipe = CombinerPipeline(
        panel, horizon=10, train_ratio=0.7,
        combiner_cfg={"ewma": {"halflife": 20, "min_periods": 20}},
    )
    model, features = pipe.fit("ewma", records)
    w = model.weights
    assert ((w >= 0) | w.isna()).all().all()
    sums = w.sum(axis=1).dropna()
    assert np.allclose(sums, 1.0, atol=1e-9)  # normalized (incl. equal-weight fallback)
    report = pipe.evaluate_period(model, features, pipe.test_dates, "ewma", len(records))
    assert report.rank_ic == report.rank_ic


def test_ewma_weights_have_no_lookahead(panel, records):
    """Weights up to date t must not change when future returns are altered."""
    import pandas as pd

    pipe = CombinerPipeline(
        panel, horizon=10, train_ratio=0.7,
        combiner_cfg={"ewma": {"halflife": 20, "min_periods": 20}},
    )
    features = pipe.compute_features(records)
    w_base = pipe.combine_ewma(features).weights

    cut = pipe.panel.dates[len(pipe.panel.dates) // 2]
    fwd_tampered = pipe.fwd.copy()
    fwd_tampered.loc[fwd_tampered.index > cut] = -fwd_tampered.loc[fwd_tampered.index > cut]
    pipe.fwd = fwd_tampered
    w_tampered = pipe.combine_ewma(features).weights

    lag = pipe.horizon + 1
    safe = w_base.index[w_base.index <= cut][:-lag]
    pd.testing.assert_frame_equal(w_base.loc[safe], w_tampered.loc[safe])


def test_ewma_downweights_dead_factor(panel, records):
    """A factor whose values are pure noise should end up with low average weight."""
    import pandas as pd

    rng = np.random.default_rng(0)
    pipe = CombinerPipeline(
        panel, horizon=10, train_ratio=0.7,
        combiner_cfg={"ewma": {"halflife": 20, "min_periods": 20}},
    )
    features = pipe.compute_features(records)
    noise = pd.DataFrame(
        rng.standard_normal(features[0].shape), index=features[0].index, columns=features[0].columns
    )
    from alphamcts.model.combiner import _rank_normalize

    features.append(_rank_normalize(pipe.panel.apply_universe(noise)))
    w = pipe.combine_ewma(features).weights
    mean_w = w.dropna().mean()
    assert mean_w.iloc[-1] < 1.0 / len(features)  # noise factor below uniform share
    assert mean_w.iloc[0] > mean_w.iloc[-1]       # genuine reversal factor above it


def test_periodic_backtest_runs_and_costs_less_than_daily(panel, records):
    pipe = CombinerPipeline(panel, horizon=10, train_ratio=0.7, backtest_cfg={"cost": 0.0015})
    model, features = pipe.fit("equal", records)
    pred = pipe.predict_panel(model, features, pipe.test_dates)

    aer_daily_free, _, _ = pipe.backtest_topk_dropn(pred)
    aer_periodic_free = None
    pipe.bt_cfg = {"cost": 0.0}
    aer_daily_0, _, s0 = pipe.backtest_topk_dropn(pred)
    aer_periodic_0, _, sp = pipe.backtest_periodic(pred)
    assert np.isfinite(aer_periodic_0) and len(sp) > 10

    pipe.bt_cfg = {"cost": 0.0015}
    aer_daily_c, _, _ = pipe.backtest_topk_dropn(pred)
    aer_periodic_c, _, _ = pipe.backtest_periodic(pred)
    # both pay costs...
    assert aer_periodic_c < aer_periodic_0
    # ...but the periodic schedule pays less cost drag than daily churn
    assert (aer_periodic_0 - aer_periodic_c) < (aer_daily_0 - aer_daily_c)


def test_backtest_costs_reduce_returns(panel, records):
    pipe_free = CombinerPipeline(panel, horizon=10, train_ratio=0.7, backtest_cfg={"cost": 0.0})
    pipe_cost = CombinerPipeline(panel, horizon=10, train_ratio=0.7, backtest_cfg={"cost": 0.01})
    features_free = pipe_free.compute_features(records)
    x, y, _ = pipe_free._dataset(features_free, pipe_free.train_dates)
    model = pipe_free._fit_lightgbm(x, y)
    pred = pipe_free.predict_panel(model, features_free, pipe_free.test_dates)
    aer_free, _, _ = pipe_free.backtest_topk_dropn(pred)
    aer_cost, _, _ = pipe_cost.backtest_topk_dropn(pred)
    assert aer_cost < aer_free
