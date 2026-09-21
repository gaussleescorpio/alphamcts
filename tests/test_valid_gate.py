"""Train/valid sign-consistency gate + validation-decay penalty on the overfitting dimension."""

import math

from alphamcts.data.base import FIELDS
from alphamcts.evaluation.evaluator import MultiDimEvaluator
from alphamcts.evaluation.metrics import AlphaMetrics
from alphamcts.expression.formula import AlphaFormula
from alphamcts.zoo import AlphaZoo

WINDOW_RANGE = (2, 60)

REVERSAL = [
    {"name": "Pct", "param": ["lookback"], "input": ["close"], "output": "mom"},
    {"name": "Neg", "param": [], "input": ["mom"], "output": "alpha"},
]


def make_formula(argsets):
    return AlphaFormula.from_json(
        {"name": "a", "description": "", "formula": REVERSAL, "arguments": argsets},
        FIELDS,
        WINDOW_RANGE,
    )


def metrics_with(rank_ic: float, rank_ic_valid: float) -> AlphaMetrics:
    return AlphaMetrics(
        ic=rank_ic, ic_ir=0.5, rank_ic=rank_ic, rank_ir=0.5,
        turnover=0.5, max_corr=0.0, coverage=1.0,
        rank_ic_valid=rank_ic_valid,
    )


# ---------------------------------------------------------------------------
# Zoo admission gate
# ---------------------------------------------------------------------------

def test_gate_rejects_valid_sign_flip():
    zoo = AlphaZoo({"min_rank_ic": 0.01, "min_rank_ir": 0.1, "require_valid_sign": True})
    ok, reasons = zoo.check_effective(metrics_with(0.03, -0.01))
    assert not ok
    assert any("flips sign" in r for r in reasons)


def test_gate_accepts_consistent_sign_and_skips_nan():
    zoo = AlphaZoo({"min_rank_ic": 0.01, "min_rank_ir": 0.1, "require_valid_sign": True})
    ok, _ = zoo.check_effective(metrics_with(0.03, 0.005))
    assert ok
    # NaN validation statistic (old zoos / no valid_range) must not block admission
    ok, _ = zoo.check_effective(metrics_with(0.03, float("nan")))
    assert ok


def test_gate_disabled_by_default():
    zoo = AlphaZoo({"min_rank_ic": 0.01, "min_rank_ir": 0.1})
    ok, _ = zoo.check_effective(metrics_with(0.03, -0.01))
    assert ok


def test_gate_persists_in_criteria(tmp_path):
    zoo = AlphaZoo({"require_valid_sign": True})
    zoo.save(tmp_path / "zoo.json")
    loaded = AlphaZoo.load(tmp_path / "zoo.json", FIELDS, WINDOW_RANGE)
    assert loaded.require_valid_sign is True


# ---------------------------------------------------------------------------
# Overfitting-dimension decay penalty
# ---------------------------------------------------------------------------

def test_generalization_multiplier_buckets():
    m = MultiDimEvaluator._generalization_multiplier
    assert m(metrics_with(0.03, -0.01)) == (0.2, m(metrics_with(0.03, -0.01))[1])  # sign flip
    assert m(metrics_with(0.03, 0.003))[0] == 0.6    # ratio 0.1: severe decay
    assert m(metrics_with(0.03, 0.015))[0] == 0.85   # ratio 0.5: moderate decay
    assert m(metrics_with(0.03, 0.025))[0] == 1.0    # ratio ~0.83: fine
    assert m(metrics_with(0.03, float("nan"))) == (1.0, "")  # no validation info


def test_evaluator_attaches_validation_and_penalizes(panel):
    zoo = AlphaZoo({"min_rank_ic": 0.005, "min_rank_ir": 0.1, "max_control_ic": 0.10})
    dates = panel.dates
    train_range = (str(dates[0].date()), str(dates[199].date()))
    valid_range = (str(dates[200].date()), str(dates[-1].date()))
    ev = MultiDimEvaluator(
        panel, horizon=10, zoo=zoo, top_frac=0.1,
        train_range=train_range, valid_range=valid_range,
    )
    result = ev.evaluate(make_formula([{"lookback": 5}]), lambda e, h: (8.0, "ok"))
    assert not math.isnan(result.metrics.rank_ic_valid)
    mult, _ = MultiDimEvaluator._generalization_multiplier(result.metrics)
    assert result.scores["overfitting"] == 0.8 * mult


def test_evaluator_without_valid_range_leaves_nan(panel):
    zoo = AlphaZoo({"min_rank_ic": 0.005, "min_rank_ir": 0.1, "max_control_ic": 0.10})
    ev = MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)
    result = ev.evaluate(make_formula([{"lookback": 5}]), lambda e, h: (8.0, "ok"))
    assert math.isnan(result.metrics.rank_ic_valid)
    assert result.scores["overfitting"] == 0.8
