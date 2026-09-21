import pytest

from alphamcts.data.base import FIELDS
from alphamcts.evaluation.evaluator import MultiDimEvaluator
from alphamcts.expression.formula import AlphaFormula, FormulaError
from alphamcts.zoo import AlphaRecord, AlphaZoo

WINDOW_RANGE = (2, 60)


def make_formula(steps, argsets, name="a"):
    return AlphaFormula.from_json(
        {"name": name, "description": "", "formula": steps, "arguments": argsets},
        FIELDS,
        WINDOW_RANGE,
    )


REVERSAL = [
    {"name": "Pct", "param": ["lookback"], "input": ["close"], "output": "mom"},
    {"name": "Neg", "param": [], "input": ["mom"], "output": "alpha"},
]


@pytest.fixture
def evaluator(panel):
    # The tiny synthetic cross-section (30 stocks) makes the asset-mismatch control
    # noisy, so it is relaxed here; the gate itself is covered in test_robustness.py.
    zoo = AlphaZoo({"min_rank_ic": 0.005, "min_rank_ir": 0.1, "max_control_ic": 0.10})
    return MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)


def scorer(expr, history):
    return 8.0, "simple and interpretable"


def test_reversal_alpha_has_positive_ic(evaluator):
    f = make_formula(REVERSAL, [{"lookback": 5}, {"lookback": 20}])
    result = evaluator.evaluate(f, scorer)
    # synthetic market embeds a 5-day reversal effect
    assert result.metrics.rank_ic > 0.01
    assert result.chosen_args["lookback"] in (5, 20)
    assert 0.0 <= result.overall <= 1.0
    assert result.scores["overfitting"] == 0.8


def test_best_argument_set_selected(evaluator):
    from alphamcts.evaluation.metrics import mean_and_ir, rank_ic_series

    f = make_formula(REVERSAL, [{"lookback": 5}, {"lookback": 55}])
    result = evaluator.evaluate(f, scorer)
    # the evaluator must pick the argument set with the higher standalone RankIC
    per_args = {
        args["lookback"]: mean_and_ir(
            rank_ic_series(evaluator.compute_values(f, args), evaluator.fwd_train)
        )[0]
        for args in f.arguments
    }
    best_lookback = max(per_args, key=per_args.get)
    assert result.chosen_args["lookback"] == best_lookback


def test_degenerate_formula_rejected(evaluator):
    f = make_formula(
        [{"name": "Greater", "param": [], "input": ["close", "close"], "output": "alpha"}],
        [{}],
    )
    with pytest.raises(FormulaError, match="constant"):
        evaluator.evaluate(f, scorer)


def test_zoo_admission_and_duplicates(evaluator):
    f = make_formula(REVERSAL, [{"lookback": 5}])
    result = evaluator.evaluate(f, scorer)
    zoo = evaluator.zoo
    record = AlphaRecord(
        name=f.name,
        description="",
        expression=result.expression,
        formula=f,
        args=result.chosen_args,
        metrics=result.metrics,
        values=result.values,
    )
    ok, reasons = zoo.add(record)
    assert ok, reasons
    ok, reasons = zoo.add(record)
    assert not ok and "duplicate expression" in reasons


def test_diversity_blocks_correlated_alpha(evaluator, panel):
    f1 = make_formula(REVERSAL, [{"lookback": 5}])
    r1 = evaluator.evaluate(f1, scorer)
    evaluator.zoo.add(
        AlphaRecord("a1", "", r1.expression, f1, r1.chosen_args, r1.metrics, values=r1.values)
    )
    # nearly identical alpha: reversal with lookback 5 scaled -> perfectly correlated
    f2 = make_formula(
        REVERSAL + [{"name": "Tanh", "param": [], "input": ["alpha"], "output": "alpha2"}],
        [{"lookback": 5}],
        name="b",
    )
    r2 = evaluator.evaluate(f2, scorer)
    assert r2.metrics.max_corr > 0.95
    ok, reasons = evaluator.zoo.check_effective(r2.metrics)
    assert not ok
    assert any("max corr" in r for r in reasons)


def test_relative_rank_and_exemplars(evaluator):
    f = make_formula(REVERSAL, [{"lookback": 5}])
    r = evaluator.evaluate(f, scorer)
    zoo = evaluator.zoo
    zoo.add(AlphaRecord("a1", "", r.expression, f, r.chosen_args, r.metrics, values=r.values))

    assert zoo.relative_rank("rank_ic", r.metrics.rank_ic + 1.0) == 0.0
    assert zoo.relative_rank("rank_ic", r.metrics.rank_ic - 1.0) == 1.0

    assert zoo.exemplars("turnover", r.values) == []
    assert len(zoo.exemplars("effectiveness", r.values, k=1)) == 1
    assert len(zoo.exemplars("diversity", r.values, k=1)) == 1


def test_zoo_save_load(tmp_path, evaluator):
    f = make_formula(REVERSAL, [{"lookback": 5}])
    r = evaluator.evaluate(f, scorer)
    zoo = evaluator.zoo
    zoo.add(AlphaRecord("a1", "d", r.expression, f, r.chosen_args, r.metrics, values=r.values))
    path = tmp_path / "zoo.json"
    zoo.save(path)

    loaded = AlphaZoo.load(path, FIELDS, WINDOW_RANGE)
    assert len(loaded) == 1
    rec = loaded.records[0]
    assert rec.expression == r.expression
    assert abs(rec.metrics.rank_ic - r.metrics.rank_ic) < 1e-12
