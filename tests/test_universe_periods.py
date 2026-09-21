import numpy as np
import pandas as pd
import pytest

from alphamcts.data.base import FIELDS, MarketPanel
from alphamcts.data.synthetic import generate_synthetic_panel
from alphamcts.evaluation.evaluator import MultiDimEvaluator
from alphamcts.expression.formula import AlphaFormula
from alphamcts.model import CombinerPipeline
from alphamcts.zoo import AlphaRecord, AlphaZoo

WINDOW_RANGE = (2, 60)


@pytest.fixture
def masked_panel():
    base = generate_synthetic_panel(n_stocks=30, n_days=300, seed=7)
    # PIT-style mask: first half of stocks members early, membership rotates mid-sample
    mask = pd.DataFrame(False, index=base.dates, columns=base.instruments)
    half = 150
    mask.iloc[:half, :20] = True
    mask.iloc[half:, 10:] = True
    return MarketPanel(fields=base.fields, universe_mask=mask)


def test_apply_universe_masks_non_members(masked_panel):
    close = masked_panel.fields["close"]
    masked = masked_panel.apply_universe(close)
    assert masked.iloc[0, 25] != masked.iloc[0, 25]   # non-member -> NaN
    assert masked.iloc[0, 5] == masked.iloc[0, 5]     # member -> value
    assert masked.iloc[200, 5] != masked.iloc[200, 5] # rotated out -> NaN


def test_dates_between():
    panel = generate_synthetic_panel(n_stocks=5, n_days=100, seed=0)
    d = panel.dates_between(str(panel.dates[10].date()), str(panel.dates[20].date()))
    assert len(d) == 11
    assert panel.dates_between(None, None).equals(panel.dates)


def test_evaluator_with_train_range_and_mask(masked_panel):
    zoo = AlphaZoo()
    start, end = str(masked_panel.dates[60].date()), str(masked_panel.dates[200].date())
    ev = MultiDimEvaluator(
        masked_panel, horizon=10, zoo=zoo, train_range=(start, end), top_frac=0.1
    )
    assert ev.train_dates[0] == masked_panel.dates[60]
    assert ev.train_dates[-1] == masked_panel.dates[200]

    f = AlphaFormula.from_json(
        {
            "name": "rev",
            "description": "",
            "formula": [
                {"name": "Pct", "param": ["w"], "input": ["close"], "output": "r"},
                {"name": "Neg", "param": [], "input": ["r"], "output": "a"},
            ],
            "arguments": [{"w": 5}],
        },
        FIELDS,
        WINDOW_RANGE,
    )
    result = ev.evaluate(f, lambda e, h: (8.0, "ok"))
    # values restricted to the PIT universe: no valid value for a non-member date/stock
    v = result.values
    date_early = ev.train_dates[10]
    assert np.isnan(v.loc[date_early].iloc[25])
    # cross-section per day never exceeds member count (20)
    assert int(v.notna().sum(axis=1).max()) <= 20


def test_combiner_periods_and_mask(masked_panel):
    zoo = AlphaZoo()
    ev = MultiDimEvaluator(masked_panel, horizon=10, zoo=zoo, train_ratio=0.5, top_frac=0.1)
    f = AlphaFormula.from_json(
        {
            "name": "rev",
            "description": "",
            "formula": [
                {"name": "Pct", "param": ["w"], "input": ["close"], "output": "r"},
                {"name": "Neg", "param": [], "input": ["r"], "output": "a"},
            ],
            "arguments": [{"w": 5}],
        },
        FIELDS,
        WINDOW_RANGE,
    )
    r = ev.evaluate(f, lambda e, h: (8.0, "ok"))
    record = AlphaRecord("rev", "", r.expression, f, r.chosen_args, r.metrics, values=r.values)

    dates = masked_panel.dates
    periods = {
        "valid": (str(dates[150].date()), str(dates[220].date())),
        "test": (str(dates[221].date()), str(dates[-1].date())),
    }
    pipe = CombinerPipeline(
        masked_panel,
        horizon=10,
        train_range=(str(dates[0].date()), str(dates[149].date())),
        eval_periods=periods,
    )
    assert set(pipe.eval_periods) == {"valid", "test"}
    model, features = pipe.fit("lightgbm", [record])
    for name, d in pipe.eval_periods.items():
        report = pipe.evaluate_period(model, features, d, "lightgbm", 1)
        assert report.ic == report.ic
    # predictions restricted to members
    pred = pipe.predict_panel(model, features, pipe.eval_periods["test"])
    assert int(pred.notna().sum(axis=1).max()) <= 20
