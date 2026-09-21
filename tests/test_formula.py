import numpy as np
import pytest

from alphamcts.data.base import FIELDS
from alphamcts.expression.formula import AlphaFormula, FormulaError

WINDOW_RANGE = (2, 60)

VALID = {
    "name": "intraday_pressure_zscore",
    "description": "End-of-day buying pressure, smoothed and normalized.",
    "formula": [
        {"name": "Sub", "param": [], "input": ["close", "vwap"], "output": "pressure"},
        {"name": "Ma", "param": ["smooth_window"], "input": ["pressure"], "output": "trend"},
        {"name": "Zscore", "param": ["norm_window"], "input": ["trend"], "output": "alpha"},
    ],
    "arguments": [
        {"smooth_window": 20, "norm_window": 30},
        {"smooth_window": 10, "norm_window": 20},
    ],
}


def parse(obj):
    return AlphaFormula.from_json(obj, FIELDS, WINDOW_RANGE)


def test_parse_and_render():
    f = parse(VALID)
    assert f.param_names() == ["smooth_window", "norm_window"]
    assert f.to_string(f.arguments[0]) == "Zscore(Ma((close-vwap),20),30)"
    assert f.to_string() == "Zscore(Ma((close-vwap),smooth_window),norm_window)"


def test_evaluate_matches_manual(panel):
    f = parse(VALID)
    out = f.evaluate(panel.fields, {"smooth_window": 20, "norm_window": 30})
    pressure = panel.fields["close"] - panel.fields["vwap"]
    trend = pressure.rolling(20, min_periods=20).mean()
    manual = (trend - trend.rolling(30, min_periods=30).mean()) / trend.rolling(30, min_periods=30).std()
    diff = (out - manual).abs().stack().dropna()
    assert diff.max() < 1e-10
    assert out.shape == panel.fields["close"].shape


def test_unknown_operator():
    bad = dict(VALID, formula=[{"name": "Wavelet", "param": [], "input": ["close"], "output": "a"}])
    bad["arguments"] = [{}]
    with pytest.raises(FormulaError, match="unknown operator"):
        parse(bad)


def test_undefined_input():
    bad = dict(VALID, formula=[{"name": "Ma", "param": ["w"], "input": ["momentum"], "output": "a"}])
    bad["arguments"] = [{"w": 5}]
    with pytest.raises(FormulaError, match="neither an available data field"):
        parse(bad)


def test_numeric_input_rejected():
    bad = dict(VALID, formula=[{"name": "Add", "param": [], "input": ["close", "0.5"], "output": "a"}])
    bad["arguments"] = [{}]
    with pytest.raises(FormulaError, match="numeric"):
        parse(bad)


def test_missing_argument_value():
    bad = dict(VALID, arguments=[{"smooth_window": 20}])
    with pytest.raises(FormulaError, match="missing values"):
        parse(bad)


def test_window_out_of_range():
    bad = dict(VALID, arguments=[{"smooth_window": 500, "norm_window": 30}])
    with pytest.raises(FormulaError, match="outside the allowed range"):
        parse(bad)


def test_arity_mismatch():
    bad = dict(VALID, formula=[{"name": "Corr", "param": ["w"], "input": ["close"], "output": "a"}])
    bad["arguments"] = [{"w": 10}]
    with pytest.raises(FormulaError, match="expects 2 input"):
        parse(bad)


def test_too_many_params():
    steps = [
        {"name": "Ma", "param": ["w1"], "input": ["close"], "output": "a"},
        {"name": "Ma", "param": ["w2"], "input": ["a"], "output": "b"},
        {"name": "Ma", "param": ["w3"], "input": ["b"], "output": "c"},
        {"name": "Ma", "param": ["w4"], "input": ["c"], "output": "d"},
    ]
    bad = dict(VALID, formula=steps, arguments=[{"w1": 5, "w2": 5, "w3": 5, "w4": 5}])
    with pytest.raises(FormulaError, match="at most 3"):
        parse(bad)


def test_tree_depth_and_size():
    f = parse(VALID)
    t = f.tree()
    assert t.depth() == 4  # Zscore -> Ma -> Sub -> leaf
    assert t.size() == 5
