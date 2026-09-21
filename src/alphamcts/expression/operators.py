"""Vectorized operators from the paper (Appendix E, Table 2).

Every operator maps (T x N) DataFrames to a (T x N) DataFrame. Rolling operators use
strictly backward-looking windows (`min_periods=window`), so there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def _roll(x: pd.DataFrame, t: int):
    return x.rolling(int(t), min_periods=int(t))


# ---------------------------------------------------------------------------
# Unary element-wise
# ---------------------------------------------------------------------------

def _neg(x):
    return -x


def _abs(x):
    return x.abs()


def _square(x):
    return x * x


def _inverse(x):
    return _clean(1.0 / x)


def _sign(x):
    return np.sign(x)


def _sin(x):
    return np.sin(x)


def _cos(x):
    return np.cos(x)


def _tanh(x):
    return np.tanh(x)


def _log(x):
    with np.errstate(all="ignore"):
        return _clean(pd.DataFrame(np.log(x.where(x > 0)), index=x.index, columns=x.columns))


# ---------------------------------------------------------------------------
# Unary temporal (one window/lag parameter)
# ---------------------------------------------------------------------------

def _delay(x, t):
    return x.shift(int(t))


def _diff(x, t):
    return x - x.shift(int(t))


def _pct(x, t):
    return _clean(x / x.shift(int(t)) - 1.0)


def _ma(x, t):
    return _roll(x, t).mean()


def _med(x, t):
    return _roll(x, t).median()


def _sum(x, t):
    return _roll(x, t).sum()


def _std(x, t):
    return _roll(x, t).std()


def _max(x, t):
    return _roll(x, t).max()


def _min(x, t):
    return _roll(x, t).min()


def _rank(x, t):
    return _roll(x, t).rank(pct=True)


def _skew(x, t):
    return _roll(x, t).skew()


def _kurt(x, t):
    return _roll(x, t).kurt()


def _vari(x, t):
    return _clean(_std(x, t) / _ma(x, t))


def _zscore(x, t):
    return _clean((x - _ma(x, t)) / _std(x, t))


def _autocorr(x, t, n):
    return _clean(_roll(x, t).corr(x.shift(int(n))))


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------

def _add(x, y):
    return x + y


def _sub(x, y):
    return x - y


def _mul(x, y):
    return x * y


def _div(x, y):
    return _clean(x / y)


def _greater(x, y):
    return (x > y).astype(float).where(x.notna() & y.notna())


def _less(x, y):
    return (x < y).astype(float).where(x.notna() & y.notna())


def _cov(x, y, t):
    return _clean(_roll(x, t).cov(y))


def _corr(x, y, t):
    return _clean(_roll(x, t).corr(y))


def _ts_regress(x, y, t):
    # slope of the OLS regression of y on x over the past t days: Cov(x, y) / Var(x)
    return _clean(_roll(x, t).cov(y) / _roll(x, t).var())


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    n_inputs: int
    n_params: int
    description: str
    func: Callable
    infix: str | None = None  # rendering hint for arithmetic operators
    aliases: tuple[str, ...] = field(default=())

    def __call__(self, *args) -> pd.DataFrame:
        out = self.func(*args)
        if isinstance(out, np.ndarray):
            ref = args[0]
            out = pd.DataFrame(out, index=ref.index, columns=ref.columns)
        return _clean(out)


_SPECS: list[OperatorSpec] = [
    # unary element-wise
    OperatorSpec("Neg", 1, 0, "The opposite value of x: -x.", _neg, infix=None, aliases=("Opposite", "Negative")),
    OperatorSpec("Abs", 1, 0, "The absolute value of x.", _abs),
    OperatorSpec("Square", 1, 0, "The square of x: x^2.", _square, aliases=("Pow2",)),
    OperatorSpec("Inverse", 1, 0, "The inverse value of x: 1/x.", _inverse, aliases=("Inv",)),
    OperatorSpec("Sign", 1, 0, "The sign of x (-1, 0 or 1).", _sign),
    OperatorSpec("Sin", 1, 0, "Sine of x.", _sin),
    OperatorSpec("Cos", 1, 0, "Cosine of x.", _cos),
    OperatorSpec("Tanh", 1, 0, "Hyperbolic tangent of x.", _tanh),
    OperatorSpec("Log", 1, 0, "Natural logarithm of x (NaN for non-positive values).", _log),
    # unary temporal
    OperatorSpec("Delay", 1, 1, "The value of x at t trading days prior.", _delay),
    OperatorSpec("Diff", 1, 1, "x minus its value t days prior: x - Delay(x, t).", _diff),
    OperatorSpec("Pct", 1, 1, "Rate of change of x relative to its value t days prior.", _pct, aliases=("Ret", "Roc")),
    OperatorSpec("Ma", 1, 1, "Mean of x over the past t days.", _ma, aliases=("Mean", "Sma")),
    OperatorSpec("Med", 1, 1, "Median of x over the past t days.", _med, aliases=("Median",)),
    OperatorSpec("Sum", 1, 1, "Sum of x over the past t days.", _sum),
    OperatorSpec("Std", 1, 1, "Standard deviation of x over the past t days.", _std),
    OperatorSpec("Max", 1, 1, "Maximum of x over the past t days.", _max),
    OperatorSpec("Min", 1, 1, "Minimum of x over the past t days.", _min),
    OperatorSpec("Rank", 1, 1, "Percentile rank of x relative to its values over the past t days.", _rank, aliases=("TsRank",)),
    OperatorSpec("Skew", 1, 1, "Skewness of x over the past t days.", _skew),
    OperatorSpec("Kurt", 1, 1, "Kurtosis of x over the past t days.", _kurt),
    OperatorSpec("Vari", 1, 1, "Coefficient of variation: Std(x, t) / Ma(x, t).", _vari, aliases=("Variation",)),
    OperatorSpec("Zscore", 1, 1, "Z-score of x based on its mean and std over the past t days.", _zscore),
    OperatorSpec("Autocorr", 1, 2, "Autocorrelation of x with lag n over the past t days (params: t, n).", _autocorr),
    # binary
    OperatorSpec("Add", 2, 0, "x + y.", _add, infix="+", aliases=("Plus",)),
    OperatorSpec("Sub", 2, 0, "x - y.", _sub, infix="-", aliases=("Minus", "Subtract")),
    OperatorSpec("Mul", 2, 0, "x * y.", _mul, infix="*", aliases=("Multiply", "Times")),
    OperatorSpec("Div", 2, 0, "x / y.", _div, infix="/", aliases=("Divide",)),
    OperatorSpec("Greater", 2, 0, "1.0 if x > y else 0.0.", _greater, aliases=("Gt",)),
    OperatorSpec("Less", 2, 0, "1.0 if x < y else 0.0.", _less, aliases=("Lt",)),
    OperatorSpec("Cov", 2, 1, "Covariance between x and y over the past t days.", _cov, aliases=("Covariance",)),
    OperatorSpec("Corr", 2, 1, "Pearson correlation between x and y over the past t days.", _corr, aliases=("Correlation",)),
    OperatorSpec(
        "TsRegress", 2, 1,
        "Slope of the linear regression of y (second input) on x (first input) over the past "
        "t days: Cov(x, y) / Var(x). Measures the time-series sensitivity of y to x.",
        _ts_regress,
        aliases=("ts_regress", "Regress", "RegSlope", "Beta"),
    ),
]

OPERATORS: dict[str, OperatorSpec] = {spec.name: spec for spec in _SPECS}

_LOOKUP: dict[str, OperatorSpec] = {}
for spec in _SPECS:
    _LOOKUP[spec.name.lower()] = spec
    for alias in spec.aliases:
        _LOOKUP[alias.lower()] = spec


def get_operator(name: str) -> OperatorSpec | None:
    return _LOOKUP.get(name.strip().lower())


def operators_prompt_block() -> str:
    """Render the operator list for LLM prompts."""
    lines = []
    for spec in _SPECS:
        inputs = ", ".join(f"x{i+1}" if spec.n_inputs > 1 else "x" for i in range(spec.n_inputs))
        params = ", ".join(("t", "n")[: spec.n_params])
        sig = f"{spec.name}({inputs}" + (f", {params}" if params else "") + ")"
        lines.append(f"- {sig}: {spec.description}")
    return "\n".join(lines)
