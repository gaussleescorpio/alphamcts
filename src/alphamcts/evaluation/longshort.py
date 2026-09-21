"""Long-short quantile decomposition of an alpha signal.

Each day the cross-section is split into ``n`` quantile buckets by signal value
(Q1 = lowest, Qn = highest). Three diagnostics summarize the decomposition:

- **quantile mean returns**: per-bucket mean forward return (per-day cross-sectional
  mean, then averaged over days) -- an effective alpha should be monotone in the
  bucket index with Qn - Q1 > 0;
- **monotonic fraction**: share of adjacent bucket pairs ordered correctly;
- **long-short Sharpe**: annualized Sharpe of the daily-rebalanced, cost-free
  Qn-minus-Q1 portfolio on *next-day* returns (tradable-space diagnostic, robust
  to overlapping-horizon autocorrelation that would inflate a Sharpe computed on
  raw h-day forward returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def quantile_labels(pred: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Per-day quantile bucket (1..n) by cross-sectional rank; NaN where pred is NaN."""
    r = pred.rank(axis=1, pct=True, method="first")
    q = np.ceil(r * n).clip(1, n)
    return q.where(pred.notna())


def quantile_mean_returns(pred: pd.DataFrame, fwd: pd.DataFrame, n: int = 5) -> pd.Series:
    """Mean forward return per quantile bucket, indexed 1..n (Q1 = lowest signal)."""
    q = quantile_labels(pred, n)
    out = {}
    for k in range(1, n + 1):
        daily = fwd.where(q == k).mean(axis=1)
        out[k] = float(daily.mean()) if daily.notna().any() else float("nan")
    return pd.Series(out, name="mean_fwd_return")


def monotonic_fraction(qmeans: pd.Series) -> float:
    """Fraction of adjacent bucket pairs with mean(Q_{k+1}) >= mean(Q_k)."""
    vals = qmeans.dropna().sort_index().to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float(np.mean(np.diff(vals) >= 0))


def long_short_returns(pred: pd.DataFrame, next_day_ret: pd.DataFrame, n: int = 5) -> pd.Series:
    """Daily Qn-minus-Q1 return series (equal weight in each leg, cost-free).

    ``next_day_ret`` must already be aligned to the signal date, i.e. the return
    earned by a position formed on that day's signal (``close.pct_change().shift(-1)``).
    """
    q = quantile_labels(pred, n)
    long = next_day_ret.where(q == n).mean(axis=1)
    short = next_day_ret.where(q == 1).mean(axis=1)
    return (long - short).dropna()


def annualized_sharpe(series: pd.Series, periods: int = TRADING_DAYS) -> float:
    s = series.dropna()
    if len(s) < 20:
        return float("nan")
    std = float(s.std())
    if std < 1e-12:
        return float("nan")
    return float(s.mean() / std * np.sqrt(periods))


def long_short_sharpe(pred: pd.DataFrame, next_day_ret: pd.DataFrame, n: int = 5) -> float:
    return annualized_sharpe(long_short_returns(pred, next_day_ret, n))


def monotonicity_score(q_spread: float, q_monotonic: float, ls_sharpe: float) -> float:
    """Composite cross-sectional monotonicity score in [0, 1] (MCTS dimension).

    - Q5 - Q1 spread must be positive, otherwise the score is 0 (wrong direction);
    - otherwise 40% comes from the fraction of correctly ordered adjacent quantile
      pairs and 60% from the long-short Sharpe, saturating at 2.0.
    """
    if not q_spread == q_spread or q_spread <= 0.0:
        return 0.0
    mono = q_monotonic if q_monotonic == q_monotonic else 0.0
    ls = ls_sharpe if ls_sharpe == ls_sharpe else 0.0
    return 0.4 * mono + 0.6 * min(max(ls / 2.0, 0.0), 1.0)
