"""Anti-overfitting admission tests (ported from the FactorForge funnel).

All tests operate on training-period data only; the out-of-sample window never
participates in admission decisions.

- Newey-West (HAC) one-sided mean test on the daily RankIC series, with Bartlett
  weights and lag equal to the holding horizon (overlapping-label autocorrelation).
- Time-block stability: fraction of chronological blocks with positive mean IC.
- Yearly sign consistency: fraction of calendar years with positive mean IC.
- Negative control (asset mismatch): factor values are shifted by one instrument
  cross-sectionally; a genuine cross-sectional signal must collapse to ~0 RankIC.
- Reconstruction R^2: how well the candidate's daily ranks can be linearly
  reconstructed from the existing zoo's ranks (fit on the first 60% of days,
  scored on the last 40%). High R^2 = redundant, likely a re-mix of known alphas.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import cs_rank, rank_ic_series


def newey_west_mean_test(series: pd.Series, lag: int | None = None) -> tuple[float, float]:
    """One-sided HAC t-test of mean > 0. Returns (t_stat, p_value)."""
    s = np.asarray(series.dropna(), dtype=float)
    n = len(s)
    if n < 3:
        return float("nan"), 1.0
    if lag is None:
        lag = int(4 * (n / 100.0) ** (2.0 / 9.0))  # Newey-West automatic bandwidth
    lag = max(0, min(int(lag), n - 1))
    mean = float(s.mean())
    d = s - mean
    lrv = float(np.dot(d, d)) / n
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)  # Bartlett kernel
        lrv += 2.0 * w * float(np.dot(d[ell:], d[:-ell])) / n
    se = math.sqrt(max(lrv, 1e-18) / n)
    t = mean / se
    p = 0.5 * math.erfc(t / math.sqrt(2.0))  # sf of the standard normal
    return t, p


def block_positive_fraction(series: pd.Series, n_blocks: int = 4, min_obs_per_block: int = 20) -> float:
    """Fraction of chronological blocks whose mean is positive."""
    s = series.dropna()
    if len(s) < n_blocks * min_obs_per_block:
        return float("nan")
    blocks = np.array_split(np.arange(len(s)), n_blocks)
    means = [float(s.iloc[idx].mean()) for idx in blocks if len(idx) > 0]
    if not means:
        return float("nan")
    return sum(1 for m in means if m > 0) / len(means)


def yearly_positive_fraction(series: pd.Series, min_obs_per_year: int = 60) -> float:
    """Fraction of calendar years (with enough observations) whose mean is positive."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    grouped = s.groupby(s.index.year)
    means = [float(g.mean()) for _, g in grouped if len(g) >= min_obs_per_year]
    if not means:
        return float("nan")
    return sum(1 for m in means if m > 0) / len(means)


def asset_mismatch_ic(values: pd.DataFrame, fwd_returns: pd.DataFrame) -> float:
    """|mean RankIC| after shifting factor values by one instrument (negative control)."""
    shifted = pd.DataFrame(
        np.roll(values.to_numpy(), 1, axis=1), index=values.index, columns=values.columns
    )
    ic = rank_ic_series(shifted, fwd_returns).dropna()
    if ic.empty:
        return float("nan")
    return abs(float(ic.mean()))


def _rank_matrix(df: pd.DataFrame) -> np.ndarray:
    """Cross-sectional rank in [-0.5, 0.5], NaN -> 0 (neutral)."""
    return (cs_rank(df) - 0.5).fillna(0.0).to_numpy()


def reconstruction_r2(
    candidate: pd.DataFrame,
    pool: list[pd.DataFrame],
    train_frac: float = 0.6,
    ridge: float = 1e-4,
) -> float:
    """Out-of-time R^2 of reconstructing candidate ranks from pool ranks.

    Fit ridge-regularized OLS on the first `train_frac` of days, score R^2 on the
    remaining days. Rows where the candidate has no value are dropped.
    """
    if not pool:
        return float("nan")
    n_days = len(candidate.index)
    split = int(n_days * train_frac)
    if split < 10 or n_days - split < 10:
        return float("nan")

    y_full = (cs_rank(candidate) - 0.5).to_numpy()
    x_full = np.stack([_rank_matrix(p.reindex_like(candidate)) for p in pool], axis=-1)

    def _flatten(day_slice: slice) -> tuple[np.ndarray, np.ndarray]:
        y = y_full[day_slice].ravel()
        x = x_full[day_slice].reshape(-1, len(pool))
        valid = np.isfinite(y)
        return x[valid], y[valid]

    x_tr, y_tr = _flatten(slice(0, split))
    x_te, y_te = _flatten(slice(split, n_days))
    if len(y_tr) < 100 or len(y_te) < 100:
        return float("nan")

    xtx = x_tr.T @ x_tr + ridge * np.eye(len(pool))
    beta = np.linalg.solve(xtx, x_tr.T @ y_tr)
    resid = y_te - x_te @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot
