"""Backtesting metrics: IC, RankIC, RankIR, turnover and factor correlation.

All metrics operate on (dates x instruments) DataFrames and are computed cross-sectionally
per day, then aggregated over time (paper Appendix G, "Predictive Performance Evaluation
Metrics").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per day."""
    return df.rank(axis=1, pct=True)


def cs_pearson(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Pearson correlation between two aligned panels."""
    mask = x.notna() & y.notna()
    n = mask.sum(axis=1)
    xv = x.where(mask)
    yv = y.where(mask)
    xd = xv.sub(xv.mean(axis=1), axis=0)
    yd = yv.sub(yv.mean(axis=1), axis=0)
    cov = (xd * yd).sum(axis=1)
    denom = np.sqrt((xd * xd).sum(axis=1) * (yd * yd).sum(axis=1))
    with np.errstate(all="ignore"):
        corr = cov / denom
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr[n < 3] = np.nan
    return corr


def ic_series(factor: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    return cs_pearson(factor, fwd_returns)


def rank_ic_series(factor: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    return cs_pearson(cs_rank(factor), cs_rank(fwd_returns))


def mean_and_ir(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if s.empty:
        return float("nan"), float("nan")
    mean = float(s.mean())
    std = float(s.std())
    ir = mean / std if std > 1e-12 else float("nan")
    return mean, ir


def top_portfolio_weights(values: pd.DataFrame, top_frac: float = 0.1) -> pd.DataFrame:
    """Equal-weight top-fraction portfolio implied by the factor values."""
    ranks = values.rank(axis=1, ascending=False, method="first")
    n_valid = values.notna().sum(axis=1)
    k = (n_valid * top_frac).clip(lower=1).round()
    member = ranks.le(k, axis=0)
    weights = member.div(member.sum(axis=1).replace(0, np.nan), axis=0)
    return weights.fillna(0.0)


def daily_turnover(values: pd.DataFrame, top_frac: float = 0.1) -> float:
    """Average daily change in portfolio holdings: mean of sum_i |w_t,i - w_{t-1,i}| in [0, 2].

    Days on which the portfolio is empty (warm-up NaNs) are excluded, as is the initial
    portfolio construction day.
    """
    weights = top_portfolio_weights(values, top_frac)
    active = weights.sum(axis=1) > 0
    weights = weights.loc[active]
    if len(weights) < 2:
        return float("nan")
    turn = weights.diff().abs().sum(axis=1).iloc[1:]
    return float(turn.mean())


def factor_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Correlation between two alpha factors: average daily cross-sectional Spearman corr."""
    series = cs_pearson(cs_rank(a), cs_rank(b)).dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


@dataclass
class AlphaMetrics:
    ic: float
    ic_ir: float
    rank_ic: float
    rank_ir: float
    turnover: float
    max_corr: float
    coverage: float  # fraction of valid factor values
    # anti-overfitting admission statistics (training period only)
    hac_t: float = float("nan")            # Newey-West t-stat of mean RankIC
    hac_p: float = float("nan")            # one-sided HAC p-value
    block_pos_frac: float = float("nan")   # fraction of time blocks with positive mean IC
    year_pos_frac: float = float("nan")    # fraction of calendar years with positive mean IC
    control_ic: float = float("nan")       # |RankIC| of the asset-mismatch negative control
    recon_r2: float = float("nan")         # out-of-time reconstruction R^2 from the zoo
    # long-short quantile decomposition (training period only)
    q_spread: float = float("nan")         # mean fwd return spread Q5 - Q1 (quintiles by signal)
    q_monotonic: float = float("nan")      # fraction of adjacent quantile pairs correctly ordered
    ls_sharpe: float = float("nan")        # annualized Sharpe of the daily Q5-Q1 portfolio (cost-free)
    # hold-out validation period (only filled when the evaluator has a valid_range)
    rank_ic_valid: float = float("nan")    # mean RankIC on the validation period
    rank_ir_valid: float = float("nan")    # RankIR on the validation period

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "AlphaMetrics":
        return cls(**{k: float(d.get(k, float("nan"))) for k in cls.__dataclass_fields__})


def compute_alpha_metrics(
    values: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    top_frac: float = 0.1,
    max_corr: float = float("nan"),
) -> AlphaMetrics:
    ic, ic_ir = mean_and_ir(ic_series(values, fwd_returns))
    rank_ic, rank_ir = mean_and_ir(rank_ic_series(values, fwd_returns))
    turnover = daily_turnover(values, top_frac)
    coverage = float(values.notna().mean().mean())
    return AlphaMetrics(
        ic=ic,
        ic_ir=ic_ir,
        rank_ic=rank_ic,
        rank_ir=rank_ir,
        turnover=turnover,
        max_corr=max_corr,
        coverage=coverage,
    )
