"""Cross-sectional industry + size neutralization.

Removes, per trading day, the component of a panel explained by industry membership
(SW L1 dummies, i.e. within-industry demeaning) and log market cap (single-variable
OLS after industry demeaning, which by Frisch-Waugh equals the full dummy+size OLS
residual). Used to strip style/regime exposure from both target returns and factor
values so mining and evaluation measure pure stock-selection alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN_INDUSTRY = "UNK"


def industry_demean(df: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    """Subtract the per-day industry mean from each cell (NaN cells stay NaN)."""
    values = df.stack().dropna()
    if values.empty:
        return df
    keys = industry.stack().reindex(values.index).fillna(UNKNOWN_INDUSTRY)
    keys = keys.replace("", UNKNOWN_INDUSTRY)
    dates = values.index.get_level_values(0)
    demeaned = values - values.groupby([dates, keys]).transform("mean")
    return demeaned.unstack().reindex(index=df.index, columns=df.columns)


def neutralize_panel(
    df: pd.DataFrame,
    industry: pd.DataFrame | None = None,
    log_mv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day residual of df on industry dummies and/or log market cap."""
    out = df
    if industry is not None:
        out = industry_demean(out, industry)
    if log_mv is not None:
        x = log_mv.where(out.notna())
        if industry is not None:
            x = industry_demean(x, industry)
        else:
            x = x.sub(x.mean(axis=1), axis=0)
        x = x.fillna(0.0).where(out.notna())
        num = (x * out).sum(axis=1)
        den = (x * x).sum(axis=1)
        beta = num / den.replace(0.0, np.nan)
        out = out - x.mul(beta.fillna(0.0), axis=0)
    return out
