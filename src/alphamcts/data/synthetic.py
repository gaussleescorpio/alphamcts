"""Synthetic market generator.

Produces an OHLCV+VWAP panel with realistic cross-sectional structure so the mining pipeline
can be exercised end-to-end offline. The log price of each stock is the sum of:

- a fundamental random walk (market factor with per-stock betas + idiosyncratic noise), and
- a mean-reverting temporary mispricing component u_t (AR(1) with rho < 1).

Because u_t decays, recent price run-ups predict negative future returns: short-horizon
reversal alphas (and volume-conditioned variants) carry genuine predictive power, which the
mining loop can discover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MarketPanel


def generate_synthetic_panel(
    n_stocks: int = 100,
    n_days: int = 1200,
    seed: int = 0,
    mispricing_rho: float = 0.9,
    mispricing_vol: float = 0.008,
    fundamental_vol: float = 0.012,
) -> MarketPanel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-04", periods=n_days)
    instruments = pd.Index([f"S{i:04d}" for i in range(n_stocks)], name="instrument")

    betas = rng.normal(1.0, 0.3, size=n_stocks)
    market = rng.normal(0.0002, 0.008, size=n_days)
    fundamental_noise = rng.normal(0.0, fundamental_vol, size=(n_days, n_stocks))
    fund_log = np.cumsum(market[:, None] * betas[None, :] + fundamental_noise, axis=0)

    # Temporary mispricing: AR(1) that mean-reverts, creating a short-horizon reversal effect.
    u = np.zeros((n_days, n_stocks))
    eps_u = rng.normal(0.0, mispricing_vol, size=(n_days, n_stocks))
    for t in range(1, n_days):
        u[t] = mispricing_rho * u[t - 1] + eps_u[t]

    log_close = np.log(20.0) + fund_log + u
    close = np.exp(log_close)

    prev_close = np.vstack([close[:1], close[:-1]])
    gap = rng.normal(0.0, 0.003, size=(n_days, n_stocks))
    open_ = prev_close * (1.0 + gap)

    intraday = np.abs(rng.normal(0.0, 0.008, size=(n_days, n_stocks)))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    vwap = (open_ + high + low + close) / 4.0 * (1.0 + rng.normal(0.0, 0.001, size=(n_days, n_stocks)))
    vwap = np.clip(vwap, low, high)

    rets = np.diff(log_close, axis=0, prepend=log_close[:1])
    base_volume = rng.lognormal(mean=13.0, sigma=0.5, size=n_stocks)
    vol_noise = rng.lognormal(mean=0.0, sigma=0.35, size=(n_days, n_stocks))
    # Volume rises with |return| and spikes on large mispricing shocks (which later revert).
    volume = base_volume[None, :] * vol_noise * (1.0 + 25.0 * np.abs(rets) + 40.0 * np.abs(eps_u))
    spikes = rng.random(size=(n_days, n_stocks)) < 0.01
    volume = volume * np.where(spikes, 3.0, 1.0)

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=instruments)

    return MarketPanel(
        fields={
            "open": frame(open_),
            "high": frame(high),
            "low": frame(low),
            "close": frame(close),
            "volume": frame(volume),
            "vwap": frame(vwap),
        }
    )
