"""Market data panel abstraction.

A :class:`MarketPanel` holds one (T x N) DataFrame per raw field (dates x instruments),
matching the paper's setting: daily OHLC prices, volume and VWAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@dataclass
class MarketPanel:
    """Panel of raw market data: one (dates x instruments) frame per field.

    `target_returns` optionally carries externally-computed forward returns (e.g. a
    database `fwd_20` column, signal-date aligned); when present it overrides the
    close-to-close computation in :meth:`forward_returns`.

    `universe_mask` optionally marks point-in-time universe membership (e.g. CSI300
    constituents as of each date). Factor values are computed on full history (so rolling
    windows survive membership changes) but evaluation, ranking and backtests are
    restricted to member stocks per date.
    """

    fields: dict[str, pd.DataFrame]
    target_returns: pd.DataFrame | None = None
    universe_mask: pd.DataFrame | None = None
    industry: pd.DataFrame | None = None      # per-day SW L1 industry code ('' = unknown)
    log_mv: pd.DataFrame | None = None        # per-day log circulating market cap
    neutralize_factors: bool = False          # neutralize factor values in eval/combining

    def __post_init__(self) -> None:
        missing = [f for f in FIELDS if f not in self.fields]
        if missing:
            raise ValueError(f"MarketPanel missing fields: {missing}")
        ref = self.fields["close"]
        for name, df in self.fields.items():
            if not df.index.equals(ref.index) or not df.columns.equals(ref.columns):
                raise ValueError(f"Field '{name}' is not aligned with 'close'")
        if self.target_returns is not None:
            self.target_returns = self.target_returns.reindex(
                index=ref.index, columns=ref.columns
            )
        if self.universe_mask is not None:
            self.universe_mask = (
                self.universe_mask.reindex(index=ref.index, columns=ref.columns)
                .fillna(False)
                .astype(bool)
            )
        if self.industry is not None:
            self.industry = self.industry.reindex(index=ref.index, columns=ref.columns)
        if self.log_mv is not None:
            self.log_mv = self.log_mv.reindex(index=ref.index, columns=ref.columns)

    def neutralize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Industry+size residual of a panel; identity when no exposure data loaded."""
        if self.industry is None and self.log_mv is None:
            return df
        from .neutralize import neutralize_panel

        return neutralize_panel(df, industry=self.industry, log_mv=self.log_mv)

    def maybe_neutralize_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.neutralize(df) if self.neutralize_factors else df

    def apply_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict a (dates x instruments) panel to point-in-time universe members."""
        if self.universe_mask is None:
            return df
        return df.where(self.universe_mask)

    def dates_between(self, start: str | None, end: str | None) -> pd.Index:
        dates = self.dates
        mask = pd.Series(True, index=dates)
        if start is not None:
            mask &= dates >= pd.Timestamp(start)
        if end is not None:
            mask &= dates <= pd.Timestamp(end)
        return dates[mask]

    @property
    def dates(self) -> pd.Index:
        return self.fields["close"].index

    @property
    def instruments(self) -> pd.Index:
        return self.fields["close"].columns

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_stocks(self) -> int:
        return len(self.instruments)

    def forward_returns(self, horizon: int) -> pd.DataFrame:
        """Realized future return over `horizon` days, aligned at the prediction date t.

        Uses the externally-supplied target panel when available (already aligned at the
        signal date with next-day entry); otherwise computes close-to-close returns.
        """
        if self.target_returns is not None:
            return self.target_returns
        close = self.fields["close"]
        return close.shift(-horizon) / close - 1.0

    def daily_returns(self) -> pd.DataFrame:
        return self.fields["close"].pct_change()

    def split_date(self, train_ratio: float) -> pd.Timestamp:
        """Chronological split date: dates <= split_date are the training period."""
        idx = int(len(self.dates) * train_ratio) - 1
        idx = max(0, min(idx, len(self.dates) - 1))
        return self.dates[idx]

    def train_mask(self, train_ratio: float) -> pd.Series:
        split = self.split_date(train_ratio)
        return pd.Series(self.dates <= split, index=self.dates)


def load_panel(data_cfg: dict[str, Any]) -> MarketPanel:
    """Build a panel from the `data` section of the config."""
    source = data_cfg.get("source", "synthetic")
    if source == "synthetic":
        from .synthetic import generate_synthetic_panel

        return generate_synthetic_panel(
            n_stocks=int(data_cfg.get("n_stocks", 100)),
            n_days=int(data_cfg.get("n_days", 1200)),
            seed=int(data_cfg.get("seed", 0)),
        )
    if source == "csv":
        from .csv_loader import load_csv_dir

        csv_dir = data_cfg.get("csv_dir")
        if not csv_dir:
            raise ValueError("data.csv_dir must be set when data.source is 'csv'")
        return load_csv_dir(csv_dir)
    if source == "clickhouse":
        from .clickhouse_loader import load_clickhouse_panel

        return load_clickhouse_panel(data_cfg.get("clickhouse", data_cfg))
    raise ValueError(f"Unknown data source: {source!r}")
