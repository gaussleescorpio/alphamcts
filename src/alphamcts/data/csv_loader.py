"""Load a MarketPanel from a directory of per-instrument CSV files.

Each file `<instrument>.csv` must have columns: date, open, high, low, close, volume
and optionally vwap (approximated as (open+high+low+close)/4 when absent).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import FIELDS, MarketPanel

_REQUIRED = ("open", "high", "low", "close", "volume")


def load_csv_dir(csv_dir: str | Path) -> MarketPanel:
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    per_field: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for path in files:
        inst = path.stem
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if "vwap" not in df.columns:
            df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        for field in FIELDS:
            per_field[field][inst] = df[field].astype(float)

    fields = {f: pd.DataFrame(series).sort_index() for f, series in per_field.items()}
    return MarketPanel(fields=fields)
