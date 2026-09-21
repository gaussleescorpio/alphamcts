"""Load a MarketPanel from a local ClickHouse instance (HTTP interface).

Designed for the `quant_system` schema:

- price table (default ``daily_hfq``): ts_code, trade_date (yyyymmdd string), open, high,
  low, close, vol, amount  (hfq = adjusted prices; vwap is approximated by OHLC/4 because
  raw amount/vol are unadjusted and inconsistent with hfq prices);
- returns table (default ``forward_returns``): ts_code, trade_date, fwd_3/fwd_5/fwd_20/fwd_60,
  where trade_date is the signal date and the return accrues from the next day's entry
  (no look-ahead). The chosen column becomes the panel's target returns.

Universe modes:

- liquidity (default): top-N stocks by average daily amount with a minimum presence ratio;
- index PIT (``index_code`` set, e.g. ``000300.SH``): point-in-time index membership built
  from the monthly ``index_weight`` snapshots — each trading day uses the latest snapshot
  published on or before that day (forward-fill, no look-ahead). Prices are loaded for
  every stock that was ever a member in the window, and the membership mask is attached
  to the panel so evaluation/backtests are restricted to members per date.
"""

from __future__ import annotations

import io
import logging
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .base import MarketPanel

logger = logging.getLogger("alphamcts")


def _ch_query(url: str, user: str, password: str, sql: str, timeout: float = 120.0) -> str:
    params = urllib.parse.urlencode({"user": user, "password": password})
    req = urllib.request.Request(f"{url}/?{params}", data=sql.encode("utf-8"), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _ch_frame(url: str, user: str, password: str, sql: str) -> pd.DataFrame:
    text = _ch_query(url, user, password, sql + " FORMAT TSVWithNames")
    return pd.read_csv(io.StringIO(text), sep="\t")


def load_clickhouse_panel(cfg: dict) -> MarketPanel:
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8123))
    url = f"http://{host}:{port}"
    user = str(cfg.get("user", "default"))
    password = str(cfg.get("password", ""))
    database = str(cfg.get("database", "quant_system"))
    price_table = f"{database}.{cfg.get('price_table', 'daily_hfq')}"
    returns_table = f"{database}.{cfg.get('returns_table', 'forward_returns')}"
    target_column = str(cfg.get("target_column", "fwd_20"))
    start_date = str(cfg.get("start_date", "20180101")).replace("-", "")
    end_date = str(cfg.get("end_date", "20991231")).replace("-", "")
    top_n = int(cfg.get("universe_size", 800))
    min_presence = float(cfg.get("min_presence", 0.8))
    index_code = cfg.get("index_code")
    weight_table = f"{database}.{cfg.get('weight_table', 'index_weight')}"

    date_filter = f"trade_date >= '{start_date}' AND trade_date <= '{end_date}'"

    membership: pd.DataFrame | None = None
    if index_code:
        # ------------------------------------------------------------------
        # Point-in-time index universe from monthly snapshots
        # ------------------------------------------------------------------
        snaps = _ch_frame(
            url, user, password,
            f"""
            SELECT con_code AS ts_code, trade_date
            FROM {weight_table}
            WHERE index_code = '{index_code}'
              AND trade_date >= '{_shift_months_back(start_date, 2)}'
              AND trade_date <= '{end_date}'
            """,
        )
        if snaps.empty:
            raise ValueError(f"No {index_code} snapshots in {weight_table}")
        snaps["trade_date"] = pd.to_datetime(snaps["trade_date"], format="%Y%m%d")
        snaps["member"] = True
        # Each snapshot is the full constituent list: absent cells mean "not a member at
        # that snapshot" and must be False *before* the forward-fill, so stocks rotate out.
        membership = (
            snaps.pivot_table(index="trade_date", columns="ts_code", values="member", aggfunc="any")
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
        codes = membership.columns.tolist()
        logger.info(
            "ClickHouse PIT universe %s: %d snapshots (%s..%s), %d distinct members",
            index_code, membership.shape[0],
            membership.index[0].date(), membership.index[-1].date(), len(codes),
        )
    else:
        # ------------------------------------------------------------------
        # Liquidity universe: top-N by average daily amount with min coverage
        # ------------------------------------------------------------------
        n_days = int(_ch_query(
            url, user, password,
            f"SELECT uniqExact(trade_date) FROM {price_table} WHERE {date_filter}",
        ).strip())
        if n_days == 0:
            raise ValueError(f"No rows in {price_table} between {start_date} and {end_date}")

        universe_df = _ch_frame(
            url, user, password,
            f"""
            SELECT ts_code, avg(amount) AS avg_amount, count() AS n
            FROM {price_table}
            WHERE {date_filter}
            GROUP BY ts_code
            HAVING n >= {int(min_presence * n_days)}
            ORDER BY avg_amount DESC
            LIMIT {top_n}
            """,
        )
        codes = universe_df["ts_code"].tolist()
        if not codes:
            raise ValueError("Universe selection returned no instruments")
        logger.info("ClickHouse universe: %d instruments over %d days (%s..%s)",
                    len(codes), n_days, start_date, end_date)

    codes_sql = ",".join(f"'{c}'" for c in codes)

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    px = _ch_frame(
        url, user, password,
        f"""
        SELECT ts_code, trade_date, open, high, low, close, vol AS volume
        FROM {price_table}
        WHERE {date_filter} AND ts_code IN ({codes_sql})
        """,
    )
    px["trade_date"] = pd.to_datetime(px["trade_date"], format="%Y%m%d")

    fields: dict[str, pd.DataFrame] = {}
    for col in ("open", "high", "low", "close", "volume"):
        fields[col] = px.pivot(index="trade_date", columns="ts_code", values=col).sort_index()
    fields["vwap"] = (fields["open"] + fields["high"] + fields["low"] + fields["close"]) / 4.0

    # ------------------------------------------------------------------
    # Target forward returns (signal-date aligned, next-day entry -> no look-ahead)
    # ------------------------------------------------------------------
    fr = _ch_frame(
        url, user, password,
        f"""
        SELECT ts_code, trade_date, {target_column} AS target
        FROM {returns_table}
        WHERE {date_filter} AND ts_code IN ({codes_sql})
        """,
    )
    fr["trade_date"] = pd.to_datetime(fr["trade_date"], format="%Y%m%d")
    target = fr.pivot(index="trade_date", columns="ts_code", values="target").sort_index()
    target = target.replace([np.inf, -np.inf], np.nan)

    ref = fields["close"]
    target = target.reindex(index=ref.index, columns=ref.columns)
    coverage = float(target.notna().mean().mean())
    logger.info("Target %s coverage: %.1f%% of panel cells", target_column, coverage * 100)

    # ------------------------------------------------------------------
    # Optional industry + size exposures for cross-sectional neutralization
    # ------------------------------------------------------------------
    industry_panel: pd.DataFrame | None = None
    log_mv: pd.DataFrame | None = None
    if bool(cfg.get("neutralize", False)):
        mktcap_table = f"{database}.{cfg.get('mktcap_table', 'china_a_share_mktcap')}"
        industry_table = f"{database}.{cfg.get('industry_table', 'sw_industry_member')}"

        mv = _ch_frame(
            url, user, password,
            f"""
            SELECT ts_code, trade_date, circ_mv
            FROM {mktcap_table}
            WHERE {date_filter} AND ts_code IN ({codes_sql}) AND circ_mv > 0
            """,
        )
        mv["trade_date"] = pd.to_datetime(mv["trade_date"], format="%Y%m%d")
        log_mv = (
            np.log(mv.pivot(index="trade_date", columns="ts_code", values="circ_mv"))
            .sort_index()
            .reindex(index=ref.index, columns=ref.columns)
        )

        intervals = _ch_frame(
            url, user, password,
            f"""
            SELECT DISTINCT ts_code, l1_code, in_date, out_date
            FROM {industry_table}
            WHERE ts_code IN ({codes_sql})
            """,
        )
        # dates may arrive as ints/floats (empty out_date -> NaN); normalize to Timestamps
        intervals["in_date"] = pd.to_datetime(
            intervals["in_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
        intervals["out_date"] = pd.to_datetime(
            intervals["out_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
        industry_panel = pd.DataFrame("", index=ref.index, columns=ref.columns, dtype=object)
        for row in intervals.itertuples(index=False):
            if row.in_date is pd.NaT:
                continue
            # membership interval is [in_date, out_date): the stock belongs to the
            # industry up to, but excluding, the reclassification date
            sel = ref.index >= row.in_date
            if row.out_date is not pd.NaT:
                sel &= ref.index < row.out_date
            industry_panel.loc[sel, row.ts_code] = row.l1_code
        ind_coverage = float((industry_panel != "").mean().mean())
        logger.info(
            "Neutralization exposures: log_mv coverage %.1f%%, industry coverage %.1f%%",
            float(log_mv.notna().mean().mean()) * 100, ind_coverage * 100,
        )

        from .neutralize import neutralize_panel

        target = neutralize_panel(target, industry=industry_panel, log_mv=log_mv)
        logger.info("Target %s neutralized (industry + log size)", target_column)

    universe_mask: pd.DataFrame | None = None
    if membership is not None:
        # As-of membership: each trading day uses the latest snapshot published on or
        # before that day (forward-fill), so there is no look-ahead.
        universe_mask = (
            membership.reindex(membership.index.union(ref.index))
            .sort_index()
            .ffill()
            .reindex(ref.index)
            .reindex(columns=ref.columns)
            .fillna(False)
            .astype(bool)
        )
        daily_members = universe_mask.sum(axis=1)
        logger.info(
            "PIT membership per day: min=%d median=%d max=%d",
            int(daily_members.min()), int(daily_members.median()), int(daily_members.max()),
        )

    return MarketPanel(
        fields=fields,
        target_returns=target,
        universe_mask=universe_mask,
        industry=industry_panel,
        log_mv=log_mv,
        neutralize_factors=bool(cfg.get("neutralize", False)),
    )


def _shift_months_back(yyyymmdd: str, months: int) -> str:
    ts = pd.Timestamp(f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}") - pd.DateOffset(months=months)
    return ts.strftime("%Y%m%d")
