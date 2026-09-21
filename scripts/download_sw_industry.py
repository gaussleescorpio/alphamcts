"""Download Shenwan (SW) industry classification + historical membership from tushare
and store them in ClickHouse for point-in-time industry lookups.

Tables created (database quant_system):

- sw_industry_classify: index_code, industry_name, level (L1/L2/L3), parent_code, src
- sw_industry_member:   l1_code, l1_name, l2_code, l2_name, l3_code, l3_name,
                        ts_code, name, in_date, out_date ('' = still a member), is_new

Membership rows are (in_date, out_date] intervals, so the industry of any stock on any
historical date can be resolved without look-ahead bias:

    SELECT l1_code FROM sw_industry_member
    WHERE ts_code = 'X' AND in_date <= D AND (out_date = '' OR out_date > D)

Requires TUSHARE_TOKEN in the environment (or in .env at the project root).

Usage:
    python scripts/download_sw_industry.py [--src SW2021] [--drop]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

CH_URL = "http://127.0.0.1:8123/"
CH_AUTH = ("gauss", "123456")
DB = "quant_system"


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def ch(query: str, data: str | None = None) -> str:
    resp = requests.post(
        CH_URL, params={"query": query, "database": DB}, data=data, auth=CH_AUTH, timeout=60
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ClickHouse error {resp.status_code}: {resp.text[:500]}")
    return resp.text


def insert_df(table: str, df: pd.DataFrame) -> None:
    df = df.fillna("")
    payload = df.to_json(orient="records", lines=True, force_ascii=False)
    ch(f"INSERT INTO {table} FORMAT JSONEachRow", data=payload.encode("utf-8"))


def fetch_with_retry(fn, retries: int = 5, wait: float = 30.0, **kwargs) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = fn(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # tushare raises plain Exception on rate limits
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(wait)
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="SW2021", help="classification source (SW2021 / SW2014)")
    ap.add_argument("--drop", action="store_true", help="drop and recreate the target tables")
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        print("TUSHARE_TOKEN is not set (env or .env).")
        return 1

    import tushare as ts

    pro = ts.pro_api(token)

    if args.drop:
        ch("DROP TABLE IF EXISTS sw_industry_classify")
        ch("DROP TABLE IF EXISTS sw_industry_member")

    ch(
        """CREATE TABLE IF NOT EXISTS sw_industry_classify (
            index_code String, industry_name String, level String,
            parent_code String, src String
        ) ENGINE = ReplacingMergeTree ORDER BY (src, level, index_code)"""
    )
    ch(
        """CREATE TABLE IF NOT EXISTS sw_industry_member (
            l1_code String, l1_name String, l2_code String, l2_name String,
            l3_code String, l3_name String, ts_code String, name String,
            in_date String, out_date String, is_new String
        ) ENGINE = ReplacingMergeTree ORDER BY (ts_code, l3_code, in_date)"""
    )

    # 1. classification catalog, all three levels
    frames = []
    for level in ("L1", "L2", "L3"):
        df = fetch_with_retry(pro.index_classify, level=level, src=args.src)
        print(f"index_classify {level}: {len(df)} rows")
        frames.append(df)
        time.sleep(1)
    classify = pd.concat(frames, ignore_index=True)
    cols = ["index_code", "industry_name", "level", "parent_code", "src"]
    for c in cols:
        if c not in classify.columns:
            classify[c] = args.src if c == "src" else ""
    insert_df("sw_industry_classify", classify[cols])

    # 2. historical membership per L1 industry (in_date/out_date intervals).
    # is_new='Y' returns current members (open intervals), is_new='N' returns the
    # historical records of stocks that left / were reclassified (closed intervals);
    # both are needed for point-in-time lookups.
    l1 = classify[classify["level"] == "L1"]
    total = 0
    for _, row in l1.iterrows():
        code, name = row["index_code"], row["industry_name"]
        for flag in ("Y", "N"):
            df = fetch_with_retry(pro.index_member_all, l1_code=code, is_new=flag)
            total += len(df)
            print(f"index_member_all {code} {name} is_new={flag}: {len(df)} rows (total {total})")
            if not df.empty:
                cols_m = ["l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
                          "ts_code", "name", "in_date", "out_date", "is_new"]
                for c in cols_m:
                    if c not in df.columns:
                        df[c] = ""
                insert_df("sw_industry_member", df[cols_m])
            time.sleep(1)  # stay under the per-minute API quota

    # 3. sanity checks
    print("\n--- verification ---")
    print(ch(
        "SELECT count(), uniqExact(ts_code), min(in_date), "
        "countIf(out_date = '') FROM sw_industry_member"
    ).strip(), "  (rows / stocks / earliest in_date / open intervals)")
    print(ch(
        "SELECT count() FROM sw_industry_member WHERE in_date <= '20180102' "
        "AND (out_date = '' OR out_date > '20180102')"
    ).strip(), "  (members on 2018-01-02)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
