"""Diagnose why OOS RankIC is positive while long-only excess return is negative.

Decomposes the gap for the equal-weight combination of a mined zoo:

1. Decile analysis of the prediction vs (a) neutralized forward returns (what the
   factors are trained to predict) and (b) raw forward returns (what a long-only
   portfolio actually earns) - reveals short-side concentration and the
   neutralized->raw conversion gap.
2. Long-side arithmetic: top-decile raw return vs the universe mean.
3. Cost drag: backtest with and without transaction costs.

Usage:
    python scripts/diagnose_ic_vs_aer.py --config configs/hs300.yaml \
        --zoo runs/hs300_neutral/zoo.json --period test
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphamcts.config import load_config  # noqa: E402
from alphamcts.data.base import FIELDS, load_panel  # noqa: E402
from alphamcts.model.combiner import CombinerPipeline  # noqa: E402
from alphamcts.zoo import AlphaZoo  # noqa: E402


def decile_table(pred: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 10) -> pd.Series:
    """Average forward return per prediction decile (10 = best-ranked)."""
    ranks = pred.rank(axis=1, pct=True)
    bucket = np.ceil(ranks * n_q).clip(1, n_q)
    rows = []
    for q in range(1, n_q + 1):
        rows.append(fwd.where(bucket == q).stack().mean())
    return pd.Series(rows, index=range(1, n_q + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hs300.yaml")
    ap.add_argument("--zoo", default="runs/hs300_neutral/zoo.json")
    ap.add_argument("--period", default="test")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg.get("data", {})
    horizon = int(data_cfg.get("horizon", 10))
    wr = tuple(cfg.get("window_range", [2, 60]))
    periods = {k: tuple(v) for k, v in data_cfg.get("periods", {}).items()}

    panel = load_panel(data_cfg)  # neutralized target & factors per config
    raw_cfg = copy.deepcopy(data_cfg)
    raw_cfg["clickhouse"]["neutralize"] = False
    panel_raw = load_panel(raw_cfg)

    zoo = AlphaZoo.load(ROOT / args.zoo, FIELDS, wr)
    records = list(zoo.records)

    pipe = CombinerPipeline(
        panel, horizon=horizon, backtest_cfg=cfg.get("backtest", {}),
        train_range=periods["train"],
        eval_periods={k: v for k, v in periods.items() if k != "train"},
    )
    model, features = pipe.fit("equal", records)
    dates = pipe.eval_periods[args.period]
    pred = pipe.predict_panel(model, features, dates)

    fwd_neu = panel.apply_universe(panel.forward_returns(horizon)).loc[dates]
    fwd_raw = panel_raw.apply_universe(panel_raw.forward_returns(horizon)).loc[dates]

    print(f"\n=== {args.period}: decile mean {horizon}-day forward returns "
          f"(1 = worst-ranked, 10 = best-ranked) ===")
    tbl = pd.DataFrame({
        "neutralized": decile_table(pred, fwd_neu),
        "raw": decile_table(pred, fwd_raw),
    })
    print((tbl * 100).round(3).to_string())

    universe_mean = fwd_raw.stack().mean()
    top = tbl.loc[10, "raw"]
    bottom = tbl.loc[1, "raw"]
    print(f"\nuniverse mean raw fwd_{horizon}:  {universe_mean * 100:.3f}%")
    print(f"top-decile raw   - universe:  {(top - universe_mean) * 100:+.3f}%")
    print(f"bottom-decile raw - universe: {(bottom - universe_mean) * 100:+.3f}%")
    print(f"long-short (10 - 1) raw:      {(top - bottom) * 100:+.3f}%")

    print()
    for cost in (0.0015, 0.0):
        pipe.bt_cfg = dict(cfg.get("backtest", {}), cost=cost)
        aer_d, ir_d, _ = pipe.backtest_topk_dropn(pred)
        aer_p, ir_p, _ = pipe.backtest_periodic(pred)
        print(f"cost={cost:.4f}  daily topk_dropn: AER={aer_d:+.4f} IR={ir_d:+.3f}   "
              f"periodic {horizon}d rebalance: AER={aer_p:+.4f} IR={ir_p:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
