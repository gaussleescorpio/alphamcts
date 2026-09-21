"""Long-short quantile decomposition report for a mined zoo.

For every alpha in the zoo and every configured period (train/valid/test), prints the
Q1..Q5 mean forward returns (quintiles by signal), the Q5-Q1 spread, the monotonic
fraction and the annualized Sharpe of the cost-free daily Q5-Q1 portfolio (on raw
next-day returns). Factors passing the training-period gates (spread > 0,
monotonicity >= --min-mono, LS Sharpe >= --min-ls-sharpe) are then combined
equal-weight and the combo is re-evaluated on valid/test, including the periodic
top-k backtest.

Usage:
    python scripts/longshort_report.py --config configs/hs300.yaml \
        --run runs/hs300_neutral [--min-ls-sharpe 1.0] [--min-mono 0.75]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphamcts.config import load_config  # noqa: E402
from alphamcts.data.base import FIELDS, load_panel  # noqa: E402
from alphamcts.evaluation.longshort import (  # noqa: E402
    long_short_sharpe,
    monotonic_fraction,
    quantile_mean_returns,
)
from alphamcts.evaluation.metrics import mean_and_ir, rank_ic_series  # noqa: E402
from alphamcts.model.combiner import CombinerPipeline  # noqa: E402
from alphamcts.zoo import AlphaZoo  # noqa: E402


def decompose(pred: pd.DataFrame, fwd: pd.DataFrame, ret1: pd.DataFrame) -> dict:
    qmeans = quantile_mean_returns(pred, fwd)
    rank_ic, _ = mean_and_ir(rank_ic_series(pred, fwd))
    return {
        "qmeans": qmeans,
        "spread": float(qmeans.iloc[-1] - qmeans.iloc[0]),
        "mono": monotonic_fraction(qmeans),
        "ls_sharpe": long_short_sharpe(pred, ret1),
        "rank_ic": rank_ic,
    }


def fmt_row(name: str, d: dict) -> str:
    q = " ".join(f"{v * 100:+6.2f}" for v in d["qmeans"])
    return (f"  {name:<6} Q1..Q5[%]: {q}   spread={d['spread'] * 100:+.2f}%  "
            f"mono={d['mono']:.2f}  LS_Sharpe={d['ls_sharpe']:+.2f}  RankIC={d['rank_ic']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hs300.yaml")
    ap.add_argument("--run", default="runs/hs300_neutral")
    ap.add_argument("--min-ls-sharpe", type=float, default=1.0)
    ap.add_argument("--min-mono", type=float, default=0.75)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg.get("data", {})
    horizon = int(data_cfg.get("horizon", 10))
    wr = tuple(cfg.get("window_range", [2, 60]))
    periods = {k: tuple(v) for k, v in data_cfg.get("periods", {}).items()}

    panel = load_panel(data_cfg)
    zoo = AlphaZoo.load(ROOT / args.run / "zoo.json", FIELDS, wr)

    fwd = panel.apply_universe(panel.forward_returns(horizon))
    ret1 = panel.apply_universe(panel.fields["close"].pct_change().shift(-1))
    period_dates = {name: panel.dates_between(lo, hi) for name, (lo, hi) in periods.items()}

    survivors = []
    print(f"\n=== Per-alpha long-short decomposition ({len(zoo)} alphas) ===")
    for i, rec in enumerate(zoo.records, 1):
        values = panel.apply_universe(
            panel.maybe_neutralize_factor(rec.formula.evaluate(panel.fields, rec.args))
        )
        print(f"\n#{i} {rec.expression[:100]}")
        stats = {}
        for name, dates in period_dates.items():
            stats[name] = decompose(values.loc[dates], fwd.loc[dates], ret1.loc[dates])
            print(fmt_row(name, stats[name]))
        tr = stats["train"]
        ok = (tr["spread"] > 0 and tr["mono"] >= args.min_mono
              and tr["ls_sharpe"] >= args.min_ls_sharpe)
        print(f"  train gates (spread>0, mono>={args.min_mono}, "
              f"LS_Sharpe>={args.min_ls_sharpe}): {'PASS' if ok else 'FAIL'}")
        if ok:
            survivors.append(rec)

    print(f"\n=== Survivors: {len(survivors)} / {len(zoo)} ===")
    for rec in survivors:
        print(f"  {rec.expression[:100]}")
    if not survivors:
        print("No survivors; nothing to combine.")
        return 0

    eval_periods = {k: v for k, v in periods.items() if k != "train"}
    pipe = CombinerPipeline(
        panel,
        horizon=horizon,
        combiner_cfg=cfg.get("combiner", {}),
        backtest_cfg=cfg.get("backtest", {}),
        train_range=periods["train"],
        eval_periods=eval_periods,
    )
    model, features = pipe.fit("equal", survivors)

    print("\n=== Equal-weight combo of survivors ===")
    for name in ("train", *eval_periods):
        dates = period_dates[name]
        pred = pipe.predict_panel(model, features, dates)
        d = decompose(pred, fwd.loc[dates], ret1.loc[dates])
        print(fmt_row(name, d))
    for name in eval_periods:
        report = pipe.evaluate_period(model, features, period_dates[name], "equal", len(survivors))
        print(f"  [{name}] periodic backtest: RankIC={report.rank_ic:+.4f}  "
              f"AER={report.aer:+.4f}  IR={report.ir:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
