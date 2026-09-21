"""Combine a filtered subset of the zoo and compare against the full set.

Filter rule: drop factors whose validation-period RankIC flips negative (train is
positive by construction of the admission gates). This uses only train+valid
information, so the test period remains untouched out-of-sample.

Usage:
    python scripts/combo_subset.py --config configs/hs300.yaml --run runs/hs300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphamcts.config import load_config  # noqa: E402
from alphamcts.data.base import FIELDS, load_panel  # noqa: E402
from alphamcts.evaluation.metrics import mean_and_ir, rank_ic_series  # noqa: E402
from alphamcts.model.combiner import CombinerPipeline  # noqa: E402
from alphamcts.zoo import AlphaZoo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hs300.yaml")
    ap.add_argument("--run", default="runs/hs300")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg.get("data", {})
    horizon = int(data_cfg.get("horizon", 20))
    wr = tuple(cfg.get("window_range", [2, 60]))
    periods = {k: tuple(v) for k, v in data_cfg.get("periods", {}).items()}

    panel = load_panel(data_cfg)
    zoo = AlphaZoo.load(ROOT / args.run / "zoo.json", FIELDS, wr)

    fwd = panel.apply_universe(panel.forward_returns(horizon))
    valid_dates = panel.dates_between(*periods["valid"])

    keep, drop = [], []
    for rec in zoo.records:
        values = panel.apply_universe(
            panel.maybe_neutralize_factor(rec.formula.evaluate(panel.fields, rec.args))
        )
        ric_valid, _ = mean_and_ir(rank_ic_series(values.loc[valid_dates], fwd.loc[valid_dates]))
        (keep if ric_valid > 0 else drop).append(rec)
    print(f"kept {len(keep)} / {len(zoo.records)} (dropped {len(drop)} valid-flipped factors)\n")

    eval_periods = {k: v for k, v in periods.items() if k != "train"}
    pipe = CombinerPipeline(
        panel,
        horizon=horizon,
        combiner_cfg=cfg.get("combiner", {}),
        backtest_cfg=cfg.get("backtest", {}),
        train_range=periods["train"],
        eval_periods=eval_periods,
    )

    for label, records in [("full zoo", list(zoo.records)), ("no-valid-flip subset", keep)]:
        for model_name in ("equal", "ewma"):
            model, features = pipe.fit(model_name, records)
            for period_name, dates in pipe.eval_periods.items():
                r = pipe.evaluate_period(model, features, dates, model_name, len(records))
                print(f"[{label:>20} | {model_name:>5} | {period_name:>5}] "
                      f"RankIC={r.rank_ic:+.4f}  AER={r.aer:+.4f}  IR={r.ir:+.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
