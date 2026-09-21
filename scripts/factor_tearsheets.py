"""Alphalens tear sheets for every alpha in a mined zoo.

For each factor, produces one PNG with a 3x2 grid:

- row 1: cumulative long-short (Q5 - Q1, daily) return curve;
- row 2: cumulative quantile returns (5 layers, demeaned vs universe mean, daily);
- row 3: 20-day RankIC: 60-day rolling mean, with period mean IC / ICIR annotated;
- row 4: cumulative IC "equity curve": daily 20d RankIC accumulated like returns
  (net-value style, analogous to the LS backtest curve); the slope is the mean IC and
  the smoothness reflects the ICIR. The annotated period ICIR is computed on
  non-overlapping (every `horizon` days) IC observations;

columns: in-sample (train + valid) vs out-of-sample (test). Factor values are
neutralized + PIT-universe-masked (identical to the mining pipeline); prices are
raw hfq closes, so the curves live in tradable space.

Usage:
    python scripts/factor_tearsheets.py --config configs/hs300.yaml --run runs/hs300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphalens.performance import (  # noqa: E402
    compute_mean_returns_spread,
    factor_information_coefficient,
    mean_return_by_quantile,
)
from alphalens.utils import get_clean_factor_and_forward_returns  # noqa: E402

from alphamcts.config import load_config  # noqa: E402
from alphamcts.data.base import FIELDS, load_panel  # noqa: E402
from alphamcts.zoo import AlphaZoo  # noqa: E402

ANN = 252


def alphalens_data(values: pd.DataFrame, prices: pd.DataFrame, start: str, end: str,
                   horizon: int):
    """Slice the factor to [start, end] and build the alphalens factor-data frame."""
    window = values.loc[str(start):str(end)]
    factor = window.stack().dropna()
    factor.index = factor.index.set_names(["date", "asset"])
    return get_clean_factor_and_forward_returns(
        factor, prices, quantiles=5, periods=(1, horizon), max_loss=0.9,
    )


def panel_stats(fd, horizon: int) -> dict:
    """LS series (1D), quantile cum curves, IC series and summary scalars."""
    per_col = f"{horizon}D"
    ic = factor_information_coefficient(fd)[per_col].dropna()
    mean_ic = float(ic.mean())
    icir = float(ic.mean() / ic.std()) if ic.std() > 1e-12 else float("nan")
    # non-overlapping IC observations: fwd_20 windows sampled every `horizon` days
    ic_gap = ic.iloc[::horizon]
    icir_gap = float(ic_gap.mean() / ic_gap.std()) if ic_gap.std() > 1e-12 else float("nan")

    mrq, _ = mean_return_by_quantile(fd, by_date=True, demeaned=True)
    spread, _ = compute_mean_returns_spread(mrq, upper_quant=5, lower_quant=1)
    ls_1d = spread["1D"].dropna()
    ls_sharpe = float(ls_1d.mean() / ls_1d.std() * ANN ** 0.5) if ls_1d.std() > 1e-12 else float("nan")

    q_cum = {}
    for q in sorted(mrq.index.get_level_values("factor_quantile").unique()):
        r = mrq.loc[q, "1D"].dropna()
        q_cum[int(q)] = (1.0 + r).cumprod() - 1.0

    return {
        "ic": ic, "mean_ic": mean_ic, "icir": icir,
        "ic_gap": ic_gap, "icir_gap": icir_gap,
        "ls_cum": (1.0 + ls_1d).cumprod() - 1.0, "ls_sharpe": ls_sharpe,
        "q_cum": q_cum,
    }


def plot_factor(idx: int, name: str, expr: str, stats_is: dict, stats_oos: dict,
                out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(f"#{idx} {name}\n{expr[:110]}", fontsize=10)

    for col, (label, st) in enumerate([("In-sample (train+valid)", stats_is),
                                       ("Out-of-sample (test)", stats_oos)]):
        ax = axes[0][col]
        st["ls_cum"].plot(ax=ax, color="tab:blue")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{label} | LS (Q5-Q1) cum, Sharpe={st['ls_sharpe']:.2f}", fontsize=9)
        ax.set_ylabel("cum return")

        ax = axes[1][col]
        for q, curve in st["q_cum"].items():
            curve.plot(ax=ax, label=f"Q{q}", lw=1.1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{label} | quantile cum (demeaned)", fontsize=9)
        ax.legend(fontsize=7, ncol=5)

        ax = axes[2][col]
        ic = st["ic"]
        ic.rolling(60, min_periods=20).mean().plot(ax=ax, color="tab:green")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(st["mean_ic"], color="tab:red", lw=0.8, ls="--")
        ax.set_title(f"{label} | 60d rolling RankIC, mean={st['mean_ic']:.4f}, "
                     f"ICIR={st['icir']:.2f}", fontsize=9)

        ax = axes[3][col]
        ic.cumsum().plot(ax=ax, color="tab:purple")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(
            f"{label} | cumulative IC (net-value style), mean={st['mean_ic']:.4f}, "
            f"non-overlap ICIR={st['icir_gap']:.2f}", fontsize=9)
        ax.set_ylabel("cum IC")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
    is_range = (periods["train"][0], periods["valid"][1])
    oos_range = periods["test"]

    panel = load_panel(data_cfg)
    zoo = AlphaZoo.load(ROOT / args.run / "zoo.json", FIELDS, wr)
    prices = panel.fields["close"]

    out_dir = ROOT / args.run / "tearsheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rec in enumerate(zoo.records, 1):
        values = panel.apply_universe(
            panel.maybe_neutralize_factor(rec.formula.evaluate(panel.fields, rec.args))
        )
        try:
            fd_is = alphalens_data(values, prices, *is_range, horizon=horizon)
            fd_oos = alphalens_data(values, prices, *oos_range, horizon=horizon)
            st_is = panel_stats(fd_is, horizon)
            st_oos = panel_stats(fd_oos, horizon)
        except Exception as exc:  # noqa: BLE001 - report and continue with next factor
            print(f"#{i} {rec.name}: alphalens failed ({type(exc).__name__}: {exc})")
            continue

        out_path = out_dir / f"{i:02d}_{rec.name[:40]}.png"
        plot_factor(i, rec.name, rec.expression, st_is, st_oos, out_path)
        rows.append({
            "idx": i, "name": rec.name,
            "IS_ic": st_is["mean_ic"], "IS_icir": st_is["icir"],
            "IS_icir_gap": st_is["icir_gap"], "IS_ls_sharpe": st_is["ls_sharpe"],
            "OOS_ic": st_oos["mean_ic"], "OOS_icir": st_oos["icir"],
            "OOS_icir_gap": st_oos["icir_gap"], "OOS_ls_sharpe": st_oos["ls_sharpe"],
        })
        print(f"#{i:2d} {rec.name[:38]:<38} "
              f"IS: IC={st_is['mean_ic']:+.4f} ICIR={st_is['icir']:+.2f} LS={st_is['ls_sharpe']:+.2f} | "
              f"OOS: IC={st_oos['mean_ic']:+.4f} ICIR={st_oos['icir']:+.2f} LS={st_oos['ls_sharpe']:+.2f}")

    summary = pd.DataFrame(rows).set_index("idx")
    summary.to_csv(out_dir / "summary.csv")
    print(f"\nSaved {len(rows)} tear sheets + summary.csv -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
