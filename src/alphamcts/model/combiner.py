"""Downstream alpha combination and portfolio backtest.

Follows the paper's Experiment 1 protocol:

- select the top-k alphas by RankIR from the zoo (k = 10/50/100);
- cross-sectionally rank-normalize alpha values and target returns;
- train a combination model (LightGBM, and an MLP - PyTorch if available, otherwise a
  scikit-learn fallback) on the training period;
- evaluate predictions on the held-out test period with IC / RankIC;
- run a top-k/drop-n long-only backtest with transaction costs and report Annualized
  Excess Return (AER) and Information Ratio (IR) against the equal-weight universe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.base import MarketPanel
from ..evaluation.metrics import cs_rank, ic_series, rank_ic_series
from ..zoo import AlphaRecord

logger = logging.getLogger("alphamcts")

TRADING_DAYS = 252


@dataclass
class BacktestReport:
    model: str
    n_alphas: int
    ic: float
    rank_ic: float
    aer: float
    ir: float
    daily_excess: pd.Series = field(repr=False, default_factory=pd.Series)

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "n_alphas": self.n_alphas,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "aer": self.aer,
            "ir": self.ir,
        }


def _rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [-0.5, 0.5]; NaN -> 0 (neutral)."""
    return (cs_rank(df) - 0.5).fillna(0.0)


class _EqualWeight:
    """Training-free combiner: the equal-weight average of rank-normalized alphas."""

    def predict(self, arr: np.ndarray) -> np.ndarray:
        return arr.mean(axis=1)


class _PrecomputedPred:
    """Combiner whose full prediction panel is precomputed (e.g. EWMA-IC weighting).

    `weights` holds the per-date factor weights actually used, for decay reporting.
    """

    def __init__(self, pred: pd.DataFrame, weights: pd.DataFrame):
        self.pred = pred
        self.weights = weights


class CombinerPipeline:
    def __init__(
        self,
        panel: MarketPanel,
        horizon: int,
        train_ratio: float = 0.7,
        combiner_cfg: dict | None = None,
        backtest_cfg: dict | None = None,
        train_range: tuple[str, str] | None = None,
        eval_periods: dict[str, tuple[str, str]] | None = None,
    ):
        self.panel = panel
        self.horizon = horizon
        self.cfg = combiner_cfg or {}
        self.bt_cfg = backtest_cfg or {}
        if train_range is not None:
            self.train_dates = panel.dates_between(*train_range)
        else:
            mask = panel.train_mask(train_ratio)
            self.train_dates = mask[mask].index
        if eval_periods is not None:
            self.eval_periods = {
                name: panel.dates_between(*rng) for name, rng in eval_periods.items()
            }
        else:
            mask = panel.train_mask(train_ratio)
            self.eval_periods = {"test": mask[~mask].index}
        self.test_dates = next(iter(self.eval_periods.values()))
        self.fwd = panel.apply_universe(panel.forward_returns(horizon))

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def compute_features(self, records: list[AlphaRecord]) -> list[pd.DataFrame]:
        feats = []
        for r in records:
            values = self.panel.maybe_neutralize_factor(r.formula.evaluate(self.panel.fields, r.args))
            feats.append(_rank_normalize(self.panel.apply_universe(values)))
        return feats

    def _dataset(self, features: list[pd.DataFrame], dates: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stacked (samples x alphas) matrix, rank-normalized targets and a validity mask."""
        target = _rank_normalize(self.fwd)
        y_raw = self.fwd.loc[dates]
        x = np.stack([f.loc[dates].to_numpy().ravel() for f in features], axis=1)
        y = target.loc[dates].to_numpy().ravel()
        valid = ~np.isnan(y_raw.to_numpy().ravel())
        return x[valid], y[valid], valid

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _fit_lightgbm(self, x: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        p = self.cfg.get("lightgbm", {})
        model = lgb.LGBMRegressor(
            num_leaves=int(p.get("num_leaves", 32)),
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 8)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            reg_alpha=float(p.get("reg_alpha", 0.1)),
            reg_lambda=float(p.get("reg_lambda", 0.1)),
            verbose=-1,
        )
        model.fit(x, y)
        return model

    def _fit_mlp(self, x: np.ndarray, y: np.ndarray):
        p = self.cfg.get("mlp", {})
        hidden = tuple(int(h) for h in p.get("hidden", (256, 128, 64)))
        try:
            return self._fit_mlp_torch(x, y, p, hidden)
        except ImportError:
            logger.info("PyTorch not available; using scikit-learn MLPRegressor (no dropout).")
            from sklearn.neural_network import MLPRegressor

            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                learning_rate_init=float(p.get("lr", 0.001)),
                batch_size=min(int(p.get("batch_size", 1024)), max(len(x) // 4, 1)),
                max_iter=int(p.get("max_epochs", 50)),
                early_stopping=True,
                n_iter_no_change=int(p.get("patience", 5)),
                random_state=0,
            )
            model.fit(x, y)
            return model

    def _fit_mlp_torch(self, x: np.ndarray, y: np.ndarray, p: dict, hidden: tuple[int, ...]):
        import torch
        from torch import nn

        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layers: list[nn.Module] = []
        dims = (x.shape[1],) + hidden
        dropout = float(p.get("dropout", 0.3))
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        net = nn.Sequential(*layers).to(device)

        n_val = max(int(len(x) * 0.1), 1)
        x_tr, y_tr = x[:-n_val], y[:-n_val]
        x_val = torch.tensor(x[-n_val:], dtype=torch.float32, device=device)
        y_val = torch.tensor(y[-n_val:], dtype=torch.float32, device=device)

        opt = torch.optim.Adam(net.parameters(), lr=float(p.get("lr", 0.001)))
        loss_fn = nn.MSELoss()
        batch = int(p.get("batch_size", 1024))
        patience = int(p.get("patience", 5))
        best_val, best_state, bad = float("inf"), None, 0

        for _epoch in range(int(p.get("max_epochs", 50))):
            perm = np.random.permutation(len(x_tr))
            net.train()
            for i in range(0, len(x_tr), batch):
                idx = perm[i : i + batch]
                xb = torch.tensor(x_tr[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(y_tr[idx], dtype=torch.float32, device=device)
                opt.zero_grad()
                loss = loss_fn(net(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(x_val).squeeze(-1), y_val))
            if val < best_val - 1e-6:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

        class _TorchWrapper:
            def predict(self, arr: np.ndarray) -> np.ndarray:
                net.eval()
                with torch.no_grad():
                    t = torch.tensor(arr, dtype=torch.float32, device=device)
                    return net(t).squeeze(-1).cpu().numpy()

        return _TorchWrapper()

    # ------------------------------------------------------------------
    # Prediction + evaluation
    # ------------------------------------------------------------------

    def combine_ewma(self, features: list[pd.DataFrame]) -> _PrecomputedPred:
        """Walk-forward EWMA-IC weighted combination (dynamic weighting + decay pause).

        Weight of factor i at date t = EWMA(halflife) of its daily RankIC series,
        lagged by horizon+1 days (the return of the IC observed at s only fully
        realizes at s+horizon with next-day entry, so it is usable at s+horizon+1),
        clipped at zero (a factor with non-positive recent IC is paused) and
        normalized across factors. Falls back to equal weight while no factor has
        positive recent IC (including the warm-up period).
        """
        p = self.cfg.get("ewma", {})
        halflife = float(p.get("halflife", 63))
        min_periods = int(p.get("min_periods", 60))
        lag = self.horizon + 1

        ics = pd.concat(
            [rank_ic_series(f, self.fwd) for f in features], axis=1
        )
        ics.columns = range(len(features))
        raw = ics.ewm(halflife=halflife, min_periods=min_periods).mean().shift(lag)
        weights = raw.clip(lower=0.0)
        total = weights.sum(axis=1)
        weights = weights.div(total.where(total > 0), axis=0)
        weights = weights.fillna(1.0 / len(features))  # warm-up / all-paused fallback

        pred = sum(f.mul(weights[i], axis=0).fillna(0.0) for i, f in enumerate(features))
        tradable = self.panel.fields["close"].notna()
        pred = self.panel.apply_universe(pred.where(tradable))
        return _PrecomputedPred(pred, weights)

    def predict_panel(self, model, features: list[pd.DataFrame], dates: pd.Index) -> pd.DataFrame:
        if isinstance(model, _PrecomputedPred):
            return model.pred.loc[dates]
        x = np.stack([f.loc[dates].to_numpy().ravel() for f in features], axis=1)
        pred = model.predict(x).reshape(len(dates), -1)
        pred_df = pd.DataFrame(pred, index=dates, columns=self.panel.instruments)
        tradable = self.panel.fields["close"].loc[dates].notna()
        return self.panel.apply_universe(pred_df.where(tradable))

    def fit(self, model_name: str, records: list[AlphaRecord]):
        """Train a combination model on the training period; returns (model, features)."""
        features = self.compute_features(records)
        if model_name == "equal":
            return _EqualWeight(), features
        if model_name == "ewma":
            return self.combine_ewma(features), features
        x_tr, y_tr, _ = self._dataset(features, self.train_dates)
        if model_name == "lightgbm":
            model = self._fit_lightgbm(x_tr, y_tr)
        elif model_name == "mlp":
            model = self._fit_mlp(x_tr, y_tr)
        else:
            raise ValueError(f"Unknown combiner model: {model_name!r}")
        return model, features

    def evaluate_period(
        self,
        model,
        features: list[pd.DataFrame],
        dates: pd.Index,
        model_name: str,
        n_alphas: int,
    ) -> BacktestReport:
        pred = self.predict_panel(model, features, dates)
        fwd = self.fwd.loc[dates]
        ic = float(ic_series(pred, fwd).mean())
        rank_ic = float(rank_ic_series(pred, fwd).mean())
        if str(self.bt_cfg.get("mode", "topk_dropn")) == "periodic":
            aer, ir, daily_excess = self.backtest_periodic(pred)
        else:
            aer, ir, daily_excess = self.backtest_topk_dropn(pred)
        return BacktestReport(
            model=model_name,
            n_alphas=n_alphas,
            ic=ic,
            rank_ic=rank_ic,
            aer=aer,
            ir=ir,
            daily_excess=daily_excess,
        )

    def run_model(self, model_name: str, records: list[AlphaRecord]) -> BacktestReport:
        model, features = self.fit(model_name, records)
        return self.evaluate_period(model, features, self.test_dates, model_name, len(records))

    # ------------------------------------------------------------------
    # Periodic full rebalance: hold top-k for `horizon` days, then re-form.
    # Matches the prediction horizon (an h-day signal is traded every h days),
    # so costs are only paid on actual holding replacement at rebalance dates.
    # ------------------------------------------------------------------

    def backtest_periodic(self, pred: pd.DataFrame) -> tuple[float, float, pd.Series]:
        top_frac = float(self.bt_cfg.get("top_frac", 0.1))
        cost = float(self.bt_cfg.get("cost", 0.0015))
        n_effective = int(pred.notna().sum(axis=1).median())
        k = max(1, int(round(top_frac * n_effective)))

        daily_ret = self.panel.daily_returns()
        bench_ret = self.panel.apply_universe(daily_ret)
        dates = pred.index
        holdings: list[str] = []
        excess: dict[pd.Timestamp, float] = {}
        pending_cost = 0.0

        for i, date in enumerate(dates):
            if i > 0:
                rets = daily_ret.loc[date]
                universe = bench_ret.loc[date].dropna()
                port_ret = float(rets.reindex(holdings).mean()) if holdings else 0.0
                bench = float(universe.mean()) if not universe.empty else 0.0
                if port_ret != port_ret:
                    port_ret = bench  # holdings temporarily untradable -> assume benchmark
                excess[date] = port_ret - bench - pending_cost
                pending_cost = 0.0
            if i % self.horizon != 0:
                continue
            scores = pred.loc[date].dropna()
            if scores.empty:
                continue
            target = list(scores.sort_values(ascending=False).index[:k])
            if not holdings:
                pending_cost = cost  # initial full buy
            else:
                replaced = len(set(holdings) - set(target))
                pending_cost = cost * 2.0 * replaced / max(len(target), 1)  # sell + buy legs
            holdings = target

        series = pd.Series(excess).sort_index()
        if series.empty:
            return float("nan"), float("nan"), series
        aer = float(series.mean()) * TRADING_DAYS
        std = float(series.std())
        ir = aer / (std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else float("nan")
        return aer, ir, series

    # ------------------------------------------------------------------
    # Top-k / drop-n backtest (paper Appendix G "Backtesting Strategy")
    # ------------------------------------------------------------------

    def backtest_topk_dropn(self, pred: pd.DataFrame) -> tuple[float, float, pd.Series]:
        top_frac = float(self.bt_cfg.get("top_frac", 0.1))
        cost = float(self.bt_cfg.get("cost", 0.0015))
        # size the book off the effective per-day universe (PIT-masked), not total columns
        n_effective = int(pred.notna().sum(axis=1).median())
        k = max(1, int(round(top_frac * n_effective)))
        n_trade = max(1, int(round(k / self.horizon)))

        daily_ret = self.panel.daily_returns()
        bench_ret = self.panel.apply_universe(daily_ret)
        dates = pred.index
        holdings: list[str] = []
        excess: dict[pd.Timestamp, float] = {}
        pending_cost = 0.0

        for i, date in enumerate(dates):
            scores = pred.loc[date].dropna()
            if i > 0:
                rets = daily_ret.loc[date]
                universe = bench_ret.loc[date].dropna()
                port_ret = float(rets.reindex(holdings).mean()) if holdings else 0.0
                bench = float(universe.mean()) if not universe.empty else 0.0
                if port_ret != port_ret:
                    port_ret = bench  # holdings temporarily untradable -> assume benchmark
                excess[date] = port_ret - bench - pending_cost
                pending_cost = 0.0
            if scores.empty:
                continue
            ranked = scores.sort_values(ascending=False)
            target = list(ranked.index[:k])
            if not holdings:
                holdings = target
                pending_cost = cost  # initial full buy: cost on 100% of the book
                continue
            target_set = set(target)
            # sell up to n_trade held names that fell out of the top-k, worst-scored first
            held_scores = ranked.reindex(holdings)
            sell_candidates = [s for s in held_scores.sort_values(ascending=True, na_position="first").index
                               if s not in target_set]
            sells = sell_candidates[:n_trade]
            # buy the best-ranked names not currently held
            buy_candidates = [s for s in ranked.index if s not in holdings]
            buys = buy_candidates[: len(sells)]
            if sells and buys:
                holdings = [h for h in holdings if h not in set(sells)] + buys
                traded_frac = (len(sells) + len(buys)) / max(len(holdings), 1)
                pending_cost = cost * traded_frac

        series = pd.Series(excess).sort_index()
        if series.empty:
            return float("nan"), float("nan"), series
        aer = float(series.mean()) * TRADING_DAYS
        std = float(series.std())
        ir = aer / (std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else float("nan")
        return aer, ir, series
