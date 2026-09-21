"""Multi-dimensional alpha evaluation (paper Section 3 + Appendix D).

Each candidate alpha is backtested on the training period and scored on five dimensions:

- Effectiveness: RankIC
- Stability:     RankIR
- Turnover:      average daily portfolio turnover (lower is better)
- Diversity:     maximum correlation with the alpha zoo (lower is better)
- Overfitting risk: qualitative LLM judgment (0-10, scaled to [0, 1])

The first four are scored by relative percentile rank against the zoo (Eq. 6-7); the overall
score S(f) is the mean of all dimension scores (Eq. 8) and serves as the MCTS reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd

from ..expression.formula import AlphaFormula, FormulaError
from .longshort import (
    long_short_sharpe,
    monotonic_fraction,
    monotonicity_score,
    quantile_mean_returns,
)
from .metrics import AlphaMetrics, compute_alpha_metrics, rank_ic_series, mean_and_ir
from .robustness import (
    asset_mismatch_ic,
    block_positive_fraction,
    newey_west_mean_test,
    yearly_positive_fraction,
)

if TYPE_CHECKING:
    from ..data.base import MarketPanel
    from ..zoo import AlphaZoo

DIMENSIONS = ("effectiveness", "stability", "turnover", "diversity", "overfitting", "monotonicity")

DIMENSION_DESCRIPTIONS = {
    "effectiveness": "the alpha's core predictive power, measured by RankIC against future returns",
    "stability": "the consistency of predictive performance over time, measured by RankIR",
    "turnover": "the trading cost implied by the alpha; keep average daily portfolio turnover low",
    "diversity": "the novelty relative to already-discovered alphas; reduce correlation with the repository",
    "overfitting": "the risk that the formula is overly tailored to the training data; simplify and justify",
    "monotonicity": (
        "the cross-sectional shape of returns: when stocks are bucketed into quintiles by the "
        "signal, bucket returns should increase monotonically from Q1 to Q5, the Q5-Q1 spread "
        "must be positive, and the daily-rebalanced Q5-minus-Q1 long-short portfolio should "
        "earn a high Sharpe ratio; both legs (not just the short side) should contribute"
    ),
}

# overfit_scorer(formula_expression, refinement_history_text) -> (score_0_to_10, reason)
OverfitScorer = Callable[[str, str], tuple[float, str]]


@dataclass
class EvaluationResult:
    formula: AlphaFormula
    chosen_args: dict[str, float]
    expression: str
    metrics: AlphaMetrics
    scores: dict[str, float]
    overall: float
    overfit_reason: str
    values: pd.DataFrame  # training-period factor values (for correlation / exemplars)


class MultiDimEvaluator:
    def __init__(
        self,
        panel: "MarketPanel",
        horizon: int,
        zoo: "AlphaZoo",
        train_ratio: float = 0.7,
        top_frac: float = 0.1,
        min_coverage: float = 0.05,
        min_valid_days: int = 30,
        train_range: tuple[str, str] | None = None,
        valid_range: tuple[str, str] | None = None,
    ):
        self.panel = panel
        self.horizon = horizon
        self.zoo = zoo
        self.top_frac = top_frac
        self.min_coverage = min_coverage
        self.min_valid_days = min_valid_days
        if train_range is not None:
            self.train_dates = panel.dates_between(train_range[0], train_range[1])
        else:
            mask = panel.train_mask(train_ratio)
            self.train_dates = mask[mask].index
        self.fwd_train = panel.apply_universe(panel.forward_returns(horizon)).loc[self.train_dates]
        # next-day raw returns aligned to the signal date, for the tradable long-short Sharpe
        close = panel.fields["close"]
        self.ret1_train = panel.apply_universe(close.pct_change().shift(-1)).loc[self.train_dates]
        # hold-out validation period for the generalization check (sign consistency gate
        # and the empirical decay penalty on the overfitting dimension)
        self.valid_dates: pd.Index | None = None
        self.fwd_valid: pd.DataFrame | None = None
        if valid_range is not None:
            dates = panel.dates_between(valid_range[0], valid_range[1])
            if len(dates) > 0:
                self.valid_dates = dates
                self.fwd_valid = panel.apply_universe(panel.forward_returns(horizon)).loc[dates]

    # ------------------------------------------------------------------

    def compute_values(self, formula: AlphaFormula, args: dict[str, float]) -> pd.DataFrame:
        values = self.panel.maybe_neutralize_factor(formula.evaluate(self.panel.fields, args))
        return self.panel.apply_universe(values).loc[self.train_dates]

    def _check_degenerate(self, values: pd.DataFrame, formula: AlphaFormula) -> None:
        coverage = float(values.notna().mean().mean())
        valid_days = int((values.notna().sum(axis=1) >= 3).sum())
        if coverage < self.min_coverage or valid_days < self.min_valid_days:
            raise FormulaError(
                f"The formula '{formula.to_string()}' produces almost no valid values "
                f"(coverage={coverage:.1%}, valid days={valid_days}). Check window sizes and "
                "avoid operations that yield constant or undefined series."
            )
        stds = values.std(axis=1)
        if float(stds.fillna(0.0).max()) < 1e-12:
            raise FormulaError(
                f"The formula '{formula.to_string()}' is cross-sectionally constant and carries "
                "no ranking information. Introduce stock-specific variation."
            )

    def select_best_args(self, formula: AlphaFormula) -> tuple[dict[str, float], pd.DataFrame, float]:
        """Backtest every candidate argument set; keep the best-RankIC configuration."""
        best: tuple[dict[str, float], pd.DataFrame, float] | None = None
        last_error: FormulaError | None = None
        for args in formula.arguments:
            try:
                values = self.compute_values(formula, args)
                self._check_degenerate(values, formula)
            except FormulaError as exc:
                last_error = exc
                continue
            rank_ic, _ = mean_and_ir(rank_ic_series(values, self.fwd_train))
            if rank_ic != rank_ic:
                continue
            if best is None or rank_ic > best[2]:
                best = (args, values, rank_ic)
        if best is None:
            raise last_error or FormulaError(
                f"No argument set of '{formula.to_string()}' yields a computable RankIC."
            )
        return best

    def attach_robustness(self, metrics: AlphaMetrics, values: pd.DataFrame) -> None:
        """Fill the anti-overfitting statistics (training period only)."""
        ic = rank_ic_series(values, self.fwd_train)
        metrics.hac_t, metrics.hac_p = newey_west_mean_test(ic, lag=self.horizon)
        metrics.block_pos_frac = block_positive_fraction(ic)
        metrics.year_pos_frac = yearly_positive_fraction(ic)
        metrics.control_ic = asset_mismatch_ic(values, self.fwd_train)
        # long-short quantile decomposition
        qmeans = quantile_mean_returns(values, self.fwd_train)
        metrics.q_spread = float(qmeans.iloc[-1] - qmeans.iloc[0])
        metrics.q_monotonic = monotonic_fraction(qmeans)
        metrics.ls_sharpe = long_short_sharpe(values, self.ret1_train)

    def attach_validation(self, metrics: AlphaMetrics, formula: AlphaFormula, args: dict[str, float]) -> None:
        """Fill the hold-out validation RankIC/RankIR (only when valid_range is set)."""
        if self.valid_dates is None or self.fwd_valid is None:
            return
        values = self.panel.maybe_neutralize_factor(formula.evaluate(self.panel.fields, args))
        values = self.panel.apply_universe(values).loc[self.valid_dates]
        metrics.rank_ic_valid, metrics.rank_ir_valid = mean_and_ir(
            rank_ic_series(values, self.fwd_valid)
        )

    @staticmethod
    def _generalization_multiplier(metrics: AlphaMetrics) -> tuple[float, str]:
        """Coarse empirical penalty on the overfitting score from validation IC decay.

        Deliberately bucketed (4 levels) rather than continuous, so each candidate
        leaks only ~2 bits of validation information into the search loop.
        """
        ric_t, ric_v = metrics.rank_ic, metrics.rank_ic_valid
        if ric_v != ric_v or ric_t != ric_t or abs(ric_t) < 1e-12:
            return 1.0, ""
        ratio = ric_v / ric_t
        if ratio <= 0.0:
            return 0.2, (
                f"validation RankIC flips sign ({ric_v:+.4f} vs train {ric_t:+.4f}); "
                "the signal does not generalize -- simplify or make it regime-robust"
            )
        if ratio < 0.3:
            return 0.6, f"validation RankIC decays to {ratio:.0%} of the training value"
        if ratio < 0.7:
            return 0.85, f"validation RankIC decays to {ratio:.0%} of the training value"
        return 1.0, ""

    # ------------------------------------------------------------------

    def evaluate(
        self,
        formula: AlphaFormula,
        overfit_scorer: OverfitScorer,
        refinement_history: str = "",
    ) -> EvaluationResult:
        args, values, _ = self.select_best_args(formula)
        max_corr = self.zoo.max_corr_with(values)
        metrics = compute_alpha_metrics(values, self.fwd_train, self.top_frac, max_corr=max_corr)
        self.attach_robustness(metrics, values)
        self.attach_validation(metrics, formula, args)
        expression = formula.to_string(args)

        scores = {
            "effectiveness": 1.0 - self.zoo.relative_rank("rank_ic", metrics.rank_ic, higher_better=True),
            "stability": 1.0 - self.zoo.relative_rank("rank_ir", metrics.rank_ir, higher_better=True),
            "turnover": 1.0 - self.zoo.relative_rank("turnover", metrics.turnover, higher_better=False),
            "diversity": 1.0 - self.zoo.relative_rank("max_corr", metrics.max_corr, higher_better=False),
            # absolute composite in [0,1], like the overfitting dimension
            "monotonicity": monotonicity_score(
                metrics.q_spread, metrics.q_monotonic, metrics.ls_sharpe
            ),
        }
        overfit_score, overfit_reason = overfit_scorer(expression, refinement_history)
        overfit_score = min(max(float(overfit_score), 0.0), 10.0)
        # empirical generalization evidence: validation IC decay scales the structural score
        gen_mult, gen_note = self._generalization_multiplier(metrics)
        if gen_note:
            overfit_reason = f"{overfit_reason}; {gen_note}"
        scores["overfitting"] = overfit_score / 10.0 * gen_mult

        overall = sum(scores.values()) / len(scores)
        return EvaluationResult(
            formula=formula,
            chosen_args=args,
            expression=expression,
            metrics=metrics,
            scores=scores,
            overall=overall,
            overfit_reason=overfit_reason,
            values=values,
        )
