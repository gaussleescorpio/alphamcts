"""Effective alpha repository (the "Alpha Zoo", F_zoo in the paper).

Holds alphas that passed the effectiveness check, provides:

- admission criteria (paper Appendix G "Effective Alpha Check");
- relative-rank statistics used by the multi-dimensional evaluator (Eq. 6-7);
- few-shot exemplar sampling for refinement suggestions (Appendix D);
- FSA forbidden-structure mining;
- persistence to JSON (factor values are recomputable from the formula).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation.metrics import AlphaMetrics, factor_correlation
from .evaluation.robustness import reconstruction_r2
from .expression.formula import AlphaFormula
from .expression.subtree import mine_frequent_subtrees


@dataclass
class AlphaRecord:
    name: str
    description: str
    expression: str                      # rendered with concrete parameters
    formula: AlphaFormula
    args: dict[str, float]               # the chosen (best) argument set
    metrics: AlphaMetrics
    scores: dict[str, float] = field(default_factory=dict)
    overall: float = float("nan")
    overfit_reason: str = ""
    values: pd.DataFrame | None = None   # factor values over the training period (not persisted)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "expression": self.expression,
            "formula": self.formula.to_json(),
            "args": dict(self.args),
            "metrics": self.metrics.to_dict(),
            "scores": dict(self.scores),
            "overall": self.overall,
            "overfit_reason": self.overfit_reason,
        }

    @classmethod
    def from_json(
        cls, obj: dict[str, Any], fields: tuple[str, ...], window_range: tuple[int, int]
    ) -> "AlphaRecord":
        formula = AlphaFormula.from_json(obj["formula"], fields, window_range)
        return cls(
            name=obj["name"],
            description=obj.get("description", ""),
            expression=obj["expression"],
            formula=formula,
            args=dict(obj["args"]),
            metrics=AlphaMetrics.from_dict(obj["metrics"]),
            scores=dict(obj.get("scores", {})),
            overall=float(obj.get("overall", float("nan"))),
            overfit_reason=obj.get("overfit_reason", ""),
        )


class AlphaZoo:
    def __init__(self, criteria: dict[str, float] | None = None):
        criteria = criteria or {}
        self.min_rank_ic = float(criteria.get("min_rank_ic", 0.015))
        self.min_rank_ir = float(criteria.get("min_rank_ir", 0.3))
        self.max_turnover = float(criteria.get("max_turnover", 1.6))
        self.max_correlation = float(criteria.get("max_correlation", 0.8))
        self.max_rel_rank = float(criteria.get("max_rel_rank", 0.95))
        # anti-overfitting gates (NaN statistics are skipped, keeping old zoos loadable)
        self.max_hac_pvalue = float(criteria.get("max_hac_pvalue", 0.05))
        self.min_block_frac = float(criteria.get("min_block_frac", 0.6))
        self.min_year_frac = float(criteria.get("min_year_frac", 0.6))
        self.max_control_ic = float(criteria.get("max_control_ic", 0.02))
        self.max_recon_r2 = float(criteria.get("max_recon_r2", 0.8))
        # long-short decomposition gates (NaN statistics are skipped)
        self.min_q_spread = float(criteria.get("min_q_spread", 0.0))
        self.min_ls_sharpe = float(criteria.get("min_ls_sharpe", 0.0))
        self.min_q_monotonic = float(criteria.get("min_q_monotonic", 0.0))
        # train/valid IC sign consistency gate (skipped when rank_ic_valid is NaN,
        # e.g. no valid_range configured or old zoos)
        self.require_valid_sign = bool(criteria.get("require_valid_sign", False))
        self.records: list[AlphaRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    # ------------------------------------------------------------------
    # Relative ranking (Eq. 6)
    # ------------------------------------------------------------------

    def relative_rank(self, metric: str, value: float, higher_better: bool = True) -> float:
        """R(f, m, F_zoo): fraction of zoo alphas strictly better than `value`."""
        if not self.records or value != value:  # empty zoo or NaN value
            return 0.0
        vals = [getattr(r.metrics, metric) for r in self.records]
        vals = [v for v in vals if v == v]
        if not vals:
            return 0.0
        if higher_better:
            better = sum(1 for v in vals if v > value)
        else:
            better = sum(1 for v in vals if v < value)
        return better / len(self.records)

    def max_corr_with(self, values: pd.DataFrame) -> float:
        """Maximum |correlation| between candidate factor values and any zoo alpha."""
        best = 0.0
        for record in self.records:
            if record.values is None:
                continue
            c = factor_correlation(values, record.values)
            if c == c:
                best = max(best, abs(c))
        return best

    def reconstruction_r2_with(self, values: pd.DataFrame) -> float:
        """Out-of-time R^2 of reconstructing the candidate from current zoo members."""
        pool = [r.values for r in self.records if r.values is not None]
        return reconstruction_r2(values, pool)

    # ------------------------------------------------------------------
    # Admission (Appendix G "Effective Alpha Check")
    # ------------------------------------------------------------------

    def check_effective(self, metrics: AlphaMetrics) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not metrics.rank_ic == metrics.rank_ic or metrics.rank_ic < self.min_rank_ic:
            reasons.append(f"RankIC {metrics.rank_ic:.4f} < {self.min_rank_ic}")
        if not metrics.rank_ir == metrics.rank_ir or metrics.rank_ir < self.min_rank_ir:
            reasons.append(f"RankIR {metrics.rank_ir:.4f} < {self.min_rank_ir}")
        if metrics.turnover == metrics.turnover and metrics.turnover > self.max_turnover:
            reasons.append(f"turnover {metrics.turnover:.3f} > {self.max_turnover}")
        if metrics.max_corr == metrics.max_corr and metrics.max_corr >= self.max_correlation:
            reasons.append(f"max corr with zoo {metrics.max_corr:.3f} >= {self.max_correlation}")
        if self.relative_rank("rank_ic", metrics.rank_ic) > self.max_rel_rank:
            reasons.append(f"RankIC relative rank > {self.max_rel_rank}")
        if self.relative_rank("rank_ir", metrics.rank_ir) > self.max_rel_rank:
            reasons.append(f"RankIR relative rank > {self.max_rel_rank}")
        # anti-overfitting gates (skip when the statistic is unavailable/NaN)
        if metrics.hac_p == metrics.hac_p and metrics.hac_p > self.max_hac_pvalue:
            reasons.append(
                f"HAC p-value {metrics.hac_p:.4f} > {self.max_hac_pvalue} "
                "(mean RankIC not statistically significant)"
            )
        if metrics.block_pos_frac == metrics.block_pos_frac and metrics.block_pos_frac < self.min_block_frac:
            reasons.append(
                f"positive time-block fraction {metrics.block_pos_frac:.2f} < {self.min_block_frac} "
                "(IC sign unstable across sub-periods)"
            )
        if metrics.year_pos_frac == metrics.year_pos_frac and metrics.year_pos_frac < self.min_year_frac:
            reasons.append(
                f"positive-year fraction {metrics.year_pos_frac:.2f} < {self.min_year_frac} "
                "(IC sign inconsistent across years)"
            )
        if metrics.control_ic == metrics.control_ic and metrics.control_ic > self.max_control_ic:
            reasons.append(
                f"negative-control |RankIC| {metrics.control_ic:.4f} > {self.max_control_ic} "
                "(signal survives asset mismatch; likely an artifact)"
            )
        if metrics.recon_r2 == metrics.recon_r2 and metrics.recon_r2 > self.max_recon_r2:
            reasons.append(
                f"zoo reconstruction R^2 {metrics.recon_r2:.3f} > {self.max_recon_r2} "
                "(linearly redundant with existing alphas)"
            )
        # long-short decomposition gates (skip when the statistic is unavailable/NaN)
        if metrics.q_spread == metrics.q_spread and metrics.q_spread <= self.min_q_spread:
            reasons.append(
                f"quantile spread Q5-Q1 {metrics.q_spread:.5f} <= {self.min_q_spread} "
                "(top bucket does not outperform bottom bucket)"
            )
        if metrics.ls_sharpe == metrics.ls_sharpe and metrics.ls_sharpe < self.min_ls_sharpe:
            reasons.append(
                f"long-short Sharpe {metrics.ls_sharpe:.2f} < {self.min_ls_sharpe} "
                "(Q5-Q1 portfolio not consistently profitable)"
            )
        if metrics.q_monotonic == metrics.q_monotonic and metrics.q_monotonic < self.min_q_monotonic:
            reasons.append(
                f"quantile monotonicity {metrics.q_monotonic:.2f} < {self.min_q_monotonic} "
                "(bucket returns not ordered by signal)"
            )
        # train/valid sign consistency (skip when the validation statistic is unavailable)
        if (
            self.require_valid_sign
            and metrics.rank_ic_valid == metrics.rank_ic_valid
            and metrics.rank_ic == metrics.rank_ic
            and metrics.rank_ic * metrics.rank_ic_valid <= 0.0
        ):
            reasons.append(
                f"validation RankIC {metrics.rank_ic_valid:+.4f} flips sign vs train "
                f"{metrics.rank_ic:+.4f} (signal does not generalize to the hold-out period)"
            )
        return not reasons, reasons

    def add(self, record: AlphaRecord) -> tuple[bool, list[str]]:
        if any(r.expression == record.expression for r in self.records):
            return False, ["duplicate expression"]
        # The diversity criteria are checked against the repository *at admission time*
        # (it may have grown since the alpha was evaluated inside its search tree).
        if record.values is not None:
            record.metrics.max_corr = self.max_corr_with(record.values)
            record.metrics.recon_r2 = self.reconstruction_r2_with(record.values)
        ok, reasons = self.check_effective(record.metrics)
        if ok:
            self.records.append(record)
        return ok, reasons

    # ------------------------------------------------------------------
    # Exemplar sampling for refinement suggestions (Appendix D)
    # ------------------------------------------------------------------

    def exemplars(
        self,
        dimension: str,
        candidate_values: pd.DataFrame | None,
        k: int = 1,
        eta: float = 0.5,
    ) -> list[AlphaRecord]:
        """Few-shot exemplars for a refinement dimension.

        - effectiveness / stability / monotonicity: drop the top-eta fraction most
          correlated with the candidate, then take the top-k by RankIC / RankIR / LS Sharpe;
        - diversity: take the k least-correlated alphas;
        - turnover / overfitting: zero-shot (empty list).
        """
        if dimension in ("turnover", "overfitting") or not self.records:
            return []

        scored: list[tuple[float, AlphaRecord]] = []
        for record in self.records:
            corr = 0.0
            if candidate_values is not None and record.values is not None:
                c = factor_correlation(candidate_values, record.values)
                corr = abs(c) if c == c else 0.0
            scored.append((corr, record))

        if dimension == "diversity":
            scored.sort(key=lambda t: t[0])
            return [r for _, r in scored[:k]]

        # effectiveness / stability / monotonicity
        scored.sort(key=lambda t: -t[0])
        n_drop = int(len(scored) * eta)
        pool = [r for _, r in scored[n_drop:]] or [r for _, r in scored]
        metric = {"effectiveness": "rank_ic", "monotonicity": "ls_sharpe"}.get(dimension, "rank_ir")
        pool.sort(key=lambda r: -(getattr(r.metrics, metric) or float("-inf")))
        return pool[:k]

    # ------------------------------------------------------------------
    # Selection / FSA / persistence
    # ------------------------------------------------------------------

    def top_by_rank_ir(self, k: int) -> list[AlphaRecord]:
        ranked = sorted(self.records, key=lambda r: -(r.metrics.rank_ir if r.metrics.rank_ir == r.metrics.rank_ir else float("-inf")))
        return ranked[:k]

    def forbidden_subtrees(self, top_k: int = 3, min_support: float = 0.3, min_alphas: int = 5) -> list[str]:
        if len(self.records) < min_alphas:
            return []
        trees = [r.formula.tree() for r in self.records]
        return mine_frequent_subtrees(trees, top_k=top_k, min_support=min_support)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"criteria": {
            "min_rank_ic": self.min_rank_ic,
            "min_rank_ir": self.min_rank_ir,
            "max_turnover": self.max_turnover,
            "max_correlation": self.max_correlation,
            "max_rel_rank": self.max_rel_rank,
            "max_hac_pvalue": self.max_hac_pvalue,
            "min_block_frac": self.min_block_frac,
            "min_year_frac": self.min_year_frac,
            "max_control_ic": self.max_control_ic,
            "max_recon_r2": self.max_recon_r2,
            "min_q_spread": self.min_q_spread,
            "min_ls_sharpe": self.min_ls_sharpe,
            "min_q_monotonic": self.min_q_monotonic,
            "require_valid_sign": self.require_valid_sign,
        }, "alphas": [r.to_json() for r in self.records]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path, fields: tuple[str, ...], window_range: tuple[int, int]
    ) -> "AlphaZoo":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        zoo = cls(payload.get("criteria", {}))
        for obj in payload.get("alphas", []):
            zoo.records.append(AlphaRecord.from_json(obj, fields, window_range))
        return zoo
