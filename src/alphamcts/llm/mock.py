"""Deterministic offline LLM used for tests and end-to-end dry runs.

Routes on distinctive markers in each prompt template and answers with syntactically valid
JSON. Formula generation samples from a library of economically-motivated templates
(reversal, price-volume, volatility, intraday pressure, ...) with randomized windows, and
makes a best-effort attempt to honor the forbidden-subtree (FSA) constraint embedded in the
prompt.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable

import numpy as np

from ..data.base import FIELDS
from ..expression.formula import AlphaFormula, FormulaError
from ..expression.subtree import abstract_root_genes

WINDOW_RANGE = (2, 60)

TemplateFn = Callable[[np.random.Generator], dict[str, Any]]


def _w(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _steps(*ops: tuple[str, list[str], list[str], str]) -> list[dict[str, Any]]:
    return [{"name": n, "param": p, "input": i, "output": o} for n, p, i, o in ops]


def _t_reversal(rng) -> dict[str, Any]:
    return {
        "name": "short_term_reversal",
        "description": "Recent losers outperform recent winners as temporary mispricing reverts.",
        "formula": _steps(
            ("Pct", ["lookback_window"], ["close"], "recent_return"),
            ("Neg", [], ["recent_return"], "alpha"),
        ),
        "arguments": [{"lookback_window": _w(rng, 3, 15)}, {"lookback_window": _w(rng, 15, 40)}],
    }


def _t_reversal_zscore(rng) -> dict[str, Any]:
    return {
        "name": "normalized_reversal",
        "description": "Z-scored recent return; extreme run-ups tend to revert.",
        "formula": _steps(
            ("Pct", ["return_window"], ["close"], "recent_return"),
            ("Zscore", ["norm_window"], ["recent_return"], "zret"),
            ("Neg", [], ["zret"], "alpha"),
        ),
        "arguments": [
            {"return_window": _w(rng, 3, 10), "norm_window": _w(rng, 15, 40)},
            {"return_window": _w(rng, 5, 20), "norm_window": _w(rng, 20, 60)},
        ],
    }


def _t_pressure(rng) -> dict[str, Any]:
    return {
        "name": "intraday_pressure_zscore",
        "description": "End-of-day buying pressure (close vs vwap), smoothed then normalized.",
        "formula": _steps(
            ("Sub", [], ["close", "vwap"], "pressure"),
            ("Ma", ["smooth_window"], ["pressure"], "trend"),
            ("Zscore", ["norm_window"], ["trend"], "alpha"),
        ),
        "arguments": [
            {"smooth_window": _w(rng, 10, 25), "norm_window": _w(rng, 20, 45)},
            {"smooth_window": _w(rng, 5, 15), "norm_window": _w(rng, 15, 30)},
        ],
    }


def _t_pv_corr(rng) -> dict[str, Any]:
    return {
        "name": "price_volume_divergence",
        "description": "Negative price-volume correlation signals accumulation against the trend.",
        "formula": _steps(
            ("Corr", ["corr_window"], ["close", "volume"], "pv_corr"),
            ("Neg", [], ["pv_corr"], "alpha"),
        ),
        "arguments": [{"corr_window": _w(rng, 10, 30)}, {"corr_window": _w(rng, 30, 60)}],
    }


def _t_volume_burst_reversal(rng) -> dict[str, Any]:
    return {
        "name": "volume_burst_reversal",
        "description": "Price moves accompanied by abnormal volume revert more strongly.",
        "formula": _steps(
            ("Ma", ["volume_window"], ["volume"], "avg_volume"),
            ("Div", [], ["volume", "avg_volume"], "volume_ratio"),
            ("Pct", ["return_window"], ["close"], "recent_return"),
            ("Mul", [], ["recent_return", "volume_ratio"], "signed_burst"),
            ("Neg", [], ["signed_burst"], "alpha"),
        ),
        "arguments": [
            {"volume_window": _w(rng, 15, 40), "return_window": _w(rng, 3, 10)},
            {"volume_window": _w(rng, 20, 60), "return_window": _w(rng, 5, 15)},
        ],
    }


def _t_vol_of_vwap(rng) -> dict[str, Any]:
    return {
        "name": "vwap_volatility",
        "description": "Volatility of vwap changes; volatile names carry a premium/discount.",
        "formula": _steps(
            ("Pct", ["return_window"], ["vwap"], "vwap_ret"),
            ("Std", ["vol_window"], ["vwap_ret"], "vol"),
            ("Neg", [], ["vol"], "alpha"),
        ),
        "arguments": [
            {"return_window": _w(rng, 5, 20), "vol_window": _w(rng, 15, 40)},
        ],
    }


def _t_range_position(rng) -> dict[str, Any]:
    return {
        "name": "range_position",
        "description": "Position of the close within its recent high-low range (contrarian).",
        "formula": _steps(
            ("Min", ["range_window"], ["low"], "range_low"),
            ("Max", ["range_window"], ["high"], "range_high"),
            ("Sub", [], ["close", "range_low"], "above_low"),
            ("Sub", [], ["range_high", "range_low"], "range_span"),
            ("Div", [], ["above_low", "range_span"], "position"),
            ("Neg", [], ["position"], "alpha"),
        ),
        "arguments": [{"range_window": _w(rng, 10, 30)}, {"range_window": _w(rng, 30, 60)}],
    }


def _t_volume_accel(rng) -> dict[str, Any]:
    return {
        "name": "volume_acceleration",
        "description": "Change in short-run average volume relative to the long-run level.",
        "formula": _steps(
            ("Ma", ["short_window"], ["volume"], "short_vol"),
            ("Diff", ["diff_lag"], ["short_vol"], "vol_change"),
            ("Ma", ["long_window"], ["volume"], "long_vol"),
            ("Div", [], ["vol_change", "long_vol"], "alpha"),
        ),
        "arguments": [
            {"short_window": _w(rng, 10, 25), "diff_lag": _w(rng, 2, 5), "long_window": _w(rng, 40, 60)},
        ],
    }


def _t_skew_reversal(rng) -> dict[str, Any]:
    return {
        "name": "return_skewness",
        "description": "Stocks with positively skewed recent returns underperform (lottery effect).",
        "formula": _steps(
            ("Pct", ["return_window"], ["close"], "ret"),
            ("Skew", ["skew_window"], ["ret"], "ret_skew"),
            ("Neg", [], ["ret_skew"], "alpha"),
        ),
        "arguments": [
            {"return_window": _w(rng, 2, 5), "skew_window": _w(rng, 15, 40)},
        ],
    }


def _t_autocorr(rng) -> dict[str, Any]:
    return {
        "name": "return_autocorrelation",
        "description": "Serial correlation of returns distinguishes trending from choppy names.",
        "formula": _steps(
            ("Pct", ["return_window"], ["close"], "ret"),
            ("Autocorr", ["corr_window", "lag"], ["ret"], "alpha"),
        ),
        "arguments": [
            {"return_window": _w(rng, 2, 5), "corr_window": _w(rng, 20, 50), "lag": _w(rng, 1, 5)},
        ],
    }


def _t_rank_reversal(rng) -> dict[str, Any]:
    return {
        "name": "price_rank_reversal",
        "description": "Close near its recent maximum tends to revert.",
        "formula": _steps(
            ("Rank", ["rank_window"], ["close"], "price_rank"),
            ("Neg", [], ["price_rank"], "alpha"),
        ),
        "arguments": [{"rank_window": _w(rng, 10, 30)}, {"rank_window": _w(rng, 30, 60)}],
    }


def _t_vari_volume(rng) -> dict[str, Any]:
    return {
        "name": "volume_variation",
        "description": "Coefficient of variation of volume; unstable trading interest reverts prices.",
        "formula": _steps(
            ("Vari", ["vari_window"], ["volume"], "volume_cv"),
            ("Zscore", ["norm_window"], ["volume_cv"], "alpha"),
        ),
        "arguments": [
            {"vari_window": _w(rng, 10, 30), "norm_window": _w(rng, 20, 50)},
        ],
    }


TEMPLATES: list[TemplateFn] = [
    _t_reversal,
    _t_reversal_zscore,
    _t_pressure,
    _t_pv_corr,
    _t_volume_burst_reversal,
    _t_vol_of_vwap,
    _t_range_position,
    _t_volume_accel,
    _t_skew_reversal,
    _t_autocorr,
    _t_rank_reversal,
    _t_vari_volume,
]

_SUGGESTION_BANK = {
    "effectiveness": [
        "Condition the reversal signal on abnormal volume so it fires only when mispricing is likely.",
        "Replace the raw return with a z-scored return to sharpen the cross-sectional ordering.",
    ],
    "stability": [
        "Smooth the signal with a moving average to reduce sensitivity to single-day noise.",
        "Normalize by a longer-window z-score so the signal is comparable across regimes.",
    ],
    "turnover": [
        "Lengthen the look-back windows to slow the signal and reduce daily portfolio churn.",
        "Apply a moving average to the final signal to damp day-to-day rank changes.",
    ],
    "diversity": [
        "Use a different data field (e.g. vwap or the high-low range) as the core driver.",
        "Combine the price signal with a volume-based conditioner to decorrelate from existing alphas.",
    ],
    "overfitting": [
        "Simplify the expression by removing a nested transformation with unclear rationale.",
        "Reduce the number of tuned parameters and use conventional window lengths.",
    ],
    "monotonicity": [
        "Symmetrize the signal (e.g. z-score instead of a one-sided threshold) so both the long "
        "and the short leg carry information.",
        "Normalize by volatility so extreme buckets are not dominated by a few high-variance names.",
    ],
}


class MockLLM:
    """Offline LLM: prompt-marker routing + template-based formula generation."""

    def __init__(self, seed: int = 0, invalid_rate: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.invalid_rate = invalid_rate
        # portrait -> formula handoff is per-thread so parallel miners don't cross wires
        self._tls = threading.local()
        self._rng_lock = threading.Lock()

    @property
    def _pending_formula(self) -> dict[str, Any] | None:
        return getattr(self._tls, "pending", None)

    @_pending_formula.setter
    def _pending_formula(self, value: dict[str, Any] | None) -> None:
        self._tls.pending = value

    # ------------------------------------------------------------------

    def complete(self, prompt: str, temperature: float = 1.0) -> str:
        with self._rng_lock:  # numpy Generator is not thread-safe
            return self._complete(prompt, temperature)

    def _complete(self, prompt: str, temperature: float = 1.0) -> str:
        if "Critical Alpha Overfitting Risk Assessment" in prompt:
            return self._overfitting_response(prompt)
        if "propose targeted refinement suggestions" in prompt:
            return self._suggestion_response(prompt)
        if "Correction Required" in prompt:
            return self._formula_response(prompt, force_valid=True)
        if '"formula", and "arguments"' in prompt:
            return self._formula_response(prompt)
        # portrait generation (Fig. 15) or refinement (Fig. 18)
        return self._portrait_response(prompt)

    # ------------------------------------------------------------------

    def _parse_forbidden(self, prompt: str) -> list[str]:
        m = re.search(
            r"avoid (?:including|recommending structures containing) (?:the following sub-expressions|these sub-expressions):?\s*```?\s*(.*?)\s*```",
            prompt,
            re.S | re.I,
        )
        if not m:
            m = re.search(r"sub-expressions:\s*(.*?)(?:\n\s*\n|Original alpha)", prompt, re.S)
        if not m:
            return []
        block = m.group(1)
        return [line.strip("-• \t") for line in block.splitlines() if line.strip() and "(none)" not in line]

    def _sample_formula(self, forbidden: list[str]) -> dict[str, Any]:
        order = self.rng.permutation(len(TEMPLATES))
        fallback: dict[str, Any] | None = None
        for idx in order:
            candidate = TEMPLATES[idx](self.rng)
            fallback = fallback or candidate
            if not forbidden:
                return candidate
            try:
                f = AlphaFormula.from_json(dict(candidate), FIELDS, WINDOW_RANGE)
                genes = abstract_root_genes(f.tree())
                if not genes.intersection(forbidden):
                    return candidate
            except FormulaError:
                continue
        assert fallback is not None
        return fallback

    def _portrait_response(self, prompt: str) -> str:
        forbidden = self._parse_forbidden(prompt)
        obj = self._sample_formula(forbidden)
        self._pending_formula = obj
        pseudo = [
            f"{step['output']} = {step['name']}(input={step['input']}, param={step['param']})"
            for step in obj["formula"]
        ]
        return json.dumps(
            {"name": obj["name"], "description": obj["description"], "pseudo_code": pseudo}
        )

    def _formula_response(self, prompt: str, force_valid: bool = False) -> str:
        obj = self._pending_formula or self._sample_formula(self._parse_forbidden(prompt))
        payload = {
            "name": obj["name"],
            "description": obj["description"],
            "formula": [dict(s) for s in obj["formula"]],
            "arguments": [dict(a) for a in obj["arguments"]],
        }
        if not force_valid and self.rng.random() < self.invalid_rate:
            payload["formula"][0] = dict(payload["formula"][0], name="Wavelet")
        return json.dumps(payload)

    def _suggestion_response(self, prompt: str) -> str:
        m = re.search(r"Target Dimension:\s*(\w+)", prompt)
        dim = (m.group(1).lower() if m else "effectiveness")
        bank = _SUGGESTION_BANK.get(dim, _SUGGESTION_BANK["effectiveness"])
        pick = bank[int(self.rng.integers(0, len(bank)))]
        return json.dumps({"suggestions": [pick]})

    def _overfitting_response(self, prompt: str) -> str:
        m = re.search(r"Alpha Expression:\s*```\s*(.*?)\s*```", prompt, re.S)
        expr = m.group(1) if m else ""
        complexity = expr.count("(")
        score = float(np.clip(9.0 - 0.4 * max(0, complexity - 3) + self.rng.normal(0, 0.8), 1.0, 10.0))
        reason = (
            "Simple, economically motivated structure with few parameters."
            if complexity <= 5
            else "Nested structure raises moderate curve-fitting concerns."
        )
        return json.dumps({"reason": reason, "score": round(score, 1)})
