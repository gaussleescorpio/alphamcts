"""LLM client interface and the high-level alpha agent.

`LLMClient` is a minimal text-completion interface; `AlphaAgent` layers the paper's prompt
workflows on top of it:

- two-step generation: alpha portrait -> concrete formula (Appendix D "Alpha Formula
  Generation");
- dimension-targeted refinement suggestions (Appendix D);
- refinement: suggestion -> revised portrait -> formula;
- overfitting risk assessment (Appendix K, Fig. 17);
- iterative correction of invalid outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..expression.formula import AlphaFormula, FormulaError
from ..expression.operators import operators_prompt_block
from . import prompts


class LLMClient(Protocol):
    def complete(self, prompt: str, temperature: float = 1.0) -> str:  # pragma: no cover
        ...


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first balanced top-level JSON object from LLM output."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise FormulaError("The output does not contain a valid JSON object.")


@dataclass
class Temperatures:
    generate: float = 1.0
    correct: float = 0.8
    overfit: float = 0.1


class AlphaAgent:
    """High-level LLM workflows for alpha mining."""

    def __init__(
        self,
        client: LLMClient,
        fields: tuple[str, ...],
        window_range: tuple[int, int],
        temperatures: Temperatures | None = None,
        max_correction_attempts: int = 3,
        overfit_scorer: str = "llm",
    ):
        if overfit_scorer not in ("llm", "heuristic"):
            raise ValueError(f"overfit_scorer must be 'llm' or 'heuristic', got {overfit_scorer!r}")
        self.client = client
        self.fields = fields
        self.window_range = window_range
        self.temps = temperatures or Temperatures()
        self.max_correction_attempts = max_correction_attempts
        self.overfit_scorer = overfit_scorer

    # ------------------------------------------------------------------
    # Prompt building blocks
    # ------------------------------------------------------------------

    def _fields_block(self) -> str:
        return ", ".join(self.fields)

    @staticmethod
    def _freq_block(forbidden_subtrees: list[str]) -> str:
        return "\n".join(forbidden_subtrees) if forbidden_subtrees else "(none)"

    # ------------------------------------------------------------------
    # Portrait generation / refinement (Figures 15 and 18)
    # ------------------------------------------------------------------

    def generate_portrait(self, forbidden_subtrees: list[str]) -> dict[str, Any]:
        prompt = prompts.ALPHA_PORTRAIT_PROMPT.substitute(
            available_fields=self._fields_block(),
            available_operators=operators_prompt_block(),
            freq_subtrees=self._freq_block(forbidden_subtrees),
        )
        return self._complete_json(prompt, self.temps.generate)

    def refine_portrait(
        self,
        origin_expression: str,
        suggestions: list[str],
        forbidden_subtrees: list[str],
    ) -> dict[str, Any]:
        prompt = prompts.ALPHA_REFINEMENT_PROMPT.substitute(
            available_fields=self._fields_block(),
            available_operators=operators_prompt_block(),
            freq_subtrees=self._freq_block(forbidden_subtrees),
            origin_alpha_formula=origin_expression,
            refinement_suggestions="\n".join(f"- {s}" for s in suggestions),
        )
        return self._complete_json(prompt, self.temps.generate)

    # ------------------------------------------------------------------
    # Formula generation with iterative correction (Figure 16 + IsValid loop)
    # ------------------------------------------------------------------

    def generate_formula(self, portrait: dict[str, Any]) -> AlphaFormula:
        portrait_block = self._portrait_requirements(portrait)
        prompt = prompts.ALPHA_FORMULA_PROMPT.substitute(
            available_fields=self._fields_block(),
            available_operators=operators_prompt_block(),
            alpha_portrait_prompt=portrait_block,
            window_range=f"[{self.window_range[0]}, {self.window_range[1]}]",
        )
        text = self.client.complete(prompt, self.temps.generate)
        return self._validate_with_correction(prompt, text, portrait)

    def _validate_with_correction(
        self, base_prompt: str, text: str, portrait: dict[str, Any]
    ) -> AlphaFormula:
        last_error: FormulaError | None = None
        for _attempt in range(self.max_correction_attempts + 1):
            try:
                obj = extract_json(text)
                return AlphaFormula.from_json(
                    obj,
                    self.fields,
                    self.window_range,
                    name=str(portrait.get("name", "alpha")),
                    description=str(portrait.get("description", "")),
                )
            except FormulaError as exc:
                last_error = exc
                correction = prompts.CORRECTION_SUFFIX.substitute(
                    previous_output=text[-4000:], error=str(exc)
                )
                text = self.client.complete(base_prompt + correction, self.temps.correct)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _portrait_requirements(portrait: dict[str, Any]) -> str:
        lines = [
            f"- Alpha name: {portrait.get('name', 'alpha')}",
            f"- Investment logic: {portrait.get('description', '')}",
            "- Implement exactly this calculation (pseudo-code):",
        ]
        for line in portrait.get("pseudo_code", []) or []:
            lines.append(f"    {line}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Refinement suggestions (Appendix D)
    # ------------------------------------------------------------------

    def suggest_refinements(
        self,
        dimension: str,
        dimension_description: str,
        alpha_expression: str,
        alpha_description: str,
        evaluation_scores: dict[str, float],
        backtest_metrics: dict[str, float],
        refinement_context: str,
        examples: list[str],
        forbidden_subtrees: list[str],
    ) -> list[str]:
        prompt = prompts.SUGGESTION_PROMPT.substitute(
            dimension=dimension,
            dimension_description=dimension_description,
            alpha_expression=alpha_expression,
            alpha_description=alpha_description,
            evaluation_scores=json.dumps({k: round(v, 4) for k, v in evaluation_scores.items()}),
            backtest_metrics=json.dumps({k: round(v, 4) for k, v in backtest_metrics.items() if v == v}),
            refinement_context=refinement_context or "(root alpha, no history)",
            examples="\n".join(examples) if examples else "(repository is empty)",
            freq_subtrees=self._freq_block(forbidden_subtrees),
        )
        try:
            obj = self._complete_json(prompt, self.temps.generate)
            suggestions = obj.get("suggestions", [])
            if isinstance(suggestions, list) and suggestions:
                return [str(s) for s in suggestions][:3]
        except FormulaError:
            pass
        return [f"Improve the alpha's {dimension}."]

    # ------------------------------------------------------------------
    # Overfitting risk assessment (Figure 17)
    # ------------------------------------------------------------------

    def assess_overfitting(self, expression: str, refinement_history: str) -> tuple[float, str]:
        if self.overfit_scorer == "heuristic":
            from .heuristic import heuristic_overfit_score

            return heuristic_overfit_score(expression, refinement_history)
        prompt = prompts.OVERFITTING_PROMPT.substitute(
            alpha_formula=expression,
            refinement_history=refinement_history or "(no refinement history; initial alpha)",
        )
        try:
            obj = self._complete_json(prompt, self.temps.overfit)
            score = float(obj.get("score", 5.0))
            reason = str(obj.get("reason", ""))
            return score, reason
        except (FormulaError, TypeError, ValueError):
            return 5.0, "assessment unavailable; defaulting to medium risk"

    # ------------------------------------------------------------------

    def _complete_json(self, prompt: str, temperature: float) -> dict[str, Any]:
        text = self.client.complete(prompt, temperature)
        return extract_json(text)
