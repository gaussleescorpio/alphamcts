"""Structured alpha formula representation.

Follows the JSON schema used in the paper's prompts (Appendix K, Fig. 16):

.. code-block:: json

    {
      "name": "intraday_pressure_zscore",
      "description": "...",
      "formula": [
        {"name": "Sub", "param": [], "input": ["close", "vwap"], "output": "pressure"},
        {"name": "Ma", "param": ["smooth_window"], "input": ["pressure"], "output": "trend"},
        {"name": "Zscore", "param": ["norm_window"], "input": ["trend"], "output": "alpha"}
      ],
      "arguments": [{"smooth_window": 20, "norm_window": 30}]
    }

The formula is a sequential single-assignment program; parameters are symbolic and bound by
one of up to three candidate argument sets (each is backtested, the best is kept).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd

from .operators import OperatorSpec, get_operator

MAX_PARAMS = 3
MAX_ARGUMENT_SETS = 3


class FormulaError(ValueError):
    """Raised on invalid formulas; the message is used as LLM correction feedback."""


@dataclass
class FormulaStep:
    op: str
    params: list[str]
    inputs: list[str]
    output: str

    def to_json(self) -> dict[str, Any]:
        return {"name": self.op, "param": list(self.params), "input": list(self.inputs), "output": self.output}


@dataclass
class ExprNode:
    """Expression-tree node; leaves are raw data fields."""

    op: str | None = None
    children: list["ExprNode"] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)
    leaf: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def render(self) -> str:
        if self.is_leaf:
            return str(self.leaf)
        spec = get_operator(self.op)
        if spec is not None and spec.infix and len(self.children) == 2:
            return f"({self.children[0].render()}{spec.infix}{self.children[1].render()})"
        parts = [c.render() for c in self.children] + [str(p) for p in self.params]
        return f"{self.op}({','.join(parts)})"

    def abstracted(self) -> "ExprNode":
        """Abs(.) from the paper: replace concrete parameter values with a placeholder."""
        if self.is_leaf:
            return ExprNode(leaf=self.leaf)
        return ExprNode(
            op=self.op,
            children=[c.abstracted() for c in self.children],
            params=["t"] * len(self.params),
        )

    def iter_subtrees(self) -> Iterator[tuple["ExprNode", "ExprNode | None"]]:
        """Yield (subtree, parent) pairs for every node in the tree."""
        stack: list[tuple[ExprNode, ExprNode | None]] = [(self, None)]
        while stack:
            node, parent = stack.pop()
            yield node, parent
            for child in node.children:
                stack.append((child, node))

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class AlphaFormula:
    name: str
    description: str
    steps: list[FormulaStep]
    arguments: list[dict[str, float]]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        obj: dict[str, Any] | str,
        fields: tuple[str, ...],
        window_range: tuple[int, int],
        name: str = "",
        description: str = "",
    ) -> "AlphaFormula":
        """Parse and validate a formula from the paper's JSON schema."""
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except json.JSONDecodeError as exc:
                raise FormulaError(f"Output is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise FormulaError("Formula JSON must be an object with 'formula' and 'arguments' keys.")

        raw_steps = obj.get("formula")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise FormulaError("'formula' must be a non-empty list of operations.")

        steps: list[FormulaStep] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise FormulaError(f"Operation #{i + 1} must be an object.")
            op_name = raw.get("name") or raw.get("op")
            if not op_name:
                raise FormulaError(f"Operation #{i + 1} is missing the operator 'name'.")
            output = raw.get("output")
            if not output or not isinstance(output, str):
                raise FormulaError(f"Operation #{i + 1} ({op_name}) is missing a string 'output'.")
            params = [str(p) for p in _as_list(raw.get("param", raw.get("params")))]
            inputs = [str(x) for x in _as_list(raw.get("input", raw.get("inputs")))]
            steps.append(FormulaStep(op=str(op_name), params=params, inputs=inputs, output=output))

        raw_args = obj.get("arguments")
        if isinstance(raw_args, dict):
            raw_args = [raw_args]
        if not isinstance(raw_args, list) or not raw_args:
            raise FormulaError("'arguments' must be a non-empty list of parameter-value objects.")
        arguments: list[dict[str, float]] = []
        for i, arg_set in enumerate(raw_args[:MAX_ARGUMENT_SETS]):
            if not isinstance(arg_set, dict):
                raise FormulaError(f"Argument set #{i + 1} must be an object mapping parameter names to values.")
            arguments.append({str(k): v for k, v in arg_set.items()})

        formula = cls(
            name=str(obj.get("name", name) or name or "alpha"),
            description=str(obj.get("description", description) or description),
            steps=steps,
            arguments=arguments,
        )
        formula.validate(fields, window_range)
        return formula

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def param_names(self) -> list[str]:
        names: list[str] = []
        for step in self.steps:
            for p in step.params:
                if p not in names:
                    names.append(p)
        return names

    def validate(self, fields: tuple[str, ...], window_range: tuple[int, int]) -> None:
        defined: set[str] = set()
        field_set = set(fields)
        for i, step in enumerate(self.steps):
            spec = self._resolve_op(step, i)
            if len(step.inputs) != spec.n_inputs:
                raise FormulaError(
                    f"Operation #{i + 1} ({spec.name}) expects {spec.n_inputs} input(s), got "
                    f"{len(step.inputs)}: {step.inputs}."
                )
            if len(step.params) != spec.n_params:
                raise FormulaError(
                    f"Operation #{i + 1} ({spec.name}) expects {spec.n_params} parameter(s), got "
                    f"{len(step.params)}: {step.params}."
                )
            for inp in step.inputs:
                if _is_number(inp):
                    raise FormulaError(
                        f"Operation #{i + 1} ({spec.name}): input '{inp}' is numeric. Inputs must be "
                        "data fields or outputs of previous operations, never numbers."
                    )
                if inp not in field_set and inp not in defined:
                    raise FormulaError(
                        f"Operation #{i + 1} ({spec.name}): input '{inp}' is neither an available data "
                        f"field ({', '.join(fields)}) nor the output of a previous operation."
                    )
            if step.output in field_set:
                raise FormulaError(
                    f"Operation #{i + 1} ({spec.name}): output '{step.output}' shadows a raw data field."
                )
            if step.output in defined:
                raise FormulaError(
                    f"Operation #{i + 1} ({spec.name}): output '{step.output}' is defined more than once."
                )
            defined.add(step.output)

        params = self.param_names()
        if len(params) > MAX_PARAMS:
            raise FormulaError(
                f"The alpha uses {len(params)} parameters ({params}); at most {MAX_PARAMS} are allowed."
            )

        lo, hi = 1, int(window_range[1])
        for i, arg_set in enumerate(self.arguments):
            missing = [p for p in params if p not in arg_set]
            if missing:
                raise FormulaError(f"Argument set #{i + 1} is missing values for parameters: {missing}.")
            extra = [k for k in arg_set if k not in params]
            if extra:
                raise FormulaError(
                    f"Argument set #{i + 1} has values for undefined parameters: {extra}. "
                    f"Defined parameters are: {params}."
                )
            for p, v in arg_set.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise FormulaError(f"Argument set #{i + 1}: parameter '{p}' must be a number, got {v!r}.")
                iv = int(round(float(v)))
                if iv < lo or iv > hi:
                    raise FormulaError(
                        f"Argument set #{i + 1}: parameter '{p}'={v} is outside the allowed range "
                        f"[{lo}, {hi}]."
                    )

    @staticmethod
    def _resolve_op(step: FormulaStep, i: int) -> OperatorSpec:
        spec = get_operator(step.op)
        if spec is None:
            raise FormulaError(
                f"Operation #{i + 1}: unknown operator '{step.op}'. Use only operators from the "
                "'Available Operators' list."
            )
        return spec

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, fields: dict[str, pd.DataFrame], args: dict[str, float]) -> pd.DataFrame:
        env: dict[str, pd.DataFrame] = dict(fields)
        result: pd.DataFrame | None = None
        for i, step in enumerate(self.steps):
            spec = self._resolve_op(step, i)
            inputs = [env[name] for name in step.inputs]
            params = [int(round(float(args[p]))) for p in step.params]
            result = spec(*inputs, *params)
            env[step.output] = result
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Expression tree / rendering
    # ------------------------------------------------------------------

    def tree(self, args: dict[str, float] | None = None) -> ExprNode:
        nodes: dict[str, ExprNode] = {}

        def resolve(name: str) -> ExprNode:
            if name in nodes:
                return nodes[name]
            return ExprNode(leaf=name)

        for step in self.steps:
            spec = get_operator(step.op)
            op_name = spec.name if spec is not None else step.op
            params: list[Any] = []
            for p in step.params:
                if args is not None and p in args:
                    params.append(int(round(float(args[p]))))
                else:
                    params.append(p)
            nodes[step.output] = ExprNode(
                op=op_name,
                children=[resolve(x) for x in step.inputs],
                params=params,
            )
        return nodes[self.steps[-1].output]

    def to_string(self, args: dict[str, float] | None = None) -> str:
        return self.tree(args).render()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formula": [s.to_json() for s in self.steps],
            "arguments": [dict(a) for a in self.arguments],
        }

    def n_operators(self) -> int:
        return len(self.steps)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
