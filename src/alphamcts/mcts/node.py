"""MCTS search-tree node.

Each node represents a candidate alpha (formula + multi-dimensional evaluation) plus the
refinement action that produced it. Nodes carry visit counts N and the edge value
Q(parent, a) = maximum alpha score observed in the subtree rooted at this node (Eq. 10).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..evaluation.evaluator import EvaluationResult

_node_ids = itertools.count()


@dataclass
class RefinementStep:
    dimension: str
    suggestion: str
    expression_before: str
    expression_after: str
    scores_before: dict[str, float]
    scores_after: dict[str, float]

    def summary(self) -> str:
        def fmt(scores: dict[str, float]) -> str:
            return ", ".join(f"{k}={v:.2f}" for k, v in scores.items())

        return (
            f"[{self.dimension}] {self.suggestion}\n"
            f"    {self.expression_before}  ->  {self.expression_after}\n"
            f"    scores before: {fmt(self.scores_before)}\n"
            f"    scores after:  {fmt(self.scores_after)}"
        )


@dataclass
class MCTSNode:
    result: EvaluationResult
    parent: "MCTSNode | None" = None
    refinement: RefinementStep | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 1
    edge_q: float = float("-inf")  # Q(parent -> self); -inf until first backprop
    node_id: int = field(default_factory=lambda: next(_node_ids))

    @property
    def score(self) -> float:
        return self.result.overall

    @property
    def depth(self) -> int:
        d, node = 0, self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    # ------------------------------------------------------------------
    # Refinement history (cumulative, root -> this node)
    # ------------------------------------------------------------------

    def history_steps(self) -> list[RefinementStep]:
        steps: list[RefinementStep] = []
        node: MCTSNode | None = self
        while node is not None:
            if node.refinement is not None:
                steps.append(node.refinement)
            node = node.parent
        return list(reversed(steps))

    def history_text(self) -> str:
        steps = self.history_steps()
        if not steps:
            return ""
        return "\n".join(f"Step {i + 1}: {s.summary()}" for i, s in enumerate(steps))

    def context_text(self) -> str:
        """Rich refinement context for the LLM: own path, plus sibling and child attempts."""
        blocks: list[str] = []
        own = self.history_text()
        if own:
            blocks.append("Refinement path from the root to this alpha:\n" + own)
        if self.parent is not None:
            siblings = [
                c.refinement.summary()
                for c in self.parent.children
                if c is not self and c.refinement is not None
            ]
            if siblings:
                blocks.append("Refinements already tried on the parent alpha (siblings):\n" + "\n".join(siblings))
        child_steps = [c.refinement.summary() for c in self.children if c.refinement is not None]
        if child_steps:
            blocks.append("Refinements already tried on this alpha (children):\n" + "\n".join(child_steps))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------

    def iter_subtree(self):
        yield self
        for child in self.children:
            yield from child.iter_subtree()
