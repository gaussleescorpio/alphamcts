"""Heuristic (LLM-free) overfitting risk scorer.

Drop-in replacement for the paper's Figure-17 LLM assessment: scores the same
0-10 scale (10 = very low risk) from measurable structural proxies instead of a
Composer round-trip, which roughly halves the LLM calls per candidate:

- operator count and nesting depth (unjustified complexity);
- number of window parameters and how many look hand-tuned (not conventional
  lookbacks such as 5/10/20/60), a data-dredging signature;
- length of the refinement history (many revisions = heavier optimization).

The statistical admission gates (HAC, block/year stability, negative control,
reconstruction R^2) remain the hard defense; this score only steers MCTS search.
"""

from __future__ import annotations

import re

# Conventional lookback windows commonly used in the literature; other values
# (e.g. 47, 53) suggest the parameter was tuned to the sample.
_CONVENTIONAL_WINDOWS = {1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 21, 25, 30, 40, 50, 60, 63, 90, 120, 250}


def heuristic_overfit_score(expression: str, refinement_history: str) -> tuple[float, str]:
    n_ops = len(re.findall(r"[A-Za-z_]+\(", expression))

    depth = max_depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1

    windows = [int(m) for m in re.findall(r"(?<![\w.])(\d+)(?![\w.])", expression)]
    n_windows = len(windows)
    n_tuned = sum(1 for w in windows if w not in _CONVENTIONAL_WINDOWS)

    n_steps = len(re.findall(r"^Step \d+:", refinement_history, flags=re.M))

    penalty = (
        0.35 * max(0, n_ops - 4)
        + 0.40 * max(0, max_depth - 3)
        + 0.30 * max(0, n_windows - 3)
        + min(0.60 * n_tuned, 2.4)
        + 0.50 * max(0, n_steps - 2)
    )
    score = max(0.0, min(10.0, 10.0 - penalty))
    reason = (
        f"heuristic: {n_ops} operators (depth {max_depth}), {n_windows} windows "
        f"({n_tuned} non-conventional), {n_steps} refinement steps"
    )
    return score, reason
