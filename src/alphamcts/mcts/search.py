"""LLM-guided MCTS alpha mining (paper Algorithm 1).

One :class:`MCTSTree` implements a single search tree:

- **Selection** via UCT (Eq. 2), augmented with a *virtual expansion action* whose visit
  count is ``1 + |children|`` so any node (not just leaves) can be selected for expansion;
- **Expansion**: sample a refinement dimension via softmax over score deficits (Eq. 3),
  ask the LLM for a targeted suggestion, then for a refined formula (validated with an
  iterative correction loop and the FSA constraint, Eq. 12);
- **Evaluation** by multi-dimensional backtesting (Eq. 6-8);
- **Backpropagation** with max-updates of Q (Eq. 9-10);
- **Dynamic budget**: initial budget B, incremented by b on every in-tree score breakthrough.

:class:`AlphaMiner` orchestrates repeated trees, reviews each finished tree against the
effective-alpha criteria, updates the zoo and re-mines FSA forbidden structures.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

from ..evaluation.evaluator import (
    DIMENSION_DESCRIPTIONS,
    DIMENSIONS,
    EvaluationResult,
    MultiDimEvaluator,
)
from ..expression.formula import AlphaFormula, FormulaError
from ..expression.subtree import find_forbidden_gene
from ..llm.base import AlphaAgent
from ..zoo import AlphaRecord, AlphaZoo
from .node import MCTSNode, RefinementStep

logger = logging.getLogger("alphamcts")


@dataclass
class MiningStats:
    generated: int = 0          # unique candidate formulas evaluated (search count)
    invalid: int = 0            # candidates that never produced a computable alpha
    added_to_zoo: int = 0
    trees: int = 0
    breakthroughs: int = 0
    expressions: list[str] = field(default_factory=list)


class MCTSTree:
    def __init__(self, root: MCTSNode, exploration_weight: float = 1.0):
        self.root = root
        self.c = exploration_weight

    def select(self) -> tuple[MCTSNode, list[tuple[MCTSNode, MCTSNode]]]:
        """Descend by UCT; return (node to expand, path of (parent, child) edges)."""
        node = self.root
        path: list[tuple[MCTSNode, MCTSNode]] = []
        while True:
            if not node.children:
                return node, path
            log_n = math.log(max(node.visits, 1))
            best_child: MCTSNode | None = None
            best_uct = float("-inf")
            for child in node.children:
                q = child.edge_q if child.edge_q != float("-inf") else child.score
                uct = q + self.c * math.sqrt(log_n / child.visits)
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            # Virtual expansion action a_e: N_{s'_e} = 1 + |C(s)|, Q approximated by the
            # node's own alpha score (expanding s refines the alpha at s).
            n_virtual = 1 + len(node.children)
            uct_virtual = node.score + self.c * math.sqrt(log_n / n_virtual)
            if uct_virtual >= best_uct or best_child is None:
                return node, path
            path.append((node, best_child))
            node = best_child

    @staticmethod
    def backpropagate(path: list[tuple[MCTSNode, MCTSNode]], new_node: MCTSNode, reward: float) -> None:
        for parent, child in path:
            parent.visits += 1
            child.edge_q = max(child.edge_q, reward)
        # edge from the expanded node to the new node
        if new_node.parent is not None:
            new_node.parent.visits += 1
        new_node.edge_q = max(new_node.edge_q, reward)

    def nodes(self) -> list[MCTSNode]:
        return list(self.root.iter_subtree())


class AlphaMiner:
    def __init__(
        self,
        agent: AlphaAgent,
        evaluator: MultiDimEvaluator,
        zoo: AlphaZoo,
        mcts_cfg: dict | None = None,
        mining_cfg: dict | None = None,
        fsa_cfg: dict | None = None,
        llm_cfg: dict | None = None,
        seed: int = 0,
    ):
        mcts_cfg = mcts_cfg or {}
        mining_cfg = mining_cfg or {}
        fsa_cfg = fsa_cfg or {}
        llm_cfg = llm_cfg or {}

        self.agent = agent
        self.evaluator = evaluator
        self.zoo = zoo

        self.c = float(mcts_cfg.get("exploration_weight", 1.0))
        self.initial_budget = int(mcts_cfg.get("initial_budget", 3))
        self.budget_increment = int(mcts_cfg.get("budget_increment", 1))
        self.max_budget = int(mcts_cfg.get("max_budget", 12))
        self.dim_temperature = float(mcts_cfg.get("dimension_temperature", 1.0))
        self.e_max = float(mcts_cfg.get("e_max", 1.0))

        self.n_trees = int(mining_cfg.get("n_trees", 10))
        self.target_zoo_size = int(mining_cfg.get("target_zoo_size", 20))
        self.parallel_trees = int(mining_cfg.get("parallel_trees", 1))

        self.fsa_top_k = int(fsa_cfg.get("top_k", 3))
        self.fsa_min_support = float(fsa_cfg.get("min_support", 0.3))
        self.fsa_min_alphas = int(fsa_cfg.get("min_alphas", 5))

        self.few_shot_k = int(llm_cfg.get("few_shot_k", 1))
        self.corr_filter_eta = float(llm_cfg.get("correlation_filter_eta", 0.5))

        self.rng = np.random.default_rng(seed)
        self.forbidden: list[str] = []
        self.stats = MiningStats()

        # Parallel mining locks: `_lock` guards the zoo/FSA/checkpoint critical section,
        # `_stats_lock` the cheap counters, `_rng_lock` the shared numpy Generator.
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._rng_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Top-level mining loop
    # ------------------------------------------------------------------

    def run(self, checkpoint=None) -> MiningStats:
        """Mine up to n_trees trees; `checkpoint(stats)` is called after every tree.

        With mining.parallel_trees > 1, that many trees run concurrently: MCTS is
        sequential *within* a tree, but trees are independent (the paper's outer
        loop), so their LLM round-trips overlap. Zoo admission, FSA updates and
        checkpoints are serialized under a lock.
        """
        if self.parallel_trees <= 1:
            for tree_idx in range(self.n_trees):
                if len(self.zoo) >= self.target_zoo_size:
                    logger.info("Target zoo size reached (%d); stopping.", len(self.zoo))
                    break
                try:
                    tree = self._run_tree(tree_idx)
                except Exception as exc:  # noqa: BLE001 - one tree must not kill a long run
                    logger.warning("Tree %d aborted (%s): %s", tree_idx, type(exc).__name__, exc)
                    continue
                self._finish_tree(tree, checkpoint)
            return self.stats

        indices = iter(range(self.n_trees))
        idx_lock = threading.Lock()

        def next_index() -> int | None:
            with idx_lock:
                return next(indices, None)

        def worker() -> None:
            while True:
                with self._lock:
                    if len(self.zoo) >= self.target_zoo_size:
                        logger.info("Target zoo size reached (%d); worker stopping.", len(self.zoo))
                        return
                tree_idx = next_index()
                if tree_idx is None:
                    return
                try:
                    tree = self._run_tree(tree_idx)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tree %d aborted (%s): %s", tree_idx, type(exc).__name__, exc)
                    continue
                self._finish_tree(tree, checkpoint)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_trees) as pool:
            futures = [pool.submit(worker) for _ in range(self.parallel_trees)]
            for future in futures:
                future.result()  # propagate unexpected worker crashes
        return self.stats

    def _finish_tree(self, tree: MCTSTree, checkpoint) -> None:
        """Review a finished tree and update shared state (serialized)."""
        with self._lock:
            self.stats.trees += 1
            self._review_tree(tree)
            self.forbidden = self.zoo.forbidden_subtrees(
                top_k=self.fsa_top_k,
                min_support=self.fsa_min_support,
                min_alphas=self.fsa_min_alphas,
            )
            if self.forbidden:
                logger.info("FSA forbidden structures: %s", self.forbidden)
            if checkpoint is not None:
                checkpoint(self.stats)

    # ------------------------------------------------------------------
    # One MCTS tree (Algorithm 1)
    # ------------------------------------------------------------------

    def _run_tree(self, tree_idx: int) -> MCTSTree:
        root_result = self._generate_root()
        root = MCTSNode(result=root_result)
        tree = MCTSTree(root, exploration_weight=self.c)
        logger.info(
            "Tree %d root: %s  (S=%.3f, RankIC=%.4f)",
            tree_idx, root_result.expression, root_result.overall, root_result.metrics.rank_ic,
        )

        budget = self.initial_budget
        s_max = root.score
        expansions = 0
        while expansions < budget:
            expansions += 1
            node, path = tree.select()
            new_node = self._expand(node)
            if new_node is None:
                continue
            tree_path = path  # edges root -> selected node
            node.children.append(new_node)
            reward = new_node.score
            MCTSTree.backpropagate(tree_path, new_node, reward)
            logger.info(
                "Tree %d expand #%d (depth %d, dim=%s): %s  (S=%.3f, RankIC=%.4f)",
                tree_idx, expansions, new_node.depth,
                new_node.refinement.dimension if new_node.refinement else "-",
                new_node.result.expression, reward, new_node.result.metrics.rank_ic,
            )
            if reward > s_max:
                s_max = reward
                if budget < self.max_budget:
                    budget = min(budget + self.budget_increment, self.max_budget)
                    with self._stats_lock:
                        self.stats.breakthroughs += 1
                    logger.info("Tree %d breakthrough: budget extended to %d", tree_idx, budget)
        return tree

    def _generate_root(self) -> EvaluationResult:
        last_error: FormulaError | None = None
        for _ in range(3):
            try:
                portrait = self.agent.generate_portrait(self.forbidden)
                formula = self.agent.generate_formula(portrait)
                formula = self._enforce_fsa(formula, portrait)
                result = self.evaluator.evaluate(formula, self.agent.assess_overfitting, "")
                with self._stats_lock:
                    self.stats.generated += 1
                    self.stats.expressions.append(result.expression)
                return result
            except FormulaError as exc:
                last_error = exc
                with self._stats_lock:
                    self.stats.invalid += 1
        raise last_error or FormulaError("Failed to generate a root alpha.")

    # ------------------------------------------------------------------
    # Expansion (dimension sampling + LLM refinement + evaluation)
    # ------------------------------------------------------------------

    def _sample_dimension(self, scores: dict[str, float]) -> str:
        deficits = np.array([self.e_max - scores.get(d, 0.0) for d in DIMENSIONS])
        logits = deficits / max(self.dim_temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        with self._rng_lock:  # numpy Generator is not thread-safe
            return str(self.rng.choice(DIMENSIONS, p=probs))

    def _exemplar_lines(self, dimension: str, node: MCTSNode) -> list[str]:
        records = self.zoo.exemplars(
            dimension, node.result.values, k=self.few_shot_k, eta=self.corr_filter_eta
        )
        return [
            f"{r.expression} | RankIC={r.metrics.rank_ic:.4f}, RankIR={r.metrics.rank_ir:.2f}, "
            f"turnover={r.metrics.turnover:.2f}"
            for r in records
        ]

    def _expand(self, node: MCTSNode) -> MCTSNode | None:
        dimension = self._sample_dimension(node.result.scores)
        suggestions = self.agent.suggest_refinements(
            dimension=dimension,
            dimension_description=DIMENSION_DESCRIPTIONS[dimension],
            alpha_expression=node.result.expression,
            alpha_description=node.result.formula.description,
            evaluation_scores=node.result.scores,
            backtest_metrics=node.result.metrics.to_dict(),
            refinement_context=node.context_text(),
            examples=self._exemplar_lines(dimension, node),
            forbidden_subtrees=self.forbidden,
        )
        try:
            portrait = self.agent.refine_portrait(node.result.expression, suggestions, self.forbidden)
            formula = self.agent.generate_formula(portrait)
            formula = self._enforce_fsa(formula, portrait)
            result = self.evaluator.evaluate(
                formula, self.agent.assess_overfitting, node.history_text()
            )
        except FormulaError as exc:
            with self._stats_lock:
                self.stats.invalid += 1
            logger.debug("Expansion failed: %s", exc)
            return None
        with self._stats_lock:
            self.stats.generated += 1
            self.stats.expressions.append(result.expression)
        refinement = RefinementStep(
            dimension=dimension,
            suggestion="; ".join(suggestions),
            expression_before=node.result.expression,
            expression_after=result.expression,
            scores_before=dict(node.result.scores),
            scores_after=dict(result.scores),
        )
        return MCTSNode(result=result, parent=node, refinement=refinement)

    def _enforce_fsa(self, formula: AlphaFormula, portrait: dict) -> AlphaFormula:
        """FSA constraint (Eq. 12): one retry if a forbidden root gene is present."""
        gene = find_forbidden_gene(formula.tree(), self.forbidden)
        if gene is None:
            return formula
        try:
            retry_portrait = self.agent.refine_portrait(
                formula.to_string(),
                [f"The structure '{gene}' is forbidden (over-mined); restructure the alpha to avoid it."],
                self.forbidden,
            )
            retry = self.agent.generate_formula(retry_portrait)
            if find_forbidden_gene(retry.tree(), self.forbidden) is None:
                return retry
        except FormulaError:
            pass
        logger.debug("FSA: keeping formula containing forbidden gene %s", gene)
        return formula

    # ------------------------------------------------------------------
    # Post-tree review: add effective alphas to the zoo
    # ------------------------------------------------------------------

    def _review_tree(self, tree: MCTSTree) -> None:
        for node in tree.nodes():
            r = node.result
            record = AlphaRecord(
                name=r.formula.name,
                description=r.formula.description,
                expression=r.expression,
                formula=r.formula,
                args=r.chosen_args,
                metrics=r.metrics,
                scores=dict(r.scores),
                overall=r.overall,
                overfit_reason=r.overfit_reason,
                values=r.values,
            )
            ok, reasons = self.zoo.add(record)
            if ok:
                self.stats.added_to_zoo += 1
                logger.info("Zoo += %s (RankIC=%.4f, RankIR=%.2f) [size=%d]",
                            r.expression, r.metrics.rank_ic, r.metrics.rank_ir, len(self.zoo))
            else:
                logger.debug("Rejected %s: %s", r.expression, "; ".join(reasons))
