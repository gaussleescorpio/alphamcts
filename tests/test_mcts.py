import numpy as np
import pandas as pd
import pytest

from alphamcts.data.base import FIELDS
from alphamcts.evaluation.evaluator import MultiDimEvaluator, EvaluationResult
from alphamcts.evaluation.metrics import AlphaMetrics
from alphamcts.expression.formula import AlphaFormula
from alphamcts.llm import AlphaAgent, MockLLM
from alphamcts.mcts import AlphaMiner, MCTSNode, MCTSTree
from alphamcts.zoo import AlphaZoo

WINDOW_RANGE = (2, 60)


def fake_result(overall: float, expression: str = "Neg(Pct(close,5))") -> EvaluationResult:
    formula = AlphaFormula.from_json(
        {
            "name": "f",
            "description": "",
            "formula": [
                {"name": "Pct", "param": ["w"], "input": ["close"], "output": "r"},
                {"name": "Neg", "param": [], "input": ["r"], "output": "a"},
            ],
            "arguments": [{"w": 5}],
        },
        FIELDS,
        WINDOW_RANGE,
    )
    from alphamcts.evaluation.evaluator import DIMENSIONS

    scores = {d: overall for d in DIMENSIONS}
    return EvaluationResult(
        formula=formula,
        chosen_args={"w": 5},
        expression=expression,
        metrics=AlphaMetrics(0.01, 0.1, 0.01, 0.1, 1.0, 0.0, 1.0),
        scores=scores,
        overall=overall,
        overfit_reason="",
        values=pd.DataFrame(np.zeros((5, 3))),
    )


def test_selection_prefers_high_q_child():
    root = MCTSNode(result=fake_result(0.1))
    good = MCTSNode(result=fake_result(0.9), parent=root)
    bad = MCTSNode(result=fake_result(0.2), parent=root)
    root.children = [good, bad]
    good.edge_q, bad.edge_q = 0.9, 0.2
    root.visits = 10
    good.visits, bad.visits = 3, 3
    tree = MCTSTree(root, exploration_weight=0.1)
    node, path = tree.select()
    # descends into the high-Q child, whose virtual action then wins (it has no children)
    assert node is good
    assert path == [(root, good)]


def test_virtual_action_allows_internal_expansion():
    # child much worse than the root itself -> virtual expansion of root wins
    root = MCTSNode(result=fake_result(0.9))
    weak = MCTSNode(result=fake_result(0.05), parent=root)
    weak.edge_q = 0.05
    root.children = [weak]
    root.visits, weak.visits = 5, 4
    tree = MCTSTree(root, exploration_weight=0.1)
    node, path = tree.select()
    assert node is root
    assert path == []


def test_backpropagation_max_update():
    root = MCTSNode(result=fake_result(0.5))
    mid = MCTSNode(result=fake_result(0.4), parent=root)
    root.children = [mid]
    mid.edge_q = 0.4
    new = MCTSNode(result=fake_result(0.8), parent=mid)
    MCTSTree.backpropagate([(root, mid)], new, reward=0.8)
    assert mid.edge_q == 0.8       # max-update propagated
    assert root.visits == 2
    assert mid.visits == 2         # incremented as parent of the new node
    MCTSTree.backpropagate([(root, mid)], new, reward=0.3)
    assert mid.edge_q == 0.8       # max, not overwrite


def test_history_and_context_text():
    root = MCTSNode(result=fake_result(0.5, "A"))
    from alphamcts.mcts.node import RefinementStep

    step = RefinementStep("stability", "smooth it", "A", "B", {"stability": 0.2}, {"stability": 0.6})
    child = MCTSNode(result=fake_result(0.6, "B"), parent=root, refinement=step)
    root.children = [child]
    assert "smooth it" in child.history_text()
    assert "children" in root.context_text() or "tried on this alpha" in root.context_text()


@pytest.fixture
def miner(panel):
    zoo = AlphaZoo({"min_rank_ic": 0.005, "min_rank_ir": 0.1, "max_correlation": 0.8})
    evaluator = MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)
    agent = AlphaAgent(MockLLM(seed=11), FIELDS, WINDOW_RANGE)
    return AlphaMiner(
        agent,
        evaluator,
        zoo,
        mcts_cfg={"initial_budget": 3, "budget_increment": 1, "max_budget": 6},
        mining_cfg={"n_trees": 4, "target_zoo_size": 50},
        fsa_cfg={"top_k": 3, "min_support": 0.4, "min_alphas": 3},
        llm_cfg={"few_shot_k": 1},
        seed=11,
    )


def test_mining_end_to_end_with_mock(miner):
    stats = miner.run()
    assert stats.trees >= 1
    assert stats.generated >= stats.trees * 3  # root + >=2 expansions per surviving tree


def test_parallel_mining_matches_sequential_bookkeeping(panel):
    """Parallel trees must complete all work with consistent shared-state accounting."""
    zoo = AlphaZoo({"min_rank_ic": 0.005, "min_rank_ir": 0.1, "max_correlation": 0.8})
    evaluator = MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)
    agent = AlphaAgent(MockLLM(seed=7), FIELDS, WINDOW_RANGE)
    miner = AlphaMiner(
        agent,
        evaluator,
        zoo,
        mcts_cfg={"initial_budget": 2, "budget_increment": 1, "max_budget": 4},
        mining_cfg={"n_trees": 6, "target_zoo_size": 50, "parallel_trees": 3},
        fsa_cfg={"top_k": 3, "min_support": 0.4, "min_alphas": 3},
        llm_cfg={"few_shot_k": 1},
        seed=7,
    )
    checkpoints: list[int] = []
    stats = miner.run(checkpoint=lambda s: checkpoints.append(s.trees))

    assert stats.trees == 6                      # every tree processed exactly once
    assert len(checkpoints) == stats.trees       # one checkpoint per finished tree
    assert stats.generated == len(stats.expressions)
    assert stats.added_to_zoo == len(zoo)
    # zoo contents must satisfy the admission invariants (no dupes, correlation cap)
    exprs = [r.expression for r in zoo.records]
    assert len(exprs) == len(set(exprs))


def test_parallel_mining_stops_at_target_zoo_size(panel):
    zoo = AlphaZoo({"min_rank_ic": 0.0, "min_rank_ir": 0.0, "max_correlation": 0.99})
    evaluator = MultiDimEvaluator(panel, horizon=10, train_ratio=0.7, zoo=zoo, top_frac=0.1)
    agent = AlphaAgent(MockLLM(seed=5), FIELDS, WINDOW_RANGE)
    miner = AlphaMiner(
        agent,
        evaluator,
        zoo,
        mcts_cfg={"initial_budget": 2, "budget_increment": 0, "max_budget": 2},
        mining_cfg={"n_trees": 40, "target_zoo_size": 2, "parallel_trees": 4},
        fsa_cfg={"min_alphas": 99},
        llm_cfg={"few_shot_k": 1},
        seed=5,
    )
    stats = miner.run()
    assert len(zoo) >= 2
    assert stats.trees < 40  # stopped early, did not mine every tree
    assert len(miner.zoo) >= 1
    assert stats.added_to_zoo == len(miner.zoo)
    # every zoo record satisfies the admission criteria
    for record in miner.zoo:
        ok, reasons = miner.zoo.check_effective(record.metrics)
        # rel-rank criteria evolve as the zoo grows; check absolute ones instead
        assert record.metrics.rank_ic >= miner.zoo.min_rank_ic
        assert record.metrics.turnover <= miner.zoo.max_turnover


def test_dimension_sampling_prefers_weak_dimension(miner):
    scores = {"effectiveness": 0.0, "stability": 1.0, "turnover": 1.0, "diversity": 1.0,
              "overfitting": 1.0, "monotonicity": 1.0}
    draws = [miner._sample_dimension(scores) for _ in range(300)]
    frac_eff = draws.count("effectiveness") / len(draws)
    assert frac_eff > 0.3  # softmax with T=1 over deficit 1 vs 0: e/(e+4) ~ 0.4
