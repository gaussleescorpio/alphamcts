import pytest

from alphamcts.data.base import FIELDS
from alphamcts.expression.formula import AlphaFormula
from alphamcts.llm import AlphaAgent, MockLLM, extract_json
from alphamcts.llm.heuristic import heuristic_overfit_score
from alphamcts.llm.mock import TEMPLATES, WINDOW_RANGE


@pytest.fixture
def agent():
    return AlphaAgent(MockLLM(seed=3), FIELDS, WINDOW_RANGE)


def test_heuristic_prefers_simple_conventional_formulas():
    simple, _ = heuristic_overfit_score("Neg(Pct(close,5))", "")
    complex_, _ = heuristic_overfit_score(
        "Rank((Tanh(Ma(Zscore(Corr(Rank(Diff(close,47),53),Rank(volume,29),17),23),11))"
        "*Neg(Std(Pct(close,7),37))),19)",
        "",
    )
    assert simple > complex_
    assert simple >= 9.0
    assert complex_ <= 5.0


def test_heuristic_penalizes_long_refinement_history():
    history = "\n".join(f"Step {i}: tweaked window" for i in range(1, 7))
    fresh, _ = heuristic_overfit_score("Neg(Pct(close,5))", "")
    tuned, _ = heuristic_overfit_score("Neg(Pct(close,5))", history)
    assert fresh > tuned


def test_heuristic_scores_are_bounded():
    for expr in ("close", "Neg(Pct(close,5))", "Rank(" * 30 + "close" + ",13)" * 30):
        s, reason = heuristic_overfit_score(expr, "")
        assert 0.0 <= s <= 10.0
        assert reason


def test_agent_heuristic_mode_skips_llm():
    class ExplodingLLM:
        def complete(self, prompt: str, temperature: float = 1.0) -> str:
            raise AssertionError("LLM must not be called in heuristic mode")

    agent = AlphaAgent(
        ExplodingLLM(), FIELDS, WINDOW_RANGE, overfit_scorer="heuristic"
    )
    score, reason = agent.assess_overfitting("Neg(Pct(close,5))", "")
    assert 0.0 <= score <= 10.0
    assert "heuristic" in reason


def test_agent_rejects_unknown_overfit_scorer():
    with pytest.raises(ValueError):
        AlphaAgent(MockLLM(seed=0), FIELDS, WINDOW_RANGE, overfit_scorer="magic")


def test_extract_json_from_noisy_text():
    text = 'Sure! Here is the alpha:\n```json\n{"a": 1, "b": {"c": [1, 2]}}\n```\nHope it helps.'
    assert extract_json(text) == {"a": 1, "b": {"c": [1, 2]}}


def test_portrait_then_formula_roundtrip(agent):
    portrait = agent.generate_portrait(forbidden_subtrees=[])
    assert {"name", "description", "pseudo_code"} <= set(portrait)
    formula = agent.generate_formula(portrait)
    assert isinstance(formula, AlphaFormula)
    assert formula.name == portrait["name"]
    assert 1 <= len(formula.arguments) <= 3


def test_all_mock_templates_are_valid():
    import numpy as np

    rng = np.random.default_rng(0)
    for template in TEMPLATES:
        obj = template(rng)
        f = AlphaFormula.from_json(obj, FIELDS, WINDOW_RANGE)
        assert f.n_operators() >= 1


def test_correction_loop_fixes_invalid_output():
    client = MockLLM(seed=1, invalid_rate=1.0)  # first formula response is always invalid
    agent = AlphaAgent(client, FIELDS, WINDOW_RANGE, max_correction_attempts=2)
    portrait = agent.generate_portrait([])
    formula = agent.generate_formula(portrait)  # must survive via the correction loop
    assert isinstance(formula, AlphaFormula)


def test_forbidden_subtree_avoidance(agent):
    from alphamcts.expression.subtree import abstract_root_genes, find_forbidden_gene

    forbidden = ["Pct(close,t)"]
    portrait = agent.generate_portrait(forbidden_subtrees=forbidden)
    formula = agent.generate_formula(portrait)
    assert find_forbidden_gene(formula.tree(), forbidden) is None


def test_refinement_produces_new_formula(agent):
    portrait = agent.generate_portrait([])
    original = agent.generate_formula(portrait)
    refined_portrait = agent.refine_portrait(
        original.to_string(), ["Smooth the signal with a moving average."], []
    )
    refined = agent.generate_formula(refined_portrait)
    assert isinstance(refined, AlphaFormula)


def test_suggestions_and_overfitting(agent):
    suggestions = agent.suggest_refinements(
        dimension="turnover",
        dimension_description="reduce daily churn",
        alpha_expression="Neg(Pct(close,5))",
        alpha_description="reversal",
        evaluation_scores={"turnover": 0.2},
        backtest_metrics={"turnover": 1.5},
        refinement_context="",
        examples=[],
        forbidden_subtrees=[],
    )
    assert suggestions and all(isinstance(s, str) for s in suggestions)

    score, reason = agent.assess_overfitting("Neg(Pct(close,5))", "")
    assert 0.0 <= score <= 10.0
    assert reason
