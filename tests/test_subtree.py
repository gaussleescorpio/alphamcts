from alphamcts.data.base import FIELDS
from alphamcts.expression.formula import AlphaFormula
from alphamcts.expression.subtree import abstract_root_genes, find_forbidden_gene, mine_frequent_subtrees

WINDOW_RANGE = (2, 60)


def make(steps, args):
    return AlphaFormula.from_json(
        {"name": "a", "description": "", "formula": steps, "arguments": [args]},
        FIELDS,
        WINDOW_RANGE,
    )


def ma_close(window):
    return make([{"name": "Ma", "param": ["w"], "input": ["close"], "output": "o"}], {"w": window})


def test_abstraction_ignores_parameter_values():
    t1 = ma_close(5).tree(args={"w": 5})
    t2 = ma_close(50).tree(args={"w": 50})
    assert abstract_root_genes(t1) == abstract_root_genes(t2) == {"Ma(close,t)"}


def test_root_genes_of_composite():
    f = make(
        [
            {"name": "Sub", "param": [], "input": ["close", "vwap"], "output": "p"},
            {"name": "Ma", "param": ["w"], "input": ["p"], "output": "o"},
        ],
        {"w": 20},
    )
    genes = abstract_root_genes(f.tree())
    assert genes == {"(close-vwap)", "Ma((close-vwap),t)"}


def test_mine_frequent_closed_subtrees():
    # 4 alphas share Ma(close, t); 3 of those also share Zscore(Ma(close,t),t).
    common = [
        make(
            [
                {"name": "Ma", "param": ["w"], "input": ["close"], "output": "m"},
                {"name": "Zscore", "param": ["z"], "input": ["m"], "output": "o"},
            ],
            {"w": w, "z": 10},
        )
        for w in (5, 10, 20)
    ]
    solo = ma_close(15)
    other = make([{"name": "Std", "param": ["w"], "input": ["volume"], "output": "o"}], {"w": 10})

    trees = [f.tree() for f in common + [solo, other]]
    mined = mine_frequent_subtrees(trees, top_k=3, min_support=0.5)

    # Ma(close,t) has support 0.8; Zscore(Ma(close,t),t) has support 0.6.
    assert mined[0] == "Ma(close,t)"
    assert "Zscore(Ma(close,t),t)" in mined
    # Std(volume,t) support 0.2 < min_support
    assert "Std(volume,t)" not in mined


def test_closedness_filters_redundant_subtree():
    # Every alpha containing Ma(close,t) also wraps it in Zscore -> Ma(close,t) is not closed.
    alphas = [
        make(
            [
                {"name": "Ma", "param": ["w"], "input": ["close"], "output": "m"},
                {"name": "Zscore", "param": ["z"], "input": ["m"], "output": "o"},
            ],
            {"w": w, "z": 10},
        )
        for w in (5, 10)
    ]
    mined = mine_frequent_subtrees([f.tree() for f in alphas], top_k=5, min_support=0.5)
    assert "Zscore(Ma(close,t),t)" in mined
    assert "Ma(close,t)" not in mined


def test_find_forbidden_gene():
    f = ma_close(5)
    assert find_forbidden_gene(f.tree(), ["Ma(close,t)"]) == "Ma(close,t)"
    assert find_forbidden_gene(f.tree(), ["Std(close,t)"]) is None
