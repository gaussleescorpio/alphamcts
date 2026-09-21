"""Frequent Subtree Avoidance (FSA).

Implements the paper's Section 3 "Frequent Subtree Avoidance":

- a *root gene* is a subtree of an alpha's expression tree whose leaves are exclusively raw
  input features;
- the abstraction operator Abs(.) replaces concrete parameter values with a placeholder
  (``Ma(vwap, 20)`` -> ``Ma(vwap, t)``);
- frequent *closed* root genes (no immediate supertree with identical support) mined from the
  effective alpha repository become forbidden structures for subsequent generation.
"""

from __future__ import annotations

from collections import defaultdict

from .formula import ExprNode


def abstract_root_genes(tree: ExprNode) -> set[str]:
    """All abstracted root genes of an expression tree (operator subtrees only)."""
    genes: set[str] = set()
    abstracted = tree.abstracted()
    for node, _parent in abstracted.iter_subtrees():
        if not node.is_leaf:
            genes.add(node.render())
    return genes


def mine_frequent_subtrees(
    trees: list[ExprNode],
    top_k: int = 3,
    min_support: float = 0.3,
) -> list[str]:
    """Mine the top-k most frequent closed root genes across alpha expression trees.

    Returns rendered abstracted subtree strings, ordered by (support desc, size desc).
    """
    if not trees:
        return []

    n = len(trees)
    containing: dict[str, set[int]] = defaultdict(set)   # gene -> alpha indices containing it
    parents: dict[str, set[str]] = defaultdict(set)      # gene -> observed immediate supertrees
    sizes: dict[str, int] = {}

    for idx, tree in enumerate(trees):
        abstracted = tree.abstracted()
        for node, parent in abstracted.iter_subtrees():
            if node.is_leaf:
                continue
            key = node.render()
            containing[key].add(idx)
            sizes.setdefault(key, node.size())
            if parent is not None and not parent.is_leaf:
                parents[key].add(parent.render())

    support = {gene: len(ids) / n for gene, ids in containing.items()}

    def is_closed(gene: str) -> bool:
        return all(support.get(p, 0.0) < support[gene] for p in parents.get(gene, ()))

    candidates = [g for g, s in support.items() if s >= min_support and is_closed(g)]
    candidates.sort(key=lambda g: (-support[g], -sizes[g], g))
    return candidates[:top_k]


def find_forbidden_gene(tree: ExprNode, forbidden: list[str]) -> str | None:
    """Return the first forbidden root gene contained in the tree, or None."""
    if not forbidden:
        return None
    genes = abstract_root_genes(tree)
    for gene in forbidden:
        if gene in genes:
            return gene
    return None
