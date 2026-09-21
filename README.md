# AlphaMCTS

Replication of the AAAI-26 paper [*Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for
Formulaic Alpha Factor Mining*](https://doi.org/10.1609/aaai.v40i2.37069) (arXiv:
[2505.11122](https://arxiv.org/abs/2505.11122)).

The framework mines formulaic alpha factors by combining an LLM (as a generative prior over symbolic
formulas) with Monte Carlo Tree Search guided by multi-dimensional backtesting feedback:

1. **Formulaic alphas** are expression trees over ~30 operators (Ma, Std, Corr, Zscore, ...) and raw
   OHLCV+VWAP fields, represented in the paper's structured JSON schema.
2. **MCTS search**: UCT selection (with a virtual expansion action so internal nodes remain
   expandable), softmax sampling of the weakest evaluation dimension, two-step LLM generation
   (refinement suggestion -> concrete formula), syntax validation with iterative correction, and
   max-backpropagation of the alpha score.
3. **Multi-dimensional evaluation**: Effectiveness (RankIC), Stability (RankIR), Turnover,
   Diversity (max correlation with the zoo) scored by relative rank against the alpha zoo, plus an
   LLM-judged Overfitting Risk score. The mean is the MCTS reward.
4. **Frequent Subtree Avoidance (FSA)**: frequent closed "root genes" (parameter-abstracted
   subtrees) mined from the zoo become forbidden structures in generation prompts.
5. **Dynamic budget**: each tree starts with budget B=3; every in-tree score breakthrough adds b=1.
6. **Downstream**: top-k alphas by RankIR feed LightGBM/MLP combination models, evaluated with a
   top-k/drop-n backtest (IC / RankIC / AER / IR).

## Install

```bash
conda create -n alphamcts python=3.11 -y
conda activate alphamcts
pip install -r requirements.txt
pip install -e .
```

## Quick start (offline, MockLLM + synthetic data)

```bash
# Mine alphas into runs/demo/
alphamcts mine --config configs/default.yaml --out runs/demo

# Inspect the zoo
alphamcts report --run runs/demo

# Train combination models on the mined alpha set and backtest
alphamcts evaluate --run runs/demo
```

## Using Cursor Composer as the LLM

Set `llm.backend: cursor` in the config (or pass `--llm cursor`), and export a Cursor API key:

```bash
cp .env.example .env   # fill in CURSOR_API_KEY
export CURSOR_API_KEY=cursor_...
alphamcts mine --config configs/default.yaml --llm cursor --out runs/composer
```

The Cursor backend calls `cursor-sdk`'s `Agent.prompt` with the `composer-2.5` model (configurable
via `llm.model`).

## Data

The data layer is pluggable (`src/alphamcts/data/`):

- `synthetic` (default): a synthetic market with embedded reversal / price-volume structure so the
  pipeline is runnable end-to-end offline.
- `csv`: a directory of per-instrument CSVs with columns
  `date,open,high,low,close,volume[,vwap]`.

A Qlib adapter can be added later behind the same `MarketPanel` interface.

## Layout

```
configs/default.yaml      hyperparameters (MCTS, zoo criteria, FSA, LLM, combiner, backtest)
src/alphamcts/
  data/                   MarketPanel, synthetic generator, CSV loader
  expression/             operators (paper Table 2), formula JSON schema/eval, FSA subtree mining
  evaluation/             metrics (IC/RankIC/RankIR/turnover/corr) + multi-dimensional evaluator
  zoo.py                  effective alpha repository
  llm/                    prompts (paper Appendix K), MockLLM, Cursor SDK client
  mcts/                   Algorithm 1: UCT + virtual expansion, dynamic budget, backprop
  model/combiner.py       LightGBM/MLP combination + top-k/drop-n backtest
  cli.py                  mine / report / evaluate
tests/                    unit tests per module
```

## Tests

```bash
pytest
```
# alphamcts
