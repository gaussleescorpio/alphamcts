"""Command-line interface: mine / report / evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import load_config
from .data.base import FIELDS, load_panel
from .evaluation.evaluator import MultiDimEvaluator
from .llm import AlphaAgent, Temperatures, make_client
from .mcts import AlphaMiner
from .model import CombinerPipeline
from .zoo import AlphaZoo

logger = logging.getLogger("alphamcts")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, no override of existing variables."""
    import os

    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _setup_logging(out_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "mining.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_panel(cfg: dict):
    data_cfg = dict(cfg.get("data", {}))
    data_cfg.setdefault("seed", cfg.get("seed", 0))
    return load_panel(data_cfg), data_cfg


def _window_range(cfg: dict) -> tuple[int, int]:
    lo, hi = cfg.get("window_range", [2, 60])
    return int(lo), int(hi)


def _periods(cfg: dict) -> dict[str, tuple[str, str]] | None:
    """Optional explicit period split: data.periods.{train,valid,test} = [start, end]."""
    raw = cfg.get("data", {}).get("periods")
    if not raw:
        return None
    return {name: (str(rng[0]), str(rng[1])) for name, rng in raw.items()}


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

def cmd_mine(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.llm:
        cfg.setdefault("llm", {})["backend"] = args.llm
    out_dir = Path(args.out)
    _setup_logging(out_dir)

    panel, data_cfg = _build_panel(cfg)
    logger.info("Panel: %d stocks x %d days (%s)", panel.n_stocks, panel.n_days, data_cfg["source"])

    zoo = AlphaZoo(cfg.get("zoo", {}))
    periods = _periods(cfg)
    evaluator = MultiDimEvaluator(
        panel,
        horizon=int(data_cfg.get("horizon", 10)),
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        zoo=zoo,
        top_frac=float(cfg.get("evaluation", {}).get("top_frac", 0.1)),
        train_range=periods.get("train") if periods else None,
        valid_range=periods.get("valid") if periods else None,
    )
    if periods:
        logger.info("Mining period: %s .. %s (%d trading days)",
                    periods["train"][0], periods["train"][1], len(evaluator.train_dates))
    llm_cfg = dict(cfg.get("llm", {}))
    llm_cfg.setdefault("seed", cfg.get("seed", 0))
    client = make_client(llm_cfg)
    temps = Temperatures(
        generate=float(llm_cfg.get("temperature_generate", 1.0)),
        correct=float(llm_cfg.get("temperature_correct", 0.8)),
        overfit=float(llm_cfg.get("temperature_overfit", 0.1)),
    )
    agent = AlphaAgent(
        client,
        FIELDS,
        _window_range(cfg),
        temperatures=temps,
        max_correction_attempts=int(cfg.get("mcts", {}).get("max_correction_attempts", 3)),
        overfit_scorer=str(llm_cfg.get("overfit_scorer", "llm")),
    )
    zoo_path = out_dir / "zoo.json"
    if getattr(args, "resume", False) and zoo_path.exists():
        loaded = AlphaZoo.load(zoo_path, FIELDS, _window_range(cfg))
        zoo.records = loaded.records
        # recompute training-period values for correlation / exemplar sampling
        for record in zoo.records:
            record.values = evaluator.compute_values(record.formula, record.args)
        logger.info("Resumed with %d alphas from %s", len(zoo), zoo_path)

    miner = AlphaMiner(
        agent,
        evaluator,
        zoo,
        mcts_cfg=cfg.get("mcts", {}),
        mining_cfg=cfg.get("mining", {}),
        fsa_cfg=cfg.get("fsa", {}),
        llm_cfg=llm_cfg,
        seed=int(cfg.get("seed", 0)),
    )

    def checkpoint(stats) -> None:
        zoo.save(zoo_path)
        (out_dir / "stats.json").write_text(
            json.dumps(
                {
                    "generated": stats.generated,
                    "invalid": stats.invalid,
                    "added_to_zoo": stats.added_to_zoo,
                    "trees": stats.trees,
                    "breakthroughs": stats.breakthroughs,
                    "zoo_size": len(zoo),
                    "forbidden_subtrees": miner.forbidden,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    shutil.copy(args.config, out_dir / "config.yaml")
    stats = miner.run(checkpoint=checkpoint)
    checkpoint(stats)
    logger.info(
        "Mining done: %d candidates generated (%d invalid), %d alphas in the zoo -> %s",
        stats.generated, stats.invalid, len(zoo), out_dir / "zoo.json",
    )
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    zoo = AlphaZoo.load(run_dir / "zoo.json", FIELDS, _window_range(cfg))
    print(f"Alpha zoo: {len(zoo)} effective alphas ({run_dir / 'zoo.json'})\n")
    header = f"{'#':>3}  {'RankIC':>8}  {'RankIR':>7}  {'Turn':>6}  {'MaxCorr':>7}  Expression"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(zoo.top_by_rank_ir(len(zoo))):
        m = r.metrics
        print(f"{i + 1:>3}  {m.rank_ic:>8.4f}  {m.rank_ir:>7.2f}  {m.turnover:>6.2f}  "
              f"{m.max_corr:>7.2f}  {r.expression}")
    stats_path = run_dir / "stats.json"
    if stats_path.exists():
        print("\nSearch stats:", stats_path.read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    cfg = load_config(args.config or run_dir / "config.yaml")
    _setup_logging()

    panel, data_cfg = _build_panel(cfg)
    zoo = AlphaZoo.load(run_dir / "zoo.json", FIELDS, _window_range(cfg))
    if len(zoo) == 0:
        print("Zoo is empty; nothing to evaluate.")
        return 1

    horizon = int(data_cfg.get("horizon", 10))
    periods = _periods(cfg)
    combiner_cfg = cfg.get("combiner", {})

    payload: dict = {"per_alpha": [], "combiner": []}

    if periods:
        payload["per_alpha"] = _per_alpha_period_report(panel, zoo, horizon, periods)
        eval_periods = {k: v for k, v in periods.items() if k != "train"}
        pipe = CombinerPipeline(
            panel,
            horizon=horizon,
            combiner_cfg=combiner_cfg,
            backtest_cfg=cfg.get("backtest", {}),
            train_range=periods["train"],
            eval_periods=eval_periods,
        )
    else:
        eval_periods = None
        pipe = CombinerPipeline(
            panel,
            horizon=horizon,
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            combiner_cfg=combiner_cfg,
            backtest_cfg=cfg.get("backtest", {}),
        )

    sizes = combiner_cfg.get("alpha_set_sizes", [10])
    models = combiner_cfg.get("models", ["lightgbm"])
    for size in sizes:
        if isinstance(size, str) and size.lower() == "all":
            size = len(zoo)
        records = zoo.top_by_rank_ir(int(size))
        if not records:
            continue
        for model_name in models:
            logger.info("Combining top-%d alphas with %s (actual %d)...", size, model_name, len(records))
            model, features = pipe.fit(model_name, records)
            for period_name, dates in pipe.eval_periods.items():
                report = pipe.evaluate_period(model, features, dates, model_name, len(records))
                payload["combiner"].append(dict(report.to_dict(), period=period_name, set_size=size))
                print(
                    f"[{model_name} | top-{size} -> {len(records)} alphas | {period_name}]  "
                    f"IC={report.ic:.4f}  RankIC={report.rank_ic:.4f}  "
                    f"AER={report.aer:.4f}  IR={report.ir:.4f}"
                )
            if hasattr(model, "weights"):  # dynamic weighting: per-period decay summary
                for period_name, dates in pipe.eval_periods.items():
                    w = model.weights.loc[dates]
                    mean_w = w.mean().round(3).tolist()
                    paused = (w <= 1e-12).mean().round(2).tolist()
                    payload.setdefault("ewma_weights", {})[period_name] = {
                        "mean_weight": mean_w, "paused_frac": paused,
                        "expressions": [r.expression for r in records],
                    }
                    print(f"[{model_name} | {period_name}] mean weights: {mean_w}")
                    print(f"[{model_name} | {period_name}] paused frac:  {paused}")

    out_path = run_dir / "evaluation.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to {out_path}")
    return 0


def _per_alpha_period_report(panel, zoo, horizon: int, periods: dict[str, tuple[str, str]]) -> list[dict]:
    """Per-alpha RankIC/RankIR per period + sign-flip flags vs the training period."""
    from alphamcts.evaluation.metrics import mean_and_ir, rank_ic_series

    fwd = panel.apply_universe(panel.forward_returns(horizon))
    rows: list[dict] = []
    for record in zoo:
        values = panel.apply_universe(
            panel.maybe_neutralize_factor(record.formula.evaluate(panel.fields, record.args))
        )
        row: dict = {"expression": record.expression, "name": record.name}
        for period_name, rng in periods.items():
            dates = panel.dates_between(*rng)
            ric, rir = mean_and_ir(rank_ic_series(values.loc[dates], fwd.loc[dates]))
            row[f"rank_ic_{period_name}"] = ric
            row[f"rank_ir_{period_name}"] = rir
        train_ic = row.get("rank_ic_train", float("nan"))
        for period_name in periods:
            if period_name == "train":
                continue
            ic = row[f"rank_ic_{period_name}"]
            row[f"sign_flip_{period_name}"] = bool(ic == ic and train_ic == train_ic and ic * train_ic < 0)
        rows.append(row)

    # console table
    period_names = list(periods)
    header = "  ".join(f"RankIC[{p}]" for p in period_names) + "  flips"
    print(f"\nPer-alpha period report ({len(rows)} alphas):")
    print(f"{'#':>3}  {header}  expression")
    for i, row in enumerate(sorted(rows, key=lambda r: -r.get("rank_ic_train", 0.0))):
        ics = "  ".join(f"{row.get(f'rank_ic_{p}', float('nan')):>12.4f}" for p in period_names)
        flips = ",".join(p for p in period_names if row.get(f"sign_flip_{p}"))
        expr = row["expression"]
        print(f"{i + 1:>3}  {ics}  {flips or '-':>6}  {expr[:80]}")
    return rows


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="alphamcts", description="LLM-powered MCTS alpha mining")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mine = sub.add_parser("mine", help="run the MCTS alpha mining loop")
    p_mine.add_argument("--config", default="configs/default.yaml")
    p_mine.add_argument("--out", default="runs/latest")
    p_mine.add_argument("--llm", choices=["mock", "cursor"], default=None,
                        help="override llm.backend from the config")
    p_mine.add_argument("--resume", action="store_true",
                        help="load an existing zoo.json from --out and continue mining")
    p_mine.set_defaults(func=cmd_mine)

    p_report = sub.add_parser("report", help="print the mined alpha zoo")
    p_report.add_argument("--run", default="runs/latest")
    p_report.set_defaults(func=cmd_report)

    p_eval = sub.add_parser("evaluate", help="train combination models and backtest")
    p_eval.add_argument("--run", default="runs/latest")
    p_eval.add_argument("--config", default=None, help="override the run's saved config")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
