"""Retroactively re-screen a mined zoo with the strict anti-overfitting gates.

Replays admission (in original discovery order) against the config's zoo criteria,
recomputing the robustness statistics on the training period only.

By default the online relative-rank gate (max_rel_rank) is disabled in the replay:
it is a *relative* filter meant for the live mining loop, and replaying it against a
smaller, higher-quality surviving zoo systematically rejects factors that were fine
online. Pass --keep-rel-rank to keep it.

Usage:
    python scripts/rescreen_zoo.py --config configs/hs300.yaml \
        --zoo runs/hs300/zoo.json --out runs/hs300_strict [--keep-rel-rank]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphamcts.config import load_config  # noqa: E402
from alphamcts.data.base import FIELDS, load_panel  # noqa: E402
from alphamcts.evaluation.evaluator import MultiDimEvaluator  # noqa: E402
from alphamcts.zoo import AlphaZoo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hs300.yaml")
    ap.add_argument("--zoo", default="runs/hs300/zoo.json")
    ap.add_argument("--out", default="runs/hs300_strict")
    ap.add_argument("--keep-rel-rank", action="store_true",
                    help="also apply the online relative-rank gate during the replay")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg.get("data", {})
    horizon = int(data_cfg.get("horizon", 10))
    wr = tuple(cfg.get("window_range", [2, 60]))
    periods = {k: tuple(v) for k, v in data_cfg.get("periods", {}).items()}

    panel = load_panel(data_cfg)
    old = AlphaZoo.load(ROOT / args.zoo, FIELDS, wr)

    zoo_cfg = dict(cfg.get("zoo", {}))
    if not args.keep_rel_rank:
        zoo_cfg["max_rel_rank"] = 1.01  # a fraction can never exceed 1.0 -> gate off
    strict = AlphaZoo(zoo_cfg)
    evaluator = MultiDimEvaluator(panel, horizon, strict, train_range=periods["train"])

    kept, dropped = [], []
    for rec in old.records:  # original discovery order = online admission order
        rec.values = evaluator.compute_values(rec.formula, rec.args)
        evaluator.attach_robustness(rec.metrics, rec.values)
        ok, reasons = strict.add(rec)
        m = rec.metrics
        line = (f"{rec.expression[:60]:<60} hac_p={m.hac_p:.4f} blk={m.block_pos_frac:.2f} "
                f"yr={m.year_pos_frac:.2f} ctrl={m.control_ic:.4f} r2={m.recon_r2:.3f}")
        (kept if ok else dropped).append(line if ok else line + "  | " + "; ".join(reasons))

    print(f"kept {len(kept)} / {len(old.records)}")
    print("\n--- KEPT ---")
    print("\n".join(kept))
    print("\n--- DROPPED ---")
    print("\n".join(dropped))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    strict.save(out / "zoo.json")
    print(f"\nsaved {len(strict)} alphas -> {out / 'zoo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
