#!/usr/bin/env python3
"""Print FWT / NBT / AUC (mean ± SEM) per method and suite as a Markdown table (paper Tables 3-4).

Reads <exp>/<Suite>/<run>/generated_visuals/continual_metrics.json (complete runs only; run
plotting/compute_metrics.py first). By default every per-task value of every run is pooled
(10 x n_runs samples for FWT/AUC, 9 x n_runs for NBT), as in the paper; --per-run uses one
scalar per run instead. SEM = sample std (ddof=1) / sqrt(n).

    python plotting/metrics_table.py --experiment-dirs results/RandER_1000 results/RandER-10pct_1000 \
        --labels RandER "RandER -10%"
    python plotting/metrics_table.py --experiment-dirs results/RandER_1pct results/AnchorER-10_1pct \
        results/MIR_1pct results/GSS_1pct results/AnchorER-NoDisagreement_1pct --suites LiberoGoal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics import keep_run, parse_orders  # noqa: E402

METRICS = [("fwt", "FWT (↑)"), ("bwt", "NBT (↓)"), ("auc", "AUC (↑)")]
DEFAULT_SUITES = ["Libero10", "LiberoGoal", "LiberoObject", "LiberoSpatial"]
SUITE_NAMES = {"Libero10": "LIBERO-Long", "LiberoGoal": "LIBERO-Goal", "LiberoObject": "LIBERO-Object",
               "LiberoSpatial": "LIBERO-Spatial"}


def complete_runs(suite_dir: Path, orders):
    if not suite_dir.is_dir():
        return []
    return [r for r in sorted(suite_dir.iterdir())
            if r.is_dir() and keep_run(r, orders) and (r / "generated_visuals" / "complete.txt").is_file()]


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size >= 2 else 0.0
    return float(np.mean(arr)), sem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    p.add_argument("--per-run", action="store_true", help="One scalar per run instead of pooling per-task values.")
    p.add_argument("--orders", default=None, help="Only runs with these task-order indices, e.g. 0,1,2.")
    p.add_argument("--json", type=Path, default=None, help="Also write the numbers to this json file.")
    args = p.parse_args()
    labels = args.labels or [e.name for e in args.experiment_dirs]
    assert len(labels) == len(args.experiment_dirs)
    orders = parse_orders(args.orders)

    table = {}
    for suite in args.suites:
        print(f"\n### {SUITE_NAMES.get(suite, suite)}\n")
        print("| Method | n | " + " | ".join(name for _, name in METRICS) + " |")
        print("|---|---|" + "---|" * len(METRICS))
        for exp, lab in zip(args.experiment_dirs, labels):
            runs = complete_runs(exp / suite, orders)
            if not runs:
                print(f"| {lab} | 0 | " + " | ".join("–" for _ in METRICS) + " |")
                continue
            payloads = [json.loads((r / "generated_visuals" / "continual_metrics.json").read_text()) for r in runs]
            cells = {}
            for key, _ in METRICS:
                vals = [m[key] for m in payloads] if args.per_run else [v for m in payloads for v in m[f"{key}_per_task"]]
                cells[key] = stats(vals)
            table.setdefault(lab, {})[suite] = {k: {"mean": m, "sem": s, "n_runs": len(runs)} for k, (m, s) in cells.items()}
            print(f"| {lab} | {len(runs)} | " + " | ".join(f"{cells[k][0]:.2f} ± {cells[k][1]:.2f}" for k, _ in METRICS) + " |")
    if args.json:
        args.json.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
