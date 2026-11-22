#!/usr/bin/env python3
"""Compare methods across LIBERO suites: grouped bar charts (mean ± SEM) + JSON of every number.

Input: one experiment folder per method, laid out as <exp>/<SuiteFolder>/<run_dir>/generated_visuals/
(run plotting/compute_metrics.py first; only runs with generated_visuals/complete.txt are used).

Writes <output-dir>/<prefix><metric>_bar[<variant>].{json,pdf,png} for
  fwt / bwt / auc            mean over runs (or, with --pool-per-task, over every per-task value)
  *_worst                    per run: min fwt/auc and max bwt over tasks
  bwt_taskwise_worst         per run: mean of the 10 largest task-to-task NBT deltas
  bwt_targettasks            "NBT High Conflict" (paper Fig. 6 right / App. B.4): moving-anchor
                             Delta_{i,j} on the reference suite's top-6 conflicting pairs, pooled over
                             runs x pairs. Needs <exp>/<Suite>/generated_visuals/top6_bt_pair_delta_extraction.json
                             from plotting/high_conflict/extract_pair_deltas.py.
  mean_final_task_success    per run: mean over tasks of the last success value
"bwt" is the paper's NBT (higher = more forgetting). SEM = sample std (ddof=1) / sqrt(n).

Paper figures (paths relative to the repo root):
  Fig. 6 left (removal):   python plotting/compare_methods.py --pool-per-task --palette remove --prefix removeanchors \\
      --experiment-dirs results/RandER_1000 results/RandER-1pct_1000 results/RandER-5pct_1000 results/RandER-10pct_1000 \\
      --labels RandER "RandER -1%" "RandER -5%" "RandER -10%" --output-dir figures/remove_anchors
  Fig. 6 right (addition): python plotting/compare_methods.py --pool-per-task --palette add --prefix addanchors \\
      --experiment-dirs results/RandER_1pct results/AnchorER-10_1pct results/AnchorER-20_1pct \\
      --labels RandER "AnchorER 10%" "AnchorER 20%" --output-dir figures/add_anchors
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "high_conflict"))
from target_task_pair_metrics import TARGET_TASK_EXTRACTION_FILENAME  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics import keep_run, parse_orders  # noqa: E402

ORDERS = None  # set from --orders in main()

METRICS = ["fwt", "bwt", "auc"]
DEFAULT_SUITES = ["Libero10", "LiberoGoal", "LiberoObject", "LiberoSpatial"]
AVERAGE_LABEL = "Average"
TASKWISE_WORST_TOP_N = 10
SUITE_XTICK_LABELS = {"Libero10": "Long", "LiberoGoal": "Goal", "LiberoObject": "Object", "LiberoSpatial": "Spatial"}
PALETTES = {  # paper palettes (index i = i-th experiment dir)
    "add": ["#5B8E7D", "#facfa7", "#F4A259", "#a46e3a", "#C0CB77", "#F4E285"],
    "remove": ["#5B8E7D", "#9ab3ab", "#848e8b", "#525354", "#b49e99", "#91716b", "#2c1d1a"],
}
ERROR_KW = {"elinewidth": 1, "capthick": 1, "ecolor": "black"}
BAR_VALUE_LABEL_FONTSIZE = 9 * 0.9

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["ytick.labelsize"] = 12

Stats = Tuple[float, float]  # (mean, sem)


def sem(values) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size >= 2 else 0.0


def mean_sem(values) -> Stats:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return float(np.mean(arr)), sem(arr)


def xtick(suite: str) -> str:
    return SUITE_XTICK_LABELS.get(suite, suite)


# ----------------------------------------------------------------------------- loading
def discover_run_dirs(suite_dir: Path) -> List[Path]:
    """Runs with generated_visuals/continual_metrics.json and the complete.txt sentinel."""
    if not suite_dir.is_dir():
        return []
    out = []
    for run in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        if not keep_run(run, ORDERS) or run.name == "generated_visuals":
            continue
        gv = run / "generated_visuals"
        if not (gv / "continual_metrics.json").is_file():
            continue
        if not (gv / "complete.txt").is_file():
            warnings.warn(f"Skipping incomplete run (no complete.txt): {run}")
            continue
        out.append(run)
    return out


def load_metrics(run: Path) -> Dict:
    return json.loads((run / "generated_visuals" / "continual_metrics.json").read_text())


def run_scalars(run: Path) -> Dict[str, float]:
    m = load_metrics(run)
    return {k: float(m[k]) for k in METRICS}


def run_per_task(run: Path) -> Dict[str, np.ndarray]:
    m = load_metrics(run)
    return {k: np.asarray(m[f"{k}_per_task"], dtype=np.float64) for k in METRICS}


def run_worst(run: Path) -> Dict[str, float]:
    m = load_metrics(run)
    return {"fwt": float(np.min(m["fwt_per_task"])), "bwt": float(np.max(m["bwt_per_task"])),
            "auc": float(np.min(m["auc_per_task"]))}


def run_taskwise_worst(run: Path) -> float:
    flat = np.sort([d for row in load_metrics(run)["bwt_task_to_task_deltas"] for d in row])
    if flat.size < TASKWISE_WORST_TOP_N:
        warnings.warn(f"Only {flat.size} task-to-task deltas in {run}; using all of them.")
        return float(np.mean(flat))
    return float(np.mean(flat[-TASKWISE_WORST_TOP_N:]))


def run_mean_final_success(run: Path) -> float:
    series = json.loads((run / "generated_visuals" / "task_to_series.json").read_text())
    return float(np.mean([s[-1] for s in series.values()]))


def suite_targettasks(suite_dir: Path) -> Optional[np.ndarray]:
    path = suite_dir / "generated_visuals" / TARGET_TASK_EXTRACTION_FILENAME
    if not path.is_file():
        warnings.warn(f"Missing {path} (run plotting/high_conflict/extract_pair_deltas.py); skipping targettasks.")
        return None
    runs = [r for r in json.loads(path.read_text())["target"]["runs"] if keep_run(Path(r["run_name"]), ORDERS)]
    vals = [float(p["bwt_task_to_task_moving_anchor"]) for r in runs for p in r["extracted_pairs"]]
    return np.asarray(vals, dtype=np.float64) if vals else None


# ----------------------------------------------------------------------------- aggregation
def aggregate(exp_dirs: List[Path], labels: List[str], suites: List[str], pool_per_task: bool):
    """Returns {variant: {label: {metric: {suite|Average: (mean, sem)}}}} and per-label final success."""
    runs = {lab: {s: discover_run_dirs(e / s) for s in suites} for e, lab in zip(exp_dirs, labels)}
    out: Dict[str, Dict] = {"": {}, "_worst": {}, "_taskwise_worst": {}, "_targettasks": {}}
    final_success: Dict[str, Stats] = {}
    for exp, lab in zip(exp_dirs, labels):
        R = runs[lab]
        # vanilla bars
        vanilla = {m: {} for m in METRICS}
        if pool_per_task:
            pooled = {m: [] for m in METRICS}
            for s in suites:
                per = [run_per_task(r) for r in R[s]]
                for m in METRICS:
                    arr = np.concatenate([p[m] for p in per])
                    vanilla[m][s] = mean_sem(arr)
                    pooled[m].append(arr)
            for m in METRICS:  # Average pools every per-task value across suites
                vanilla[m][AVERAGE_LABEL] = mean_sem(np.concatenate(pooled[m]))
        else:
            for s in suites:
                sc = [run_scalars(r) for r in R[s]]
                for m in METRICS:
                    vanilla[m][s] = mean_sem([x[m] for x in sc])
            for m in METRICS:  # Average = mean of suite means, SEM across suites
                vanilla[m][AVERAGE_LABEL] = mean_sem([vanilla[m][s][0] for s in suites])
        out[""][lab] = vanilla
        # worst-case per run
        worst = {m: {} for m in METRICS}
        for s in suites:
            w = [run_worst(r) for r in R[s]]
            for m in METRICS:
                worst[m][s] = mean_sem([x[m] for x in w])
        for m in METRICS:
            worst[m][AVERAGE_LABEL] = mean_sem([worst[m][s][0] for s in suites])
        out["_worst"][lab] = worst
        # taskwise worst
        tw = {s: mean_sem([run_taskwise_worst(r) for r in R[s]]) for s in suites}
        tw[AVERAGE_LABEL] = mean_sem([tw[s][0] for s in suites])
        out["_taskwise_worst"][lab] = {"bwt": tw}
        # NBT high conflict (targettasks)
        tt = {s: suite_targettasks(exp / s) for s in suites}
        if all(v is not None for v in tt.values()):
            tts = {s: mean_sem(v) for s, v in tt.items()}
            tts[AVERAGE_LABEL] = mean_sem([tts[s][0] for s in suites])
            out["_targettasks"][lab] = {"bwt": tts}
        final_success[lab] = mean_sem([run_mean_final_success(r) for s in suites for r in R[s]])
    if len(out["_targettasks"]) != len(labels):
        out.pop("_targettasks")
    return out, final_success


def present_suites(exp_dirs: List[Path], suites: List[str]) -> List[str]:
    """Drop a suite (with a warning) if any method has no complete run in it."""
    keep = []
    for s in suites:
        missing = [str(e) for e in exp_dirs if not discover_run_dirs(e / s)]
        if missing:
            warnings.warn(f"Suite {s} skipped: no complete runs in {missing}")
        else:
            keep.append(s)
    assert keep, "No suite has complete runs for every method."
    return keep


# ----------------------------------------------------------------------------- plotting
def plot_bars(metric, variant, x_labels, labels, agg, colors, out_dir: Path, prefix: str,
              fig_width: float, ylim: Optional[Tuple[float, float]], legend: bool) -> None:
    n = len(labels)
    width = 0.88 / n
    xs = np.arange(len(x_labels), dtype=np.float64)
    xs[-1] += 0.25  # gap before "Average"
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))
    tops, lows = [], []
    for i, lab in enumerate(labels):
        means = [agg[lab][metric][s][0] for s in x_labels]
        sems = [agg[lab][metric][s][1] for s in x_labels]
        bars = ax.bar(xs + (i - (n - 1) / 2) * width, means, width, yerr=sems, capsize=2,
                      error_kw=ERROR_KW, color=colors[i], label=lab)
        for b, m, e in zip(bars, means, sems):
            ax.annotate(f"{m:.2f}", xy=(b.get_x() + b.get_width() / 2, m + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=BAR_VALUE_LABEL_FONTSIZE, fontname="serif")
        tops += [m + e for m, e in zip(means, sems)]
        lows += [m - e for m, e in zip(means, sems)]
    ax.set_xticks(xs, [xtick(s) for s in x_labels], fontname="DejaVu Sans")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xs[0] - 0.5 * n * width - 0.1, xs[-1] + 0.5 * n * width + 0.1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif metric in ("fwt", "auc"):
        ax.set_ylim(0.0, min(1.0, max(0.05, 1.15 * max(tops))) if variant == "" else 1.0)
    else:
        lo = min(0.0, 1.15 * min(lows)) if lows else 0.0
        ax.set_ylim(lo, max(0.05, 1.15 * max(tops)))
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2 if metric != "bwt" else 0.1))
    ax.tick_params(axis="y", length=3.5, width=0.45, color="0.55")
    if legend:
        ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    stem = f"{prefix}{metric}_bar{variant}"
    payload = {"metric": metric, "variant": variant, "x_labels": x_labels,
               "methods": [{"label": lab, "by_suite": {s: {"mean": agg[lab][metric][s][0], "sem": agg[lab][metric][s][1]}
                                                       for s in x_labels}} for lab in labels]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig.savefig(out_dir / f"{stem}.pdf", transparent=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dirs", nargs="+", type=Path, required=True, help="One folder per method.")
    p.add_argument("--labels", nargs="+", default=None, help="Legend labels (default: folder names).")
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES, help="Suite folders to include.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", default="compare_", help="Filename prefix of all outputs.")
    p.add_argument("--pool-per-task", action="store_true",
                   help="Vanilla bars pool every per-task value across runs (paper Fig. 6 / Tables 3-4); "
                        "default: one scalar per run.")
    p.add_argument("--palette", default="add",
                   help="'add', 'remove' (paper palettes) or comma-separated hex colors.")
    p.add_argument("--fig-width", type=float, default=6.12, help="Figure width in inches (paper: 6.12 add, 7.2 remove).")
    p.add_argument("--bwt-ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="Fix the NBT axis of the vanilla bwt plot (paper used 0 0.3). Default: auto.")
    p.add_argument("--orders", default=None,
                   help="Only use runs with these task-order indices, e.g. 0,1,2 (default: all complete runs).")
    p.add_argument("--legend", action="store_true")
    return p.parse_args()


def main() -> None:
    global ORDERS
    args = parse_args()
    ORDERS = parse_orders(args.orders)
    exp_dirs = [p.resolve() for p in args.experiment_dirs]
    labels = args.labels or [p.name for p in exp_dirs]
    assert len(labels) == len(exp_dirs), "--labels must match --experiment-dirs"
    colors = PALETTES.get(args.palette, args.palette.split(","))
    if len(colors) < len(labels):
        colors = list(colors) + [matplotlib.colormaps["tab10"](i) for i in range(len(labels) - len(colors))]
    suites = present_suites(exp_dirs, list(args.suites))
    x_labels = suites + [AVERAGE_LABEL]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agg, final_success = aggregate(exp_dirs, labels, suites, args.pool_per_task)
    for variant, per_label in agg.items():
        metrics = METRICS if variant in ("", "_worst") else ["bwt"]
        for m in metrics:
            ylim = tuple(args.bwt_ylim) if (variant == "" and m == "bwt" and args.bwt_ylim) else None
            plot_bars(m, variant, x_labels, labels, per_label, colors, args.output_dir, args.prefix,
                      args.fig_width, ylim, args.legend)
    (args.output_dir / f"{args.prefix}mean_final_task_success.json").write_text(json.dumps(
        {"methods": [{"label": lab, "mean": v[0], "sem": v[1]} for lab, v in final_success.items()]}, indent=2) + "\n")

    # console summary
    print(f"{'method':28s} " + " ".join(f"{xtick(s):>18s}" for s in x_labels))
    for variant, per_label in agg.items():
        for m in (METRICS if variant in ("", "_worst") else ["bwt"]):
            print(f"-- {m}{variant}")
            for lab in labels:
                cells = [f"{per_label[lab][m][s][0]:.3f} ± {per_label[lab][m][s][1]:.3f}" for s in x_labels]
                print(f"   {lab:25s} " + " ".join(f"{c:>18s}" for c in cells))
    print(f"Wrote figures and json to {args.output_dir}")


if __name__ == "__main__":
    main()
