#!/usr/bin/env python3
"""Violin plots of per-task FWT / NBT / AUC distributions per method and suite (paper Fig. 2A, Fig. 14).

Every per-task value of every complete run is pooled per (method, suite); "All" pools the suites.
Also prints the variance and the worst case of the pooled NBT values (paper Sec. 3.1: RandER has
lower variance and lower worst-case forgetting than FixedER).

    python plotting/fixed_vs_rand_violins.py --experiment-dirs results/FixedER_1000 results/RandER_1000 \
        --labels FixedER RandER --output-dir figures/fixed_vs_rand

Writes {fwt,bwt,auc}_lines.{pdf,png} and bwt_lines_deltas.{pdf,png} (all task-to-task deltas).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics import keep_run, parse_orders  # noqa: E402

METRICS = ["fwt", "bwt", "auc"]
DEFAULT_SUITES = ["Libero10", "LiberoGoal", "LiberoObject", "LiberoSpatial"]
ALL = "All"
XTICKS = {"Libero10": "Long", "LiberoGoal": "Goal", "LiberoObject": "Object", "LiberoSpatial": "Spatial", ALL: ALL}
COLORS = ["#BC4B51", "#5B8E7D", "#8C6D67", "#A8986B", "#C0CB77", "#F4E285", "#F4A259"]  # FixedER red, RandER green
OUTLINE = "#666666"
LINES_YMIN = {"fwt": 0.4, "auc": 0.2}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["ytick.labelsize"] = 12


def complete_runs(suite_dir: Path, orders) -> List[Path]:
    if not suite_dir.is_dir():
        return []
    out = []
    for r in sorted(suite_dir.iterdir()):
        if r.is_dir() and keep_run(r, orders):
            if (r / "generated_visuals" / "complete.txt").is_file():
                out.append(r)
            elif (r / "generated_visuals" / "continual_metrics.json").is_file():
                print(f"[WARN] skipping incomplete run {r}")
    return out


def pooled_values(exp: Path, suites: List[str], orders) -> Dict[str, Dict[str, np.ndarray]]:
    """{metric or 'deltas': {suite or All: pooled values}}."""
    out = {k: {} for k in METRICS + ["deltas"]}
    for suite in suites:
        payloads = [json.loads((r / "generated_visuals" / "continual_metrics.json").read_text())
                    for r in complete_runs(exp / suite, orders)]
        if not payloads:
            print(f"[WARN] no complete runs in {exp / suite}")
        for m in METRICS:
            out[m][suite] = np.asarray([v for p in payloads for v in p[f"{m}_per_task"]], dtype=np.float64)
        out["deltas"][suite] = np.asarray([d for p in payloads for row in p["bwt_task_to_task_deltas"] for d in row],
                                          dtype=np.float64)
    for k in out:
        parts = [out[k][s] for s in suites if out[k][s].size]
        out[k][ALL] = np.concatenate(parts) if parts else np.asarray([])
    return out


def violin_plot(key: str, x_labels, labels, data, out_dir: Path, stem: str, legend: bool) -> None:
    n = len(labels)
    width = 0.88 / n
    xs = np.arange(len(x_labels), dtype=np.float64)
    xs[-1] += 0.25
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    for j, suite in enumerate(x_labels):
        arrays, positions, colors = [], [], []
        for i, lab in enumerate(labels):
            arr = data[lab][key][suite]
            x = float(xs[j] + (i - (n - 1) / 2.0) * width)
            if arr.size == 1:
                ax.scatter(x, arr[0], s=18, facecolors="none", edgecolors=COLORS[i], linewidths=0.9, zorder=4)
            elif arr.size > 1:
                arrays.append(arr)
                positions.append(x)
                colors.append(COLORS[i])
        if arrays:
            parts = ax.violinplot(arrays, positions=positions, widths=width * 0.82, showmedians=True, showextrema=True)
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c)
                body.set_edgecolor(OUTLINE)
                body.set_linewidth(0.6)
                body.set_alpha(0.88)
            for k in ("cbars", "cmins", "cmaxes", "cmedians"):
                if k in parts:
                    parts[k].set_color(OUTLINE)
                    parts[k].set_linewidth(0.65)
    ax.set_xticks(xs, [XTICKS.get(s, s) for s in x_labels], fontname="DejaVu Sans")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xs[0] - 0.5 * n * width - 0.1, xs[-1] + 0.5 * n * width + 0.1)
    if key in LINES_YMIN:
        ax.set_ylim(LINES_YMIN[key], 1.0)
    else:
        all_y = np.concatenate([data[lab][key][s] for lab in labels for s in x_labels if data[lab][key][s].size] or [np.zeros(1)])
        ax.set_ylim(-0.2, max(0.05, float(np.max(all_y)) * 1.12))
        ax.axhline(0.0, color="black", linewidth=0.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.45, color="0.72", alpha=0.9)
    if legend:
        ax.legend(handles=[Patch(facecolor=COLORS[i], edgecolor="black", label=lab, alpha=0.88) for i, lab in enumerate(labels)],
                  frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", transparent=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    p.add_argument("--orders", default=None, help="Only runs with these task-order indices, e.g. 0,1,2.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--legend", action="store_true")
    args = p.parse_args()
    labels = args.labels or [e.name for e in args.experiment_dirs]
    assert len(labels) == len(args.experiment_dirs) and len(set(labels)) == len(labels)
    assert len(labels) <= len(COLORS), f"at most {len(COLORS)} methods"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders = parse_orders(args.orders)
    x_labels = list(args.suites) + [ALL]

    data = {lab: pooled_values(e.resolve(), list(args.suites), orders) for e, lab in zip(args.experiment_dirs, labels)}
    for m in METRICS:
        violin_plot(m, x_labels, labels, data, args.output_dir, f"{m}_lines", args.legend)
    violin_plot("deltas", x_labels, labels, data, args.output_dir, "bwt_lines_deltas", args.legend)

    print("Pooled per-task NBT: mean / variance / worst (max)")
    for lab in labels:
        for s in x_labels:
            a = data[lab]["bwt"][s]
            if a.size:
                print(f"  {lab:12s} {XTICKS.get(s, s):8s} n={a.size:3d} mean={a.mean():.3f} var={a.var():.4f} worst={a.max():.3f}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
