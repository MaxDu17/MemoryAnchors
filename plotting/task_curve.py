#!/usr/bin/env python3
"""Success curve of one task over a whole continual-learning run (paper Fig. 3 style).

    python plotting/task_curve.py results/RandER_1pct/LiberoGoal/<run>/generated_visuals/task_to_series.json \
        --task put_the_cream_cheese_in_the_bowl -o figures/cream_cheese.png --x-values 0,1,2,3,4,5,6,7,8,9

`--x-values` labels evenly spaced ticks (e.g. the task-order indices of the run); each label marks the
start of a training stage, so the line extends one stage past the last label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

LINE_COLOR = "#BC4B51"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 24


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", type=Path, help="generated_visuals/task_to_series.json of a run")
    p.add_argument("--task", required=True, help="BDDL task stem (a key of the json), e.g. put_the_cream_cheese_in_the_bowl")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output .png (a .pdf is written next to it).")
    p.add_argument("--x-values", default=None, help="Comma-separated tick labels spread over the run.")
    p.add_argument("--verbose", action="store_true", help="Axis labels and title.")
    args = p.parse_args()

    data = json.loads(args.json_path.read_text())
    assert args.task in data, f"{args.task!r} not in {args.json_path}; keys: {sorted(data)}"
    y = np.asarray(data[args.task], dtype=float)
    x = np.arange(y.size, dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(x, y, color=LINE_COLOR, linewidth=2.025, solid_capstyle="round", zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pad = 0.02 * (x[-1] - x[0]) if x.size > 1 else 0.05
    ax.set_xlim(x[0] - pad, x[-1] + pad)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    if args.x_values:
        labels = [v.strip() for v in args.x_values.split(",") if v.strip()]
        step = (x[-1] - x[0]) / len(labels) if x.size > 1 else 0.0
        ax.set_xticks(np.arange(len(labels)) * step, labels)
    ax.tick_params(axis="x", length=7.0, width=1.1, color="0.35")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.45, color="0.72", alpha=0.9)
    if args.verbose:
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Success rate")
        ax.set_title(args.task.replace("_", " "))
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), transparent=True, bbox_inches="tight")
    print(f"Wrote {png} (+ .pdf)")


if __name__ == "__main__":
    main()
