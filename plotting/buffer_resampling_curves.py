#!/usr/bin/env python3
"""Paper Fig. 2B: one task's success while fine-tuning the same checkpoint with different ER samples.

Each input is the task_to_series.json of one buffer-resampling run (experiments/buffer_resampling.sh,
shipped in results/RandER_resampling). Checkpoint k of a run with `--epochs` epochs and n evaluations
is drawn at epoch k * epochs / (n - 1); only the first `--fraction` of training is shown.

    python plotting/buffer_resampling_curves.py results/RandER_resampling/LiberoGoal/*/generated_visuals/task_to_series.json \
        --task put_the_cream_cheese_in_the_bowl -o figures/buffer_resampling.png
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

LINE_COLOR = "#5B8E7D"
LAST_COLOR = "#BC4B51"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["ytick.labelsize"] = 12


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_paths", type=Path, nargs="+", help="task_to_series.json of each run (one line per run)")
    p.add_argument("--task", default="put_the_cream_cheese_in_the_bowl", help="Task stem to plot (paper: Cream Cheese to Bowl).")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output .png (a .pdf is written next to it).")
    p.add_argument("--epochs", type=float, default=50.0, help="Training epochs of the fine-tuned task.")
    p.add_argument("--fraction", type=float, default=0.75, help="Fraction of training shown (paper: 0.75).")
    p.add_argument("--ymin", type=float, default=0.2)
    p.add_argument("--labels", nargs="+", default=None, help="Legend labels (one per json).")
    p.add_argument("--last-colored", action="store_true", help="Draw the last run in a distinct color.")
    args = p.parse_args()
    assert args.labels is None or len(args.labels) == len(args.json_paths)

    fig, ax = plt.subplots(figsize=(7.26, 2.4))
    x_max = args.epochs * args.fraction
    for i, path in enumerate(args.json_paths):
        data = json.loads(path.read_text())
        assert args.task in data, f"{args.task!r} not in {path}"
        y = np.asarray(data[args.task], dtype=float)
        x = np.arange(y.size) * (args.epochs / max(1, y.size - 1))
        keep = x <= x_max + 1e-9
        color = LAST_COLOR if (args.last_colored and i == len(args.json_paths) - 1) else LINE_COLOR
        ax.plot(x[keep], y[keep], color=color, linewidth=1.35, marker=MARKERS[i % len(MARKERS)], markersize=5.0,
                label=args.labels[i] if args.labels else None, clip_on=False, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(args.ymin, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.45, color="0.72", alpha=0.9)
    ax.set_xlabel("Training epochs (single task)")
    ax.set_ylabel("Success rate")
    if args.labels:
        ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), transparent=True)
    print(f"Wrote {png} (+ .pdf)")


if __name__ == "__main__":
    main()
