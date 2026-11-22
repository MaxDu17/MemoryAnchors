#!/usr/bin/env python3
"""Continual-learning metrics (FWT / NBT / AUC) for one or many training runs.

Reads, for every run directory:
  checkpoints/<global_step>.ckpt.json   eval results written by scripts/evaluate_runs.py
  task_boundaries.json                  (task_name, end_step) per finished task
  .hydra/config.yaml                    run config (used to check that the curriculum finished)
and writes into <run_dir>/generated_visuals/:
  task_to_series.json          success rate of every task at every evaluated checkpoint
  continual_success_matrix.npy S[i, j] = success on task j after training stage i (NaN for j > i)
  continual_metrics.json       fwt / bwt / auc (+ per-task values, task-to-task deltas, precise variants)
  complete.txt                 only if the run finished its full curriculum with an eval at every boundary
  *.png / *.pdf                per-task success curves and the success matrix

Metric definitions (paper App. B.4; c[i][k] = success on task i after training task k):
  fwt  = mean_i c[i][i]
  bwt  = mean_{i<N} mean_{k>i} (c[i][i] - c[i][k])        (= paper NBT; higher = more forgetting)
  auc  = mean_i mean_{k>=i} c[i][k]
  bwt_task_to_task_moving_anchors[i][j] = c[i][j-1] - c[i][j]   (task-task interaction Delta_{i,j})
  *_precise: the same, using every intermediate checkpoint (not used for the paper's numbers).

Usage:
  python plotting/compute_metrics.py results/RandER_1000/LiberoGoal --batch-parent --libero_split libero_goal
  python plotting/compute_metrics.py outputs/my_exp/LiberoGoal/<run_dir> --libero_split libero_goal
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_ROOT = REPO / "memory_anchors" / "task_name_mappings"
MEANS_SCORE_KEY_SUFFIX = "_mean_score"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["ytick.labelsize"] = 12

FIGURE_READY_LINE_COLOR = "#BC4B51"
FIGURE_READY_LINE_WIDTH = 2.025
FIGURE_READY_Y_TOP = 1.03
FIGURE_READY_MARKER_SIZE = 4.0


# ----------------------------------------------------------------------------- metrics
def _step_index_at_or_before(steps: Sequence[int], boundary: int) -> int:
    """Largest index k with steps[k] <= boundary (0 if none)."""
    if not steps:
        return 0
    return max(0, bisect.bisect_right(list(steps), boundary) - 1)


def compute_metrics(
    conf_mat: np.ndarray,
    steps: Sequence[int],
    task_to_series: Dict[str, Sequence[float]],
    task_ends: Sequence[Tuple[int, str]],
) -> Dict[str, Any]:
    """Row i of ``conf_mat`` = performance after training stage i, column j = task j (NaN for j > i)."""
    for task_name, series in task_to_series.items():
        assert np.all(np.isfinite(np.asarray(series, dtype=float))), f"Non-finite values for {task_name!r}"

    n_tasks = conf_mat.shape[0]
    fwt_per_task = [float(conf_mat[k, k]) for k in range(n_tasks)]
    bwt_per_task: List[float] = []
    bwt_task_to_task_deltas: List[List[float]] = []
    for k in range(n_tasks):
        if k == n_tasks - 1:
            bwt_task_to_task_deltas.append([])
            continue
        deltas = conf_mat[k, k] - conf_mat[k + 1:, k]
        bwt_task_to_task_deltas.append(deltas.astype(float).tolist())
        bwt_per_task.append(float(np.mean(deltas)))
    auc_per_task = [float(np.mean(conf_mat[k:, k])) for k in range(n_tasks)]

    curriculum = [name for _, name in task_ends]

    # Boundaries with an exact checkpoint (needed for precise / moving-anchor metrics).
    resolved: List[Tuple[int, str]] = []
    end_indices: List[int] = []
    for end_step, task_name in task_ends:
        idx = _step_index_at_or_before(steps, int(end_step))
        if steps[idx] != int(end_step):
            warnings.warn(f"No checkpoint at the task boundary of {task_name!r} (step {end_step}); "
                          f"skipping it for precise / moving-anchor metrics.")
            continue
        resolved.append((int(end_step), task_name))
        end_indices.append(idx)
    n_resolved = len(resolved)
    names = [name for _, name in resolved]
    # Training window of stage k: checkpoints after the previous boundary up to this one. For the
    # first task the untrained step-0 evaluation is excluded.
    first_start = 1 if (len(steps) > 1 and int(steps[0]) == 0) else 0
    start_indices = [first_start] + [end_indices[i - 1] + 1 for i in range(1, n_resolved)]

    fwt_precise_per_task: List[float] = []
    bwt_precise_per_task: List[float] = []
    bwt_precise_task_to_task_deltas: List[List[float]] = []
    auc_precise_per_task: List[float] = []
    for k in range(n_resolved):
        series = np.asarray(task_to_series[names[k]], dtype=float)
        train_window = series[start_indices[k]: end_indices[k] + 1]
        fwt_precise_per_task.append(float(np.mean(train_window)) if train_window.size else 0.0)
        post = series[start_indices[k]:]
        auc_precise_per_task.append(float(np.mean(post)) if post.size else 0.0)
        if k < n_resolved - 1:
            anchor = float(series[end_indices[k]])
            future = series[end_indices[k] + 1:]
            bwt_precise_task_to_task_deltas.append((anchor - future).astype(float).tolist())
            if future.size:
                bwt_precise_per_task.append(float(np.mean(anchor - future)))
        else:
            bwt_precise_task_to_task_deltas.append([])

    # Moving-anchor task-task interactions: Delta_{i,j} = c[i] at the start of stage j minus c[i]
    # at the end of stage j (window [end_{j-1}, end_j], inclusive).
    bwt_task_to_task_moving_anchors: List[List[float]] = []
    bwt_moving_anchors_per_task: List[float] = []
    for src in range(n_resolved):
        if src == n_resolved - 1:
            bwt_task_to_task_moving_anchors.append([])
            continue
        src_series = np.asarray(task_to_series[names[src]], dtype=float)
        values = []
        for trained in range(src + 1, n_resolved):
            win_start, win_end = end_indices[trained - 1], end_indices[trained]
            assert win_start < win_end, f"stage {trained} has no checkpoint after its start"
            values.append(float(src_series[win_start]) - float(src_series[win_end]))
        bwt_task_to_task_moving_anchors.append(values)
        bwt_moving_anchors_per_task.append(float(np.mean(values)))

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {
        "fwt": mean(fwt_per_task),
        "bwt": mean(bwt_per_task),
        "auc": mean(auc_per_task),
        "fwt_per_task": fwt_per_task,
        "bwt_per_task": bwt_per_task,
        "bwt_task_to_task_deltas": bwt_task_to_task_deltas,
        "auc_per_task": auc_per_task,
        "fwt_precise": mean(fwt_precise_per_task),
        "bwt_precise": mean(bwt_precise_per_task),
        "auc_precise": mean(auc_precise_per_task),
        "fwt_precise_per_task": fwt_precise_per_task,
        "bwt_precise_per_task": bwt_precise_per_task,
        "bwt_precise_task_to_task_deltas": bwt_precise_task_to_task_deltas,
        "auc_precise_per_task": auc_precise_per_task,
        "bwt_moving_anchors": mean(bwt_moving_anchors_per_task),
        "bwt_moving_anchors_per_task": bwt_moving_anchors_per_task,
        "bwt_task_to_task_moving_anchors": bwt_task_to_task_moving_anchors,
        "metric_task_order": curriculum,
    }


# ----------------------------------------------------------------------------- loading
def load_checkpoint_json(path: Path) -> Dict[str, Any]:
    """Eval files are JSON-lines (re-evaluations append); the last record is the latest."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert lines, f"No JSON data in {path}"
    return json.loads(lines[-1])


def parse_checkpoint_step(path: Path) -> int:
    stem = path.name.split(".")[0]
    if stem == "0_INITIAL":  # legacy name of the untrained checkpoint
        return 0
    try:
        return int(stem)
    except ValueError as e:
        raise ValueError(f"Checkpoint json must be named <global_step>.ckpt.json, got {path}") from e


def collect_ckpt_files(ckpt_dir: Path, pattern: str = "*.ckpt.json") -> List[Path]:
    files = [f for f in ckpt_dir.glob(pattern) if f.name.split(".")[0].isdigit() or f.name.startswith("0_INITIAL.")]
    return sorted(files, key=parse_checkpoint_step)


def load_name_mapping(libero_split: str, mapping_root: Path = DEFAULT_MAPPING_ROOT) -> Dict[str, str]:
    """BDDL task stem -> task key used in the eval json (LIBERO's legacy language-derived name)."""
    with (Path(mapping_root) / f"{libero_split}.json").open() as f:
        return json.load(f)


def aggregate_per_task_success(
    ckpt_files: Sequence[Path],
    libero_split: str,
    key_mapping: Dict[str, str],
    task_filter: Optional[Set[str]] = None,
    split: str = "test",
) -> Tuple[List[int], Dict[str, List[float]]]:
    """(sorted steps, {task stem: [success at each step]})."""
    tasks = sorted(task_filter if task_filter is not None else key_mapping.keys())
    step_to_success: Dict[int, Dict[str, float]] = {}
    for path in ckpt_files:
        data = load_checkpoint_json(path)
        values = {}
        for task in tasks:
            for key in (f"means_{split}/{key_mapping[task]}{MEANS_SCORE_KEY_SUFFIX}",
                        f"means_{split}/{libero_split}/{key_mapping[task]}{MEANS_SCORE_KEY_SUFFIX}"):
                if key in data:
                    values[task] = float(data[key])
                    break
            else:
                raise KeyError(f"No success value for task {task!r} in {path} (wrong --libero_split?)")
        step_to_success[parse_checkpoint_step(path)] = values
    steps = sorted(step_to_success)
    return steps, {t: [step_to_success[s][t] for s in steps] for t in tasks}


def _normalize_task_name(raw: str) -> str:
    """task_boundaries names are hdf5 stems (<bddl_stem>_demo)."""
    return raw[: -len("_demo")] if raw.endswith("_demo") else raw


def extract_task_ends(run_dir: Path) -> List[Tuple[int, str]]:
    """[(end_step, task stem)] sorted by step, from task_boundaries.json (or legacy logs.json.txt)."""
    boundaries = run_dir / "task_boundaries.json"
    out: List[Tuple[int, str]] = []
    if boundaries.is_file():
        for rec in json.loads(boundaries.read_text()):
            out.append((int(rec["end_step"]), _normalize_task_name(rec["task_name"])))
    else:
        for line in (run_dir / "logs.json.txt").read_text().splitlines():
            obj = json.loads(line) if line.strip() else {}
            if "task_end" in obj:
                out.append((int(obj["global_step"]), _normalize_task_name(obj["task_end"])))
    return sorted(out)


def expected_number_of_tasks(run_dir: Path) -> Optional[int]:
    """Curriculum length from the run's saved Hydra config (None if unknown)."""
    for cfg_path in (run_dir / ".hydra" / "config.yaml", run_dir / "config.yaml"):
        if cfg_path.is_file():
            task = (yaml.safe_load(cfg_path.read_text()) or {}).get("task", {})
            for key in ("testing_tasks", "tasks_to_evaluate"):
                if isinstance(task.get(key), list):
                    return len(task[key])
    return None


def _resolve_series_task_key(name: str, series_keys: Collection[str]) -> Optional[str]:
    keys = set(series_keys)
    norm = _normalize_task_name(name)
    if name in keys:
        return name
    if norm in keys:
        return norm
    for k in keys:
        if k.endswith(norm) or norm.endswith(k):
            return k
    return None


def build_continual_success_matrix(steps, task_to_series, task_ends) -> Tuple[np.ndarray, List[str]]:
    """S[i, j] = success on task j at the checkpoint at/before the end of stage i (NaN for j > i)."""
    assert task_ends, "No task boundaries found"
    curriculum = [name for _, name in task_ends]
    n = len(curriculum)
    conf = np.full((n, n), np.nan)
    for i in range(n):
        si = _step_index_at_or_before(steps, int(task_ends[i][0]))
        for j in range(i + 1):
            key = _resolve_series_task_key(curriculum[j], task_to_series.keys())
            if key is not None:
                conf[i, j] = float(task_to_series[key][si])
    return conf, curriculum


def run_is_complete(run_dir: Path, steps: Sequence[int], task_ends: Sequence[Tuple[int, str]],
                    expected_tasks: Optional[int]) -> Tuple[bool, str]:
    """A run counts only if it finished the whole curriculum and was evaluated at every boundary."""
    n_expected = expected_tasks if expected_tasks is not None else expected_number_of_tasks(run_dir)
    if n_expected is None:
        return False, "cannot determine curriculum length (no config); pass --expected-tasks"
    if len(task_ends) != n_expected:
        return False, f"{len(task_ends)}/{n_expected} tasks finished"
    missing = [s for s, _ in task_ends if int(s) not in set(int(x) for x in steps)]
    if missing:
        return False, f"no evaluation at task boundaries {missing}"
    return True, "complete"


# ----------------------------------------------------------------------------- plots
def _wrapped(name: str, width: int = 24) -> str:
    return textwrap.fill(name.replace("_", " "), width=width)


def plot_per_task_success(steps, task_to_series, task_ends, output_dir: Path, title: str, prefix: str = "") -> None:
    order = [n for _, n in task_ends if n in task_to_series]
    tasks = order + [t for t in task_to_series if t not in order]
    n_cols = min(3, len(tasks))
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False, sharex=True, sharey=True)
    for idx, task in enumerate(tasks):
        ax = axes[idx // n_cols][idx % n_cols]
        for xv in sorted({t for t, _ in task_ends}):
            ax.axvline(xv, color="0.4", linestyle="--", linewidth=0.9, alpha=0.75, zorder=0)
        ax.plot(steps, task_to_series[task], marker="o", markersize=3, zorder=1)
        ax.set_title(_wrapped(task), fontsize=8)
        ax.set_ylim(0.0, 1.0)
    for idx in range(len(tasks), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.text(0.5, 0.04, "Checkpoint (step)", ha="center")
    fig.text(0.04, 0.5, "Task success rate", va="center", rotation="vertical")
    fig.suptitle(f"Per-task success vs checkpoint ({title})")
    plt.tight_layout(rect=[0.06, 0.06, 0.98, 0.95])
    fig.savefig(output_dir / f"{prefix}per_task_success_vs_checkpoint.png")
    plt.close(fig)


def plot_figure_ready_successes(steps, task_to_series, task_ends, output_dir: Path, prefix: str = "") -> None:
    """Paper-style row of per-task curves; task k is shown from the moment it enters training."""
    tasks = [k for k in (_resolve_series_task_key(n, task_to_series) for _, n in task_ends) if k is not None]
    if not tasks:
        return
    end_indices = [_step_index_at_or_before(steps, int(s)) for s, _ in task_ends]
    starts = [0] + [end_indices[i - 1] + 1 for i in range(1, len(tasks))]
    steps_f = np.asarray(steps, dtype=float)
    width_ratios = [max(1, len(steps) - s) for s in starts]
    fig = plt.figure(figsize=(0.42 * sum(width_ratios), 2.8))
    gs = gridspec.GridSpec(1, len(tasks), figure=fig, width_ratios=width_ratios, wspace=0.06)
    for idx, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, idx])
        xs, ys = steps_f[starts[idx]:], np.asarray(task_to_series[task], dtype=float)[starts[idx]:]
        if xs.size == 0:
            ax.axis("off")
            continue
        pad = 0.02 * (xs[-1] - xs[0]) if xs[-1] > xs[0] else 0.05
        ax.set_xlim(xs[0] - pad, xs[-1] + pad)
        for xv in sorted({t for t, _ in task_ends}):
            if xs[0] <= xv <= xs[-1]:
                ax.axvline(float(xv), color="0.4", linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
        ax.plot(xs, ys, color=FIGURE_READY_LINE_COLOR, linewidth=FIGURE_READY_LINE_WIDTH, marker="o",
                markersize=FIGURE_READY_MARKER_SIZE, solid_capstyle="round", zorder=2)
        ax.set_title(f"T{idx}", fontsize=10)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_ylim(0.0, FIGURE_READY_Y_TOP)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12)
    fig.savefig(output_dir / f"{prefix}figure_ready_successes_plot.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}figure_ready_successes_plot.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_continual_success_matrix(conf_mat: np.ndarray, output_dir: Path, prefix: str = "") -> None:
    """Heatmap: x = training stage, y = evaluated task (1-indexed)."""
    n = conf_mat.shape[0]
    mat = conf_mat.T
    mask = np.isnan(mat)
    with plt.rc_context({"font.family": "sans-serif"}):
        fig, ax = plt.subplots(figsize=(max(4.0, n * 0.9 + 1.5), max(4.0, n * 0.9 + 1.0)))
        cmap = plt.cm.YlGnBu.copy()
        cmap.set_bad(color="white")
        ax.imshow(np.ma.masked_where(mask, mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        for r in range(n):
            for c in range(n):
                if mask[r, c]:
                    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white", edgecolor="white"))
                else:
                    v = float(mat[r, c])
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=max(10, 24 - n),
                            fontweight="bold", color="white" if v > 0.55 else "#333333")
        ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
        ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}continual_success_matrix.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------- run selection
def run_order(run_dir: Path) -> Optional[int]:
    """LIBERO task-order index encoded in the run name (..._orderXX_...), or None."""
    m = re.search(r"order(\d+)", Path(run_dir).name)
    return int(m.group(1)) if m else None


def parse_orders(raw: Optional[str]) -> Optional[Set[int]]:
    """'0,1,2' -> {0, 1, 2}; None/'' -> None (= all runs)."""
    return {int(x) for x in raw.split(",") if x.strip()} if raw else None


def keep_run(run_dir: Path, orders: Optional[Set[int]]) -> bool:
    return orders is None or run_order(run_dir) in orders


# ----------------------------------------------------------------------------- driver
def discover_run_dirs(parent: Path, pattern: str = "*.ckpt.json") -> List[Path]:
    """Immediate subfolders of `parent` that contain checkpoints/<pattern>."""
    return [c for c in sorted(parent.iterdir()) if (c / "checkpoints").is_dir() and any((c / "checkpoints").glob(pattern))]


def process_run(run_dir: Path, args) -> Dict[str, Any]:
    ckpt_files = collect_ckpt_files(run_dir / "checkpoints", args.pattern)
    assert ckpt_files, f"No checkpoint json files in {run_dir / 'checkpoints'}"
    output_dir = Path(args.output_dir) / run_dir.name if args.output_dir else run_dir / "generated_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.name}_" if args.name else ""

    mapping = load_name_mapping(args.libero_split, args.name_mapping_root)
    task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()} or None
    task_ends = extract_task_ends(run_dir)
    steps, task_to_series = aggregate_per_task_success(ckpt_files, args.libero_split, mapping, task_filter, args.split)
    (output_dir / f"{prefix}task_to_series.json").write_text(json.dumps(task_to_series, indent=2))

    conf, _ = build_continual_success_matrix(steps, task_to_series, task_ends)
    np.save(output_dir / f"{prefix}continual_success_matrix.npy", conf)
    metrics = compute_metrics(conf, steps, task_to_series, task_ends)
    (output_dir / f"{prefix}continual_metrics.json").write_text(json.dumps(metrics, indent=2))

    complete, reason = run_is_complete(run_dir, steps, task_ends, args.expected_tasks)
    sentinel = output_dir / "complete.txt"
    if complete:
        sentinel.write_text("complete\n")
    elif sentinel.exists():
        sentinel.unlink()
    if not args.no_plots:
        plot_per_task_success(steps, task_to_series, task_ends, output_dir, run_dir.name, prefix)
        plot_figure_ready_successes(steps, task_to_series, task_ends, output_dir, prefix)
        plot_continual_success_matrix(conf, output_dir, prefix)
    print(f"{run_dir.name}: fwt={metrics['fwt']:.3f} nbt={metrics['bwt']:.3f} auc={metrics['auc']:.3f} [{reason}]")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_dir", type=str, help="A run directory, or (with --batch-parent) a folder of run directories.")
    p.add_argument("--batch-parent", action="store_true", help="Process every run directory inside ckpt_dir.")
    p.add_argument("--libero_split", required=True,
                   choices=["libero_goal", "libero_10", "libero_object", "libero_spatial"], help="LIBERO suite.")
    p.add_argument("--name_mapping_root", type=Path, default=DEFAULT_MAPPING_ROOT,
                   help="Folder with <suite>.json task-name mappings (default: memory_anchors/task_name_mappings).")
    p.add_argument("--pattern", default="*.ckpt.json", help="Glob for eval files inside checkpoints/.")
    p.add_argument("--tasks", default="", help="Comma-separated BDDL task stems to include (default: all).")
    p.add_argument("--split", default="test", help="Eval split key (means_<split>/...).")
    p.add_argument("--expected-tasks", type=int, default=None, help="Curriculum length (default: from the run config).")
    p.add_argument("--output-dir", default="", help="Write outputs here instead of <run>/generated_visuals.")
    p.add_argument("--name", default="", help="Filename prefix for outputs.")
    p.add_argument("--no-plots", action="store_true", help="Only write the json/npy outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.ckpt_dir).resolve()
    runs = discover_run_dirs(root, args.pattern) if args.batch_parent else [root]
    assert runs, f"No run directories with checkpoints/{args.pattern} under {root}"
    for run_dir in runs:
        process_run(run_dir, args)


if __name__ == "__main__":
    main()
