#!/usr/bin/env python3
"""Step 1 of the NBT-High-Conflict pipeline: find the most-forgetting task pairs on reference runs.

For every run under a suite folder (--batch-parent) this builds, for each ordered pair
(source task i, trained task j > i), the source task's success curve over task j's training window
[end_{j-1}, end_j] and records
  * the moving-anchor forgetting  Delta_{i,j} = c_i(end_{j-1}) - c_i(end_j)  (paper App. B.4), and
  * the slope of a linear fit of source vs trained success (diagnostic only).
Pairs are averaged over runs (segments that occur at the same stage positions in several runs are
first merged, as in the paper's analysis) and the top-K (K=6) pairs by mean Delta are written to
<suite>/generated_visuals/_pairwise_correlations.json.

The paper selects the pairs on the RandER (1% buffer) runs and scores every method on them:

    python plotting/high_conflict/find_conflict_pairs.py results/RandER_1pct/LiberoGoal --batch-parent --libero_split libero_goal
"""
import argparse
import json
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_metrics import (  # noqa: E402
    DEFAULT_MAPPING_ROOT,
    keep_run,
    parse_orders,
    _step_index_at_or_before,
    aggregate_per_task_success,
    collect_ckpt_files,
    extract_task_ends,
)
from target_task_pair_metrics import (  # noqa: E402
    PAIRWISE_CORRELATIONS_TOP_NEGATIVE_KEY,
    PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY,
    WORST_POSITIVE_BT_PAIR_TOP_K,
)


@dataclass
class ReactionSegment:
    source_task: str
    trained_task: str
    source_stage_idx: int
    trained_stage_idx: int
    step_start: int
    step_end: int
    idx_start: int
    idx_end: int
    source_values: List[float]
    trained_values: List[float]


@dataclass
class CorrelationRecord:
    segment_id: str
    source_task: str
    trained_task: str
    n_points: int
    slope_m: float
    pearson_r: float


@dataclass
class NegativePairRecord:
    source_task: str
    trained_task: str
    mean_slope_m: float
    segment_count: int


@dataclass
class NegativeBackwardTransferPairRecord:
    source_task: str
    trained_task: str
    mean_negative_backward_transfer: float
    segment_count: int


@dataclass
class NegativeBackwardTransferRecord:
    segment_id: str
    source_task: str
    source_stage_idx: int
    trained_task: str
    trained_stage_idx: int
    n_points: int
    anchor_value: float
    negative_backward_transfer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract directed task reaction segments and plot histogram of pairwise "
            "linear-fit slopes."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Experiment directory (contains checkpoints/) or, with --batch-parent, "
        "a directory whose subfolders are such experiments.",
    )
    parser.add_argument(
        "--batch-parent",
        action="store_true",
        help="Treat ckpt_dir as a parent: run analysis across every immediate child "
        "that has checkpoints/ with at least one file matching --pattern. "
        "If no child qualifies but ckpt_dir itself does, run once on ckpt_dir.",
    )
    parser.add_argument(
        "--libero_split",
        type=str,
        required=True,
        help="LIBERO suite name used to load task-name mappings (e.g. libero_goal).",
    )
    parser.add_argument(
        "--name_mapping_root",
        type=str,
        default=str(DEFAULT_MAPPING_ROOT),
        help="Directory containing <libero_split>.json task-name mapping files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ckpt.json",
        help="Glob pattern for checkpoint JSON files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name for means_<split>/... extraction.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated task ids to include. Defaults to all mapped tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for artifacts. Defaults to <experiment_root>/generated_visuals.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional filename prefix. Defaults to experiment directory name.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--min-segment-len",
        type=int,
        default=2,
        help="Minimum overlapping length required to compute linear-fit slope and Pearson r.",
    )
    parser.add_argument(
        "--orders",
        type=str,
        default=None,
        help="Only use runs with these task-order indices, e.g. 0,1,2 (the paper's high-conflict pairs).",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Show histogram title and axis labels. Omitted by default for clean exports.",
    )
    return parser.parse_args()


def _resolve_series_task_key(name: str, series_keys: Sequence[str]) -> Optional[str]:
    keys_set = set(series_keys)
    if name in keys_set:
        return name
    if name.endswith("_demo") and name[: -len("_demo")] in keys_set:
        return name[: -len("_demo")]
    for key in keys_set:
        if key.endswith(name) or name.endswith(key):
            return key
    return None


def _compute_stage_windows(
    steps: Sequence[int], task_ends: Sequence[Tuple[int, str]]
) -> List[Tuple[int, int]]:
    """
    Stage *i*'s window is ``[end_indices[i-1], end_indices[i]]`` (inclusive at
    both ends), so the boundary checkpoint at ``end_indices[i-1]`` is shared
    between stage ``i-1``'s end and stage ``i``'s start. This matches the
    moving-anchor convention used by ``ckpt_rollout_viz.compute_metrics``: the
    anchor for a (source, trained) pair is the checkpoint at the moment the
    trained-task starts (i.e. the last checkpoint of the prior stage).
    """
    end_indices: List[int] = []
    for end_step, _ in task_ends:
        end_indices.append(_step_index_at_or_before(steps, int(end_step)))

    windows: List[Tuple[int, int]] = []
    for i, end_idx in enumerate(end_indices):
        start_idx = 0 if i == 0 else end_indices[i - 1]
        if i > 0:
            assert start_idx < end_idx, (
                f"Moving-anchor stage window must contain at least two checkpoints "
                f"for stage {i}: start={start_idx}, end={end_idx}"
            )
        windows.append((start_idx, end_idx))
    return windows


def build_directed_reaction_segments(
    steps: Sequence[int],
    task_to_series: Dict[str, Sequence[float]],
    task_ends: Sequence[Tuple[int, str]],
) -> List[ReactionSegment]:
    curriculum = [name for _, name in task_ends]
    windows = _compute_stage_windows(steps, task_ends)
    out: List[ReactionSegment] = []

    for trained_idx in range(len(curriculum)):
        trained_task = curriculum[trained_idx]
        trained_key = _resolve_series_task_key(trained_task, list(task_to_series.keys()))
        if trained_key is None:
            continue
        win_start, win_end = windows[trained_idx]
        trained_series = np.asarray(task_to_series[trained_key], dtype=float)
        for source_idx in range(trained_idx):
            source_task = curriculum[source_idx]
            source_key = _resolve_series_task_key(source_task, list(task_to_series.keys()))
            if source_key is None:
                continue
            source_series = np.asarray(task_to_series[source_key], dtype=float)
            source_values = source_series[win_start : win_end + 1]
            trained_values = trained_series[win_start : win_end + 1]
            out.append(
                ReactionSegment(
                    source_task=source_task,
                    trained_task=trained_task,
                    source_stage_idx=source_idx,
                    trained_stage_idx=trained_idx,
                    step_start=int(steps[win_start]),
                    step_end=int(steps[win_end]),
                    idx_start=win_start,
                    idx_end=win_end,
                    source_values=source_values.tolist(),
                    trained_values=trained_values.tolist(),
                )
            )
    return out


def segment_id(seg: ReactionSegment) -> str:
    return f"{seg.source_stage_idx}->{seg.trained_stage_idx}:{seg.source_task}|{seg.trained_task}"


def compute_pairwise_slopes(
    segments: Sequence[ReactionSegment], min_segment_len: int
) -> List[CorrelationRecord]:
    records: List[CorrelationRecord] = []

    for seg in segments:
        x = np.asarray(seg.trained_values, dtype=float)
        y = np.asarray(seg.source_values, dtype=float)
        if len(x) != len(y):
            print(f"This one doesn't have the same length in the segment! {seg}")
            continue

        finite_mask = np.isfinite(x) & np.isfinite(y)
        n_valid = int(np.sum(finite_mask))
        if n_valid < min_segment_len:
            continue

        x_f = x[finite_mask]
        y_f = y[finite_mask]
        if np.std(x_f) == 0.0 or np.std(y_f) == 0.0:
            continue

        coeffs = np.polyfit(x_f, y_f, deg=1)
        slope_m = float(coeffs[0])
        corr = float(np.corrcoef(x_f, y_f)[0, 1])
        if not np.isfinite(slope_m):
            continue
        if not np.isfinite(corr):
            continue

        records.append(
            CorrelationRecord(
                segment_id=segment_id(seg),
                source_task=seg.source_task,
                trained_task=seg.trained_task,
                n_points=n_valid,
                slope_m=slope_m,
                pearson_r=corr,
            )
        )
    return records


def plot_metric_histogram(
    values: Sequence[float],
    output_path: Path,
    bins: int,
    title: str,
    xlabel: str,
    *,
    show_legend: bool = False,
) -> Tuple[Path, Path]:
    assert len(values) > 0, "Cannot plot histogram with zero values."
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="black")
    if show_legend:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        mean_v = float(np.mean(values))
        med_v = float(np.median(values))
        std_v = float(np.std(values))
        ax.set_title(
            f"{title}\n"
            f"n={len(values)} | mean={mean_v:.3f}, median={med_v:.3f}, std={std_v:.3f}"
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path if output_path.suffix == ".png" else output_path.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def discover_batch_experiment_roots(parent: Path, pattern: str, orders=None) -> List[Path]:
    assert parent.is_dir(), f"Not a directory: {parent}"
    roots: List[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        ckpt_sub = child / "checkpoints"
        if not ckpt_sub.is_dir() or not any(ckpt_sub.glob(pattern)):
            continue
        if not keep_run(child, orders):
            continue
        if not (child / "generated_visuals" / "complete.txt").is_file():
            print(f"[WARN] skipping incomplete run (run compute_metrics.py first): {child.name}")
            continue
        roots.append(child)
    return roots


def resolve_experiment_roots(path: Path, pattern: str, batch_parent: bool, orders=None) -> List[Path]:
    if not batch_parent:
        return [path]
    children = discover_batch_experiment_roots(path, pattern, orders)
    if children:
        return children
    ckpt_sub = path / "checkpoints"
    if ckpt_sub.is_dir() and any(ckpt_sub.glob(pattern)):
        return [path]
    raise ValueError(
        f"--batch-parent: no experiment subfolders under {path} with "
        f"checkpoints matching {pattern!r}, and {path} is not a single experiment root."
    )


def compute_records_for_experiment(
    experiment_root: Path,
    args: argparse.Namespace,
    language_key_to_bddl_stem: Dict[str, str],
    task_filter: set[str],
) -> Tuple[List[ReactionSegment], List[CorrelationRecord]]:
    ckpt_dir = experiment_root / "checkpoints"
    if not ckpt_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {ckpt_dir}")
    ckpt_files = collect_ckpt_files(ckpt_dir, args.pattern)
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {ckpt_dir} with pattern {args.pattern!r}")

    steps, task_to_series = aggregate_per_task_success(
        ckpt_files=ckpt_files,
        libero_split=args.libero_split,
        key_mapping=language_key_to_bddl_stem,
        task_filter=task_filter,
        split=args.split,
    )
    task_ends = extract_task_ends(experiment_root)
    segments = build_directed_reaction_segments(steps, task_to_series, task_ends)
    corr_records = compute_pairwise_slopes(segments, min_segment_len=args.min_segment_len)
    return segments, corr_records


def compute_negative_backward_transfer_records(
    segments: Sequence[ReactionSegment], min_segment_len: int
) -> List[NegativeBackwardTransferRecord]:
    out: List[NegativeBackwardTransferRecord] = []
    for seg in segments:
        source_values = np.asarray(seg.source_values, dtype=float)
        finite_mask = np.isfinite(source_values)
        n_valid = int(np.sum(finite_mask))
        if n_valid < min_segment_len:
            continue
        source_values = source_values[finite_mask]
        anchor_value = float(source_values[0])
        diffs = anchor_value - source_values
        nbt_value = diffs[-1]  # anchor minus last value of the window = Delta_{i,j}

        if not np.isfinite(nbt_value):
            continue
        out.append(
            NegativeBackwardTransferRecord(
                segment_id=segment_id(seg),
                source_task=seg.source_task,
                source_stage_idx=seg.source_stage_idx,
                trained_task=seg.trained_task,
                trained_stage_idx=seg.trained_stage_idx,
                n_points=n_valid,
                anchor_value=anchor_value,
                negative_backward_transfer=nbt_value,
            )
        )
    return out


def deduplicate_records_by_segment_id(records: Sequence[CorrelationRecord]) -> List[CorrelationRecord]:
    ordered_records = sorted(records, key=lambda r: r.segment_id)
    deduped: List[CorrelationRecord] = []
    for segment_id_key, rec_iter in groupby(ordered_records, key=lambda r: r.segment_id):
        same_segment_records = list(rec_iter)
        first = same_segment_records[0]
        for rec in same_segment_records:
            assert rec.source_task == first.source_task, f"Mismatch source_task for segment_id {segment_id_key}"
            assert rec.trained_task == first.trained_task, f"Mismatch trained_task for segment_id {segment_id_key}"
        deduped.append(
            CorrelationRecord(
                segment_id=segment_id_key,
                source_task=first.source_task,
                trained_task=first.trained_task,
                n_points=int(np.mean([r.n_points for r in same_segment_records])),
                slope_m=float(np.mean([r.slope_m for r in same_segment_records])),
                pearson_r=float(np.mean([r.pearson_r for r in same_segment_records])),
            )
        )
    return deduped


def deduplicate_negative_backward_transfer_records(
    records: Sequence[NegativeBackwardTransferRecord],
) -> List[NegativeBackwardTransferRecord]:
    ordered_records = sorted(records, key=lambda r: r.segment_id)
    deduped: List[NegativeBackwardTransferRecord] = []
    for segment_id_key, rec_iter in groupby(ordered_records, key=lambda r: r.segment_id):
        same_transfer_records = list(rec_iter)
        first = same_transfer_records[0]
        for rec in same_transfer_records:
            assert rec.source_task == first.source_task, f"Mismatch source_task for segment_id {segment_id_key}"
            assert rec.source_stage_idx == first.source_stage_idx, (
                f"Mismatch source_stage_idx for segment_id {segment_id_key}"
            )
            assert rec.trained_task == first.trained_task, f"Mismatch trained_task for segment_id {segment_id_key}"
            assert rec.trained_stage_idx == first.trained_stage_idx, (
                f"Mismatch trained_stage_idx for segment_id {segment_id_key}"
            )
        deduped.append(
            NegativeBackwardTransferRecord(
                segment_id=segment_id_key,
                source_task=first.source_task,
                source_stage_idx=first.source_stage_idx,
                trained_task=first.trained_task,
                trained_stage_idx=first.trained_stage_idx,
                n_points=int(np.mean([r.n_points for r in same_transfer_records])),
                anchor_value=float(np.mean([r.anchor_value for r in same_transfer_records])),
                negative_backward_transfer=float(
                    np.mean([r.negative_backward_transfer for r in same_transfer_records])
                ),
            )
        )
    return deduped


def compute_top_negative_pairs(
    records: Sequence[CorrelationRecord], top_k: int = WORST_POSITIVE_BT_PAIR_TOP_K
) -> List[NegativePairRecord]:
    sorted_records = sorted(records, key=lambda r: (r.source_task, r.trained_task))
    out: List[NegativePairRecord] = []
    for pair_key, rec_iter in groupby(sorted_records, key=lambda r: (r.source_task, r.trained_task)):
        pair_records = list(rec_iter)
        out.append(
            NegativePairRecord(
                source_task=pair_key[0],
                trained_task=pair_key[1],
                mean_slope_m=float(np.mean([r.slope_m for r in pair_records])),
                segment_count=len(pair_records),
            )
        )
    return sorted(out, key=lambda r: r.mean_slope_m)[:top_k]


def print_top_negative_pairs(records: Sequence[NegativePairRecord]) -> None:
    print(
        f"Top {WORST_POSITIVE_BT_PAIR_TOP_K} most negative source-trained pairs by slope m:"
    )
    if not records:
        print("  (no valid pairs)")
        return
    for idx, rec in enumerate(records, start=1):
        print(
            f"  {idx}. {rec.source_task} -> {rec.trained_task} | "
            f"mean_m={rec.mean_slope_m:.6f} | n_segments={rec.segment_count}"
        )


def compute_top_positive_backward_transfer_pairs(
    records: Sequence[NegativeBackwardTransferRecord],
    top_k: int = WORST_POSITIVE_BT_PAIR_TOP_K,
) -> List[NegativeBackwardTransferPairRecord]:
    sorted_records = sorted(records, key=lambda r: (r.source_task, r.trained_task))
    out: List[NegativeBackwardTransferPairRecord] = []
    for pair_key, rec_iter in groupby(sorted_records, key=lambda r: (r.source_task, r.trained_task)):
        pair_records = list(rec_iter)
        out.append(
            NegativeBackwardTransferPairRecord(
                source_task=pair_key[0],
                trained_task=pair_key[1],
                mean_negative_backward_transfer=float(
                    np.mean([r.negative_backward_transfer for r in pair_records])
                ),
                segment_count=len(pair_records),
            )
        )
    return sorted(out, key=lambda r: r.mean_negative_backward_transfer, reverse=True)[:top_k]


def print_top_positive_backward_transfer_pairs(
    records: Sequence[NegativeBackwardTransferPairRecord],
) -> None:
    print(
        f"Top {WORST_POSITIVE_BT_PAIR_TOP_K} most positive source-trained pairs by backward transfer:"
    )
    if not records:
        print("  (no valid pairs)")
        return
    for idx, rec in enumerate(records, start=1):
        print(
            f"  {idx}. {rec.source_task} -> {rec.trained_task} | "
            f"mean_bt={rec.mean_negative_backward_transfer:.6f} | n_segments={rec.segment_count}"
        )


def build_summary(records: Sequence[CorrelationRecord], min_segment_len: int) -> Dict[str, Optional[float]]:
    slope_values = [r.slope_m for r in records]
    return {
        "n_segments": len(slope_values),
        "min_segment_len": min_segment_len,
        "mean_m": float(np.mean(slope_values)) if slope_values else None,
        "median_m": float(np.median(slope_values)) if slope_values else None,
        "std_m": float(np.std(slope_values)) if slope_values else None,
        "min_m": float(np.min(slope_values)) if slope_values else None,
        "max_m": float(np.max(slope_values)) if slope_values else None,
    }


def build_negative_backward_transfer_summary(
    records: Sequence[NegativeBackwardTransferRecord],
) -> Dict[str, Optional[float]]:
    values = [r.negative_backward_transfer for r in records]
    return {
        "n_values": len(values),
        "mean_negative_backward_transfer": float(np.mean(values)) if values else None,
        "median_negative_backward_transfer": float(np.median(values)) if values else None,
        "std_negative_backward_transfer": float(np.std(values)) if values else None,
        "min_negative_backward_transfer": float(np.min(values)) if values else None,
        "max_negative_backward_transfer": float(np.max(values)) if values else None,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.ckpt_dir).resolve()
    experiment_roots = resolve_experiment_roots(root, args.pattern, args.batch_parent, parse_orders(args.orders))

    name_mapping_path = Path(args.name_mapping_root) / f"{args.libero_split}.json"
    with name_mapping_path.open("r") as f:
        language_key_to_bddl_stem = json.load(f)

    if args.tasks:
        task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}
    else:
        task_filter = set(language_key_to_bddl_stem.keys())

    all_segments: List[ReactionSegment] = []
    all_records: List[CorrelationRecord] = []
    all_negative_backward_transfer_records: List[NegativeBackwardTransferRecord] = []
    run_summaries: List[Dict[str, object]] = []
    for experiment_root in experiment_roots:
        segments, corr_records = compute_records_for_experiment(
            experiment_root=experiment_root,
            args=args,
            language_key_to_bddl_stem=language_key_to_bddl_stem,
            task_filter=task_filter,
        )
        negative_backward_transfer_records = compute_negative_backward_transfer_records(
            segments=segments,
            min_segment_len=args.min_segment_len,
        )
        all_segments.extend(segments)
        all_records.extend(corr_records)
        all_negative_backward_transfer_records.extend(negative_backward_transfer_records)
        run_summaries.append(
            {
                "experiment": experiment_root.name,
                "n_segments": len(segments),
                "n_records": len(corr_records),
                "n_negative_backward_transfer_records": len(negative_backward_transfer_records),
            }
        )

    working_records = (
        deduplicate_records_by_segment_id(all_records) if args.batch_parent else list(all_records)
    )
    working_negative_backward_transfer_records = (
        deduplicate_negative_backward_transfer_records(all_negative_backward_transfer_records)
        if args.batch_parent
        else list(all_negative_backward_transfer_records)
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = root / "generated_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.name if args.name else ""
    segments_path = output_dir / f"{prefix}_pairwise_reaction_segments.json"
    correlations_path = output_dir / f"{prefix}_pairwise_correlations.json"
    histogram_path = output_dir / f"{prefix}_pairwise_correlation_histogram.png"
    negative_backward_transfer_histogram_path = (
        output_dir / f"{prefix}_negative_backward_transfer_histogram.png"
    )

    with segments_path.open("w") as f:
        json.dump(
            {
                "experiment_roots": [p.name for p in experiment_roots],
                "n_segments": len(all_segments),
                "segments": [asdict(s) for s in all_segments],
            },
            f,
            indent=2,
        )

    slope_values = [r.slope_m for r in working_records]
    negative_backward_transfer_values = [
        r.negative_backward_transfer for r in working_negative_backward_transfer_records
    ]
    top_negative_pairs = compute_top_negative_pairs(working_records)
    print_top_negative_pairs(top_negative_pairs)
    top_positive_backward_transfer_pairs = compute_top_positive_backward_transfer_pairs(
        working_negative_backward_transfer_records
    )
    print_top_positive_backward_transfer_pairs(top_positive_backward_transfer_pairs)
    summary = build_summary(working_records, min_segment_len=args.min_segment_len)
    negative_backward_transfer_summary = build_negative_backward_transfer_summary(
        working_negative_backward_transfer_records
    )
    with correlations_path.open("w") as f:
        json.dump(
            {
                "experiment_roots": [p.name for p in experiment_roots],
                "batch_parent_mode": args.batch_parent,
                "summary": summary,
                "negative_backward_transfer_summary": negative_backward_transfer_summary,
                "run_summaries": run_summaries,
                "records": [asdict(r) for r in working_records],
                "negative_backward_transfer_records": [
                    asdict(r) for r in working_negative_backward_transfer_records
                ],
                PAIRWISE_CORRELATIONS_TOP_NEGATIVE_KEY: [
                    asdict(r) for r in top_negative_pairs
                ],
                PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY: [
                    asdict(r) for r in top_positive_backward_transfer_pairs
                ],
            },
            f,
            indent=2,
        )

    histogram_png, histogram_pdf = plot_metric_histogram(
        values=slope_values,
        output_path=histogram_path,
        bins=args.bins,
        title=f"{prefix} pairwise reaction slopes",
        xlabel="Linear-fit slope (m)",
        show_legend=args.legend,
    )
    nbt_histogram_png, nbt_histogram_pdf = plot_metric_histogram(
        values=negative_backward_transfer_values,
        output_path=negative_backward_transfer_histogram_path,
        bins=args.bins,
        title=f"{prefix} negative backward transfer",
        xlabel="Negative backward transfer",
        show_legend=args.legend,
    )

    print(f"Saved segments: {segments_path}")
    print(f"Saved slopes and debug correlations: {correlations_path}")
    print(f"Saved histogram: {histogram_png}")
    print(f"Saved histogram PDF: {histogram_pdf}")
    print(f"Saved NBT histogram: {nbt_histogram_png}")
    print(f"Saved NBT histogram PDF: {nbt_histogram_pdf}")


if __name__ == "__main__":
    main()
