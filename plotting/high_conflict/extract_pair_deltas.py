#!/usr/bin/env python3
"""Step 2 of the NBT-High-Conflict pipeline: score a method on the reference high-conflict pairs.

For every suite of `reference_root` that has generated_visuals/_pairwise_correlations.json (from
find_conflict_pairs.py), read its top-K (K=6) most-forgetting (source, trained) task pairs and, for
every complete run of `target_root/<suite>`, look up the moving-anchor Delta_{i,j} of those pairs
(`bwt_task_to_task_moving_anchors` in the run's continual_metrics.json). Pairs whose trained task
comes before the source task in a run's order do not exist in that run and are skipped.

Writes <target_root>/<suite>/generated_visuals/top6_bt_pair_delta_extraction.json, which
plotting/compare_methods.py turns into the `bwt_targettasks` bars.

    python plotting/high_conflict/extract_pair_deltas.py results/RandER_1pct results/AnchorER-10_1pct
    python plotting/high_conflict/extract_pair_deltas.py results/RandER_1pct results/RandER_1pct   # the reference itself
"""
import argparse
import sys
import json
import math
import statistics
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_metrics import keep_run, parse_orders  # noqa: E402
from target_task_pair_metrics import (  # noqa: E402
    PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY,
    TARGET_TASK_EXTRACTION_AGGREGATE_PAIRS_KEY,
    TARGET_TASK_EXTRACTION_FILENAME,
    WORST_POSITIVE_BT_PAIR_TOP_K,
)

# Suite-level pool of every extracted pair's moving-anchor BWT (runs × pairs).
EXTRACTED_PAIRS_AGGREGATE_KEY = "extracted_pairs_aggregate"
EXTRACTED_PAIR_VALUE_KEY = "bwt_task_to_task_moving_anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Extract per-run moving-anchor BWT values for suite-level top "
            f"{WORST_POSITIVE_BT_PAIR_TOP_K} positive BT pairs. Per-pair value is read from "
            "`bwt_task_to_task_moving_anchors` in each run's continual_metrics, "
            "which uses the same anchor convention as the aggregate "
            "`mean_negative_backward_transfer` produced by "
            "task_pairwise_correlation_viz."
        )
    )
    parser.add_argument(
        "reference_root",
        type=str,
        help="Reference root containing suite directories with aggregate generated_visuals.",
    )
    parser.add_argument(
        "target_root",
        type=str,
        help="Target root with matching suite/run structure.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help=(
            "Output json path. Defaults to "
            f"<reference_root>/generated_visuals/{TARGET_TASK_EXTRACTION_FILENAME}"
        ),
    )
    parser.add_argument(
        "--orders",
        type=str,
        default=None,
        help="Only score runs with these task-order indices, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional run label saved in output metadata.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object json at {path}")
    return data


def mean_and_sem(values: List[float]) -> Dict[str, Optional[float] | int]:
    """Mean and SEM (sample SD / sqrt(n), ddof=1). n < 2 → sem=0.0. Empty → mean/sem None."""
    n = len(values)
    if n == 0:
        return {"mean": None, "sem": None, "n": 0}
    mean = float(statistics.fmean(values))
    if n < 2:
        sem = 0.0
    else:
        sem = float(statistics.stdev(values) / math.sqrt(n))
    return {"mean": mean, "sem": sem, "n": n}


def discover_reference_suites(reference_root: Path) -> List[Tuple[str, Path]]:
    assert reference_root.is_dir(), f"Not a directory: {reference_root}"
    out: List[Tuple[str, Path]] = []
    for child in sorted(reference_root.iterdir()):
        if not child.is_dir():
            continue
        aggregate_path = child / "generated_visuals" / "_pairwise_correlations.json"
        if aggregate_path.is_file():
            out.append((child.name, aggregate_path))
    print(f"Found the following stat paths: {out}")
    return out


def extract_worst_positive_bt_pairs(
    aggregate_json: Dict, aggregate_path: Path
) -> List[Tuple[str, str, Dict]]:
    assert PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY in aggregate_json, (
        f"Missing {PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY} in {aggregate_path}"
    )
    raw_pairs = aggregate_json[PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY]
    if not isinstance(raw_pairs, list):
        raise ValueError(
            f"{PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY} must be a list in {aggregate_path}"
        )
    out: List[Tuple[str, str, Dict]] = []
    for idx, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"top pair entry {idx} must be an object in {aggregate_path}")
        assert "source_task" in item and "trained_task" in item, (
            f"top pair entry {idx} missing source_task/trained_task in {aggregate_path}"
        )
        src = item["source_task"]
        trg = item["trained_task"]
        if not isinstance(src, str) or not isinstance(trg, str):
            raise ValueError(
                f"source_task/trained_task must be strings for entry {idx} in {aggregate_path}"
            )
        out.append((src, trg, item))
    return out


ORDERS = None  # set from --orders


def discover_run_dirs(suite_root: Path) -> List[Path]:
    if not suite_root.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(suite_root.iterdir()):
        if not child.is_dir():
            continue
        gv = child / "generated_visuals"
        if not keep_run(child, ORDERS):
            continue
        if (gv / "continual_metrics.json").is_file() and (gv / "complete.txt").is_file():
            out.append(child)
    return out


def compute_bwt_moving_anchor(
    continual_metrics: Dict,
    source_task: str,
    trained_task: str,
    run_metrics_path: Path,
) -> float | None:
    assert "metric_task_order" in continual_metrics, f"Missing metric_task_order in {run_metrics_path}"
    assert "bwt_task_to_task_moving_anchors" in continual_metrics, (
        f"Missing bwt_task_to_task_moving_anchors in {run_metrics_path}"
    )
    order = continual_metrics["metric_task_order"]
    bwt = continual_metrics["bwt_task_to_task_moving_anchors"]

    if source_task not in order or trained_task not in order:
        warnings.warn(f"pair {source_task}->{trained_task} not in the task order of {run_metrics_path}")
        return None

    source_idx = order.index(source_task)
    trained_idx = order.index(trained_task)

    if trained_idx <= source_idx:
        return None

    row = bwt[source_idx]
    if not isinstance(row, list):
        raise ValueError(f"bwt row must be list at source index {source_idx} in {run_metrics_path}")
    rel_idx = trained_idx - source_idx - 1
    if rel_idx >= len(row):
        raise IndexError(f"moving-anchor row {source_idx} too short in {run_metrics_path}")
    value = row[rel_idx]
    if not isinstance(value, (int, float)):
        raise ValueError(f"BWT moving-anchor value must be numeric in {run_metrics_path}")
    return float(value)


def extract_for_run(
    run_dir: Path,
    selected_pairs: List[Tuple[str, str, Dict]],
) -> Dict:
    run_metrics_path = run_dir / "generated_visuals" / "continual_metrics.json"
    run_metrics = load_json(run_metrics_path)

    extracted: List[Dict] = []
    missing_in_run: List[Dict] = []
    for source_task, trained_task, pair_meta in selected_pairs:
        moving_anchor_value = compute_bwt_moving_anchor(
            continual_metrics=run_metrics,
            source_task=source_task,
            trained_task=trained_task,
            run_metrics_path=run_metrics_path,
        )
        if moving_anchor_value is None:
            missing_in_run.append(
                {
                    "source_task": source_task,
                    "trained_task": trained_task,
                    "reason": "trained_before_or_missing_relative_value",
                }
            )
            continue

        extracted.append(
            {
                "source_task": source_task,
                "trained_task": trained_task,
                EXTRACTED_PAIR_VALUE_KEY: moving_anchor_value,
                "aggregate_pair_meta": pair_meta,
            }
        )

    return {
        "run_name": run_dir.name,
        "run_path": run_dir.name,
        "extracted_pairs": extracted,
        "missing_pairs": missing_in_run,
    }


def aggregate_extracted_pair_values(runs_out: List[Dict]) -> Dict:
    """Pool moving-anchor BWT across all runs × extracted pairs; report mean + SEM."""
    pooled: List[float] = []
    for run in runs_out:
        for pair in run["extracted_pairs"]:
            pooled.append(float(pair[EXTRACTED_PAIR_VALUE_KEY]))
    stats = mean_and_sem(pooled)
    return {
        "value_key": EXTRACTED_PAIR_VALUE_KEY,
        "mean": stats["mean"],
        "sem": stats["sem"],
        "n": stats["n"],
    }


def extract_for_side(
    side_root: Path,
    suite_name: str,
    selected_pairs: List[Tuple[str, str, Dict]],
) -> Dict:
    suite_dir = side_root / suite_name
    if not suite_dir.is_dir():
        return {
            "suite_path": suite_dir.name,
            "suite_missing": True,
            "runs": [],
            "missing_run_dirs": [],
            EXTRACTED_PAIRS_AGGREGATE_KEY: aggregate_extracted_pair_values([]),
        }
    print(f"\t Accessing target directory {suite_dir}")
    run_dirs = discover_run_dirs(suite_dir)
    runs_out = [
        extract_for_run(run_dir, selected_pairs)
        for run_dir in run_dirs
    ]
    return {
        "suite_path": suite_dir.name,
        "suite_missing": False,
        "runs": runs_out,
        "missing_run_dirs": [],
        EXTRACTED_PAIRS_AGGREGATE_KEY: aggregate_extracted_pair_values(runs_out),
    }


def main() -> None:
    global ORDERS
    args = parse_args()
    ORDERS = parse_orders(args.orders)
    reference_root = Path(args.reference_root).resolve()
    target_root = Path(args.target_root).resolve()

    suites = discover_reference_suites(reference_root)
    assert suites, (
        f"No suite aggregate files found at <suite>/generated_visuals/_pairwise_correlations.json "
        f"under {reference_root}"
    )

    for suite_name, aggregate_path in suites:
        aggregate_json = load_json(aggregate_path)
        selected_pairs = extract_worst_positive_bt_pairs(aggregate_json, aggregate_path)
        print(f"Using suite name {suite_name}")

        target_side = extract_for_side(target_root, suite_name, selected_pairs)
        suite_output = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "name": args.name,
            "reference_root": reference_root.name,
            "target_root": target_root.name,
            "suite_name": suite_name,
            "aggregate_path": f"{reference_root.name}/{suite_name}/generated_visuals/{aggregate_path.name}",
            TARGET_TASK_EXTRACTION_AGGREGATE_PAIRS_KEY: [
                pair_meta for _, _, pair_meta in selected_pairs
            ],
            "target": target_side,
        }

        if args.output_json:
            output_path = Path(args.output_json).resolve()
            if output_path.suffix == ".json":
                suite_output_path = (
                    output_path.parent
                    / f"{output_path.stem}_{suite_name}{output_path.suffix}"
                )
            else:
                suite_output_path = output_path / f"{suite_name}.json"
        else:
            suite_output_path = (
                target_root
                / suite_name
                / "generated_visuals"
                / TARGET_TASK_EXTRACTION_FILENAME
            )
        suite_output_path.parent.mkdir(parents=True, exist_ok=True)
        with suite_output_path.open("w") as f:
            json.dump(suite_output, f, indent=2)
        print(f"Saved: {suite_output_path}")


if __name__ == "__main__":
    main()
