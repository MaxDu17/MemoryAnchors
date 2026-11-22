#!/usr/bin/env python3
"""Memory Anchor Concentration of an ER buffer (paper Table 2 "Anchor Conc.", App. B.5). Needs a GPU.

For every task boundary N >= 2 of a training run:
  1. load the run's task-end checkpoint after task N-1 (checkpoints/task<N-1>_<task>.ckpt),
  2. recompute the Memory Anchors of task N with that policy (AnchorER `nearest_neighbor_ood`
     selection: the `--anchor_select_proportion` of all past data closest to the action-disagreeing
     new data; paper: top 10%),
  3. take the run's actual buffer (selected_indices_<task>.npy), optionally restricted to its most
     sampled `--selected-sample-proportion` (paper: 40%; uniform for buffers sampled uniformly,
     MIR's selection counts with --selected-sample-mode mir_topk),
  4. report |anchors ∩ buffer sample| / |buffer sample|.
Works for runs of any method (RandER / AnchorER / MIR / GSS); for methods without a selection
config the AnchorER defaults below are used to recompute anchors.

Requires full training-run directories (checkpoints + selected_indices_*.npy), not results/.

    python analysis/anchor_concentration.py outputs/<exp>/LiberoGoal \
        --anchor_select_proportion 0.1 --selected-sample-proportion 0.4
    python analysis/anchor_concentration.py outputs/<mir_exp>/LiberoGoal \
        --anchor_select_proportion 0.1 --selected-sample-proportion 0.4 --selected-sample-mode mir_topk

Writes <run>/generated_visuals/er_anchor_buffer_overlap.json per run and, for a suite folder,
<suite>/generated_visuals/er_anchor_buffer_overlap_aggregate.json (mean ± SEM over runs).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

import memory_anchors  # noqa: F401  (registers OmegaConf resolvers)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors import paths
from memory_anchors.algos.anchor_er import AnchorER
from memory_anchors.common.torch_util import move_batch_to_device
from memory_anchors.utils.libero_task_order import register_libero_task_order_resolver

register_libero_task_order_resolver()

# Run configs written by the original (pre-release) codebase use `umi_day.*` targets.
_LEGACY_TARGETS = [
    ("umi_day.train_network.continual_learning_algos.SelectiveER.SelectiveEpisodicReplay", "memory_anchors.algos.anchor_er.AnchorER"),
    ("umi_day.train_network.continual_learning_algos.SelectiveER", "memory_anchors.algos.anchor_er"),
    ("umi_day.train_network.continual_learning_algos.GSS", "memory_anchors.algos.gss"),
    ("umi_day.train_network.continual_learning_algos.MIRER", "memory_anchors.algos.mir"),
    ("umi_day.train_network.workspace.cl_train_policy_workspace", "memory_anchors.workspace.train_workspace"),
    ("umi_day.train_network.common", "memory_anchors.common"),
    ("umi_day.train_network", "memory_anchors"),
    ("umi_day.common", "memory_anchors.common"),
]
_LIBERO_ACTION_COMPONENTS = [{"type": "pos", "size": 3}, {"type": "rot", "size": 6}, {"type": "gripper_width", "size": 1}]
_DEFAULT_REPRESENTATION_CFG = {"individually_normalized": False, "use_select_keys": False,
                               "keep_modalities": ["agentview_rgb", "eye_in_hand_rgb"]}
_DEFAULT_SELECTION_CFG = {  # AnchorER defaults (config/learning_algorithm/anchor_er.yaml)
    "id_partition": True, "partition_device": "cuda", "noise_level": 20, "std_cutoff": 2, "n_neighbors": 5,
    "consolidation": "median", "random_proportion": 0.8, "invert_selection": False, "invert_remove_proportion": 0.05,
    "sequential_budget": "per_task", "selection_eval_mode": False, "ramp_task_contribution": False, "ramp_prop": 0.5,
    "counterexample": False, "counterexample_cfg": {"ramp_rate": 1},
}


class _OfflineAccelerator:
    """Single-process stand-in for `accelerate.Accelerator` used by the selection code."""

    is_main_process = True

    def unwrap_model(self, model):
        return model

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def main_process_first(self):
        return nullcontext()

    def wait_for_everyone(self):
        return None


class _DeviceMovingPolicy:
    """Moves batches to the policy device before the selection helpers use them."""

    def __init__(self, policy):
        self._policy = policy

    @property
    def device(self):
        return self._policy.device

    def __getattr__(self, name: str):
        return getattr(self._policy, name)

    def compute_obs_representations(self, batch, **kwargs):
        return self._policy.compute_obs_representations(move_batch_to_device(batch, self._policy.device), **kwargs)

    def compute_ood_score(self, batch, **kwargs):
        return self._policy.compute_ood_score(move_batch_to_device(batch, self._policy.device), **kwargs)


class _CheckpointWorkspace(BaseWorkspace):
    """Loads model / EMA weights and the training split of a training checkpoint."""

    include_keys = ["global_step", "epoch", "training_split_info"]
    exclude_keys = ["_output_dir"]

    def __init__(self, cfg):
        super().__init__(cfg, output_dir=None)
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.training_split_info = None


# ----------------------------------------------------------------------------- config helpers
def _remap_legacy_targets(obj):
    """Recursively rewrite legacy `_target_` strings in a plain (unresolved) config container."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key == "_target_" and isinstance(value, str):
                for old, new in _LEGACY_TARGETS:
                    if value.startswith(old):
                        value = new + value[len(old):]
                        break
                out[key] = value
            else:
                out[key] = _remap_legacy_targets(value)
        return out
    if isinstance(obj, list):
        return [_remap_legacy_targets(v) for v in obj]
    return obj


def _resolve_dataset_paths(cfg, data_root: Path) -> None:
    """Old configs store dataset paths relative to the repo (datasets/libero_abs/<suite>)."""
    for node in (cfg.task, cfg.task.dataset, cfg.task.env_runner):
        path = node.get("dataset_path", None) if OmegaConf.is_dict(node) else None
        if isinstance(path, str) and not os.path.isabs(path) and "${" not in path:
            rel = path[len("datasets/"):] if path.startswith("datasets/") else path
            node.dataset_path = str(data_root / rel)
    if "dataset_basepath" in cfg.task.dataset:
        cfg.task.dataset.dataset_basepath = None


def prepare_cfg(run_dir: Path, task_order: List[str], data_root: Path):
    raw = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=False)
    cfg = OmegaConf.create(_remap_legacy_targets(raw))
    cfg.task.dataset.task_order = list(task_order)
    _resolve_dataset_paths(cfg, data_root)
    cfg.dataloader.shuffle = False
    cfg.val_dataloader.shuffle = False
    for meta in (cfg.get("shape_meta"), cfg.task.get("shape_meta"), cfg.model.get("shape_meta")):
        if meta is not None and "action" in meta and meta.action.get("components") is None:
            meta.action.components = OmegaConf.create(_LIBERO_ACTION_COMPONENTS)
    la = cfg.learning_algorithm
    if la.get("representation_cfg") is None:
        la.representation_cfg = OmegaConf.create(_DEFAULT_REPRESENTATION_CFG)
    if la.get("selection_cfg") is None:
        la.selection_cfg = OmegaConf.create(_DEFAULT_SELECTION_CFG)
        print("[anchor_concentration] run has no selection_cfg (MIR/GSS/...): using AnchorER defaults.")
    for key, value in _DEFAULT_SELECTION_CFG.items():
        if key not in la.selection_cfg:
            la.selection_cfg[key] = value
    return cfg


# ----------------------------------------------------------------------------- core
def load_task_order(run_dir: Path) -> List[str]:
    return [r["task_name"] for r in json.loads((run_dir / "task_boundaries.json").read_text())]


def _nametag(task_name: str) -> str:
    return task_name[:150]


def load_policy(cfg, ckpt_path: Path):
    ws = _CheckpointWorkspace(cfg)
    ws.load_checkpoint(path=str(ckpt_path))
    policy = ws.model  # selection during training uses the non-EMA model in train mode
    policy.train()
    policy.cuda()
    return ws, policy


def build_anchor_er(policy, cfg) -> AnchorER:
    la = cfg.learning_algorithm
    algo = AnchorER(
        base_learner=_DeviceMovingPolicy(policy), filter_size=la.filter_size, filter_size_mode=la.filter_size_mode,
        selection_cfg=la.selection_cfg, selection_mode="nearest_neighbor_ood", batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers), er_weight=la.get("er_weight", 1),
        representation_cfg=la.representation_cfg, preload_dataset=None, add_new_data=True,
    )
    algo.register_accelerator(_OfflineAccelerator())
    return algo


def recompute_memory_anchors(algo: AnchorER, past_datasets, current_dataset, anchor_select_proportion: float) -> np.ndarray:
    """Steps 1-3 of Alg. 1 (no random fill); returns indices into ConcatDataset(past_datasets)."""
    pretrain = ConcatDataset(past_datasets)
    past_reprs, normalizer = algo.compute_dataset_representations(pretrain)
    new_reprs, _ = algo.compute_dataset_representations(current_dataset, normalizer=normalizer)
    id_idx = None
    if algo.selection_cfg.id_partition:
        _, _, id_idx = algo.partition_new_data(past_reprs, new_reprs)
    ood_stats = algo.compute_OOD_baseline(ConcatDataset([d.get_validation_dataset() for d in past_datasets]))
    target_reprs, *_ = algo.find_OOD_data(ood_stats, current_dataset, new_reprs, id_selection_indices=id_idx)
    n_anchors = int(anchor_select_proportion * len(pretrain))
    assert 0 < n_anchors <= len(pretrain), f"anchor_select_proportion gives {n_anchors} anchors"
    anchors, _, _ = algo.find_nearest_neighbors(past_reprs.detach().cpu().numpy(), target_reprs.detach().cpu().numpy(), n_anchors)
    return np.asarray(anchors, dtype=np.int64)


def compared_buffer(selected: np.ndarray, proportion: float, mode: str, mir_json: Path, rng) -> np.ndarray:
    """The buffer sample compared against the anchors (paper: the 40% most sampled)."""
    assert 0.0 < proportion <= 1.0
    if mode == "uniform":  # buffers sampled uniformly at train time: a random subset is representative
        if proportion == 1.0:
            return selected
        return selected[rng.choice(len(selected), size=int(proportion * len(selected)), replace=False)]
    report = json.loads(mir_json.read_text())  # MIR: global index -> times retrieved
    ranked = sorted(((int(i), int(c)) for i, c in report.items()), key=lambda kv: (-kv[1], kv[0]))
    k = int(proportion * len(selected))
    assert 0 < k <= len(ranked), f"need top-{k} MIR indices, report has {len(ranked)}"
    return np.asarray([i for i, _ in ranked[:k]], dtype=np.int64)


def overlap_stats(anchors: np.ndarray, buffer_sample: np.ndarray) -> Dict[str, Any]:
    a, b = set(anchors.tolist()), set(buffer_sample.tolist())
    overlap = len(a & b)
    return {"num_anchors": len(a), "num_selected": int(buffer_sample.size), "num_selected_unique": len(b),
            "num_overlap": overlap,
            # concentration = share of the (unique) compared buffer that are recomputed anchors
            "overlap_proportion": overlap / len(b),
            "anchor_recall": overlap / len(a)}


def run_concentration(run_dir: Path, args) -> Dict[str, Any]:
    task_order = load_task_order(run_dir)
    cfg = prepare_cfg(run_dir, task_order, Path(args.data_root))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset_generator = hydra.utils.instantiate(cfg.task.dataset)
    per_task = []
    for n in range(2, len(task_order) + 1):
        task = task_order[n - 1]
        if args.tasks and task not in args.tasks:
            continue
        ckpt = run_dir / "checkpoints" / f"task{n - 1}_{task_order[n - 2]}.ckpt"
        sel_path = run_dir / f"selected_indices_{_nametag(task)}.npy"
        assert ckpt.is_file(), f"missing task-end checkpoint {ckpt}"
        assert sel_path.is_file(), f"missing {sel_path}"
        print(f"=== task {n}/{len(task_order)}: {task}")
        seed = args.seed + n
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ws, policy = load_policy(cfg, ckpt)
        dataset_generator.load_custom_training_split_info(ws.training_split_info)
        past = [dataset_generator.load_subset_tasks([t]) for t in task_order[: n - 1]]
        current = dataset_generator.load_subset_tasks([task])
        selected = np.asarray(np.load(sel_path), dtype=np.int64).reshape(-1)
        sample = compared_buffer(selected, args.selected_sample_proportion, args.selected_sample_mode,
                                 run_dir / f"mir_selection_{_nametag(task)}.json", np.random.default_rng(seed))
        anchors = recompute_memory_anchors(build_anchor_er(policy, cfg), past, current, args.anchor_select_proportion)
        stats = overlap_stats(anchors, sample)
        per_task.append({"task": task, "task_index": n, **stats})
        print(f"  concentration={stats['overlap_proportion']:.4f} ({stats['num_overlap']}/{stats['num_selected_unique']})")
        del ws, policy, past, current
        torch.cuda.empty_cache()
    assert per_task, "no task boundaries evaluated"
    result = {"run": run_dir.name, "anchor_select_proportion": args.anchor_select_proportion,
              "selected_sample_proportion": args.selected_sample_proportion,
              "selected_sample_mode": args.selected_sample_mode, "per_task": per_task,
              "mean_overlap_proportion": float(np.mean([t["overlap_proportion"] for t in per_task])),
              "mean_anchor_recall": float(np.mean([t["anchor_recall"] for t in per_task]))}
    out = run_dir / "generated_visuals" / "er_anchor_buffer_overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"mean concentration {result['mean_overlap_proportion']:.4f} -> {out}")
    return result


def is_run_dir(path: Path) -> bool:
    return (path / ".hydra" / "config.yaml").is_file() and (path / "task_boundaries.json").is_file()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="A training run dir, or a suite folder of run dirs (aggregated).")
    p.add_argument("--data-root", default=paths.data_root(), help="Folder containing libero_abs/<suite>/ (default: $MEMORY_ANCHORS_DATA or <repo>/data).")
    p.add_argument("--anchor_select_proportion", type=float, default=0.1, help="Anchors = this fraction of all past data (paper: 0.1).")
    p.add_argument("--selected-sample-proportion", type=float, default=0.4, help="Compared buffer fraction (paper: 0.4).")
    p.add_argument("--selected-sample-mode", choices=["uniform", "mir_topk"], default="uniform")
    p.add_argument("--tasks", nargs="+", default=None, help="Only these incoming tasks (names from task_boundaries.json).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if is_run_dir(args.path):
        run_concentration(args.path.resolve(), args)
        return
    runs = [r for r in sorted(args.path.resolve().iterdir()) if r.is_dir() and is_run_dir(r) and any(r.glob("selected_indices_*.npy"))]
    assert runs, f"no training runs with selected_indices_*.npy under {args.path}"
    results = [run_concentration(r, args) for r in runs]
    vals = [r["mean_overlap_proportion"] for r in results]
    sem = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    agg = {"suite": args.path.name, "n_runs": len(vals), "mean_overlap_proportion": statistics.mean(vals),
           "sem_overlap_proportion": sem, "per_run": [{"run": r["run"], "mean_overlap_proportion": r["mean_overlap_proportion"]} for r in results]}
    out = args.path / "generated_visuals" / "er_anchor_buffer_overlap_aggregate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2) + "\n")
    print(f"[aggregate] concentration {agg['mean_overlap_proportion']:.4f} ± {sem:.4f} (n={len(vals)}) -> {out}")


if __name__ == "__main__":
    main()
