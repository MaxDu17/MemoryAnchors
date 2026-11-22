#!/usr/bin/env python3
"""Launch a grid of continual-learning runs (method x suites x task orders) on local GPUs.

Each run is `memory_anchors/train.py` under `accelerate launch`. Runs are queued and started
whenever a GPU has `--mem-per-run-mb` of free memory, so several runs can share a GPU.

Example (the starter anchor-removal experiment, LIBERO-Goal, 3 task orders):

    python scripts/launch.py --name RandER_1000 --method rand_er --buffer 1000 \
        --suites goal --orders 0,1,2 --gpus 0,1
    python scripts/launch.py --name RandER-10pct_1000 --method anchor_removal --buffer 1000 \
        --set learning_algorithm.selection_cfg.invert_remove_proportion=0.10 \
        --suites goal --orders 0,1,2 --gpus 0,1

Outputs: outputs/<name>/<SuiteFolder>/<date>-<time>_<name>_<suite>_orderXX/ (one Hydra run dir per
run), plus outputs/<name>/launch_state.json (resumable: completed runs are skipped on relaunch)
and outputs/<name>/launch_logs/<run>.log (stdout/stderr of each run).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gpu_utils import cuda_to_egl_map, free_memory_mb, free_port, parse_int_list, pick_gpu, visible_devices  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "memory_anchors" / "train.py"
METHODS = sorted(p.stem for p in (REPO / "memory_anchors" / "config" / "method").glob("*.yaml"))
SUITES = {  # alias -> (hydra task, output folder)
    "goal": ("libero_goal", "LiberoGoal"),
    "10": ("libero_10", "Libero10"),
    "long": ("libero_10", "Libero10"),
    "object": ("libero_object", "LiberoObject"),
    "spatial": ("libero_spatial", "LiberoSpatial"),
    "goal_resample": ("libero_goal_resample", "LiberoGoal"),  # Fig. 2B (see experiments/buffer_resampling.sh)
}
for _alias, (_task, _folder) in list(SUITES.items()):
    SUITES[_task] = (_task, _folder)


@dataclass(frozen=True)
class Run:
    key: str
    task: str
    folder: Path
    order: int
    exp_name: str


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def buffer_overrides(buffer: str | None) -> list[str]:
    """--buffer 1000 -> 1000 examples; --buffer 0.01 -> 1% of past data."""
    if buffer is None:
        return []
    value = float(buffer)
    if value >= 1 and value == int(value) and "." not in buffer:
        return [f"learning_algorithm.filter_size={int(value)}", "learning_algorithm.filter_size_mode=examples"]
    assert 0 < value <= 1, f"--buffer must be a positive integer (examples) or a proportion in (0, 1], got {buffer}"
    return [f"learning_algorithm.filter_size={value}", "learning_algorithm.filter_size_mode=prop"]


def build_runs(args) -> list[Run]:
    runs = []
    for order in args.orders:
        for alias in args.suites:
            assert alias in SUITES, f"Unknown suite {alias!r}; choose from {sorted(SUITES)}"
            task, folder = SUITES[alias]
            exp_name = f"{args.name}_{task}_order{order:02d}"
            runs.append(Run(key=f"{task}__order{order}", task=task, folder=args.out / folder,
                            order=order, exp_name=exp_name))
    return runs


def command(run: Run, args, gpu_ids: str, eval_gpu: str, port: int) -> list[str]:
    cmd = [
        "accelerate", "launch", f"--gpu_ids={gpu_ids}", "--num_processes=1", f"--main_process_port={port}",
        str(TRAIN),
        f"task={run.task}",
        f"method={args.method}",
        f"task.task_order_index={run.order}",
        f"exp_name={run.exp_name}",
        f"group_tag={args.name}",
        f"output_dir_prefix={run.folder}",
        f"logging.mode={args.wandb}",
        f"logging.project={args.wandb_project}",
        f"training.separate_process_gpu={eval_gpu}",
    ]
    return cmd + buffer_overrides(args.buffer) + list(args.set)


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Experiment name; outputs go to outputs/<name>/.")
    p.add_argument("--method", required=True, choices=METHODS, help="Preset from memory_anchors/config/method/.")
    p.add_argument("--buffer", default=None,
                   help="ER buffer size: integer = #examples (e.g. 1000), float = proportion of past data (e.g. 0.01).")
    p.add_argument("--suites", default="goal,10,object,spatial", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="Comma-separated suites: goal, 10 (=long), object, spatial.")
    p.add_argument("--orders", default="0,1,2", type=parse_int_list, help="LIBERO task-order indices (paper: 0,1,2).")
    p.add_argument("--gpus", required=True, type=parse_int_list, help="GPU indices to schedule on, e.g. 0,1,2.")
    p.add_argument("--mem-per-run-mb", type=int, default=9000, help="Free GPU memory required to start a run.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra Hydra override passed to every run (repeatable).")
    p.add_argument("--output-root", default=str(REPO / "outputs"), help="Root folder for experiment outputs.")
    p.add_argument("--wandb", default="offline", choices=["online", "offline", "disabled"], help="Weights & Biases mode.")
    p.add_argument("--wandb-project", default="memory_anchors")
    p.add_argument("--egl-map", default=None, help='Explicit CUDA->EGL device map, e.g. "0:1,1:0" (default: egl_probe or identity).')
    p.add_argument("--retry-failed", action="store_true", help="Re-queue runs marked failed/terminated.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands and exit (state file untouched).")
    p.add_argument("--poll-sec", type=float, default=10.0)
    p.add_argument("--status-sec", type=float, default=300.0)
    args = p.parse_args()
    args.out = Path(args.output_root) / args.name
    return args


def main() -> int:
    args = parse_args()
    state_path = args.out / "launch_state.json"
    state = load_state(state_path)
    queue = []
    for run in build_runs(args):
        prev = state.get(run.key, {}).get("status")
        if prev == "completed" or (prev in {"failed", "terminated"} and not args.retry_failed):
            print(f"skip {run.key} (status={prev})")
            continue
        queue.append(run)

    if args.dry_run:
        for run in queue:
            print(f"# {run.key}\n" + subprocess.list2cmdline(command(run, args, str(args.gpus[0]), str(args.gpus[0]), 29500)))
        print(f"{len(queue)} run(s) would be launched; state file not modified.")
        return 0

    egl = cuda_to_egl_map(args.gpus, args.egl_map)
    active: dict[int, tuple[Run, subprocess.Popen, int, float]] = {}
    reserved: dict[int, int] = {}
    n_ok = n_fail = 0
    stop = False

    def on_signal(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[{now()}] signal {signum}: no new launches; terminating running jobs.", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    last_status = 0.0
    print(f"[{now()}] {len(queue)} run(s) queued on GPUs {args.gpus}", flush=True)

    while queue or active:
        for pid, (run, proc, gpu, _) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            reserved[gpu] -= args.mem_per_run_mb
            status = "completed" if code == 0 else "failed"
            n_ok += code == 0
            n_fail += code != 0
            state[run.key].update(status=status, return_code=code, finished_at=now())
            save_state(state_path, state)
            print(f"[{now()}] {run.key}: {status} (code {code})", flush=True)
            del active[pid]

        if stop:
            for pid, (run, proc, _, _) in active.items():
                proc.terminate()
                state[run.key].update(status="terminated", finished_at=now())
            save_state(state_path, state)
            for _, (_, proc, _, _) in active.items():
                proc.wait()
            break

        if queue:
            free = free_memory_mb()
            while queue:
                gpu = pick_gpu(args.gpus, free, reserved, args.mem_per_run_mb)
                if gpu is None:
                    break
                run = queue.pop(0)
                devices = visible_devices(gpu, egl[gpu])
                # Hydra needs a quoted string for a comma-separated value.
                eval_gpu = devices if "," not in devices else f"'{devices}'"
                cmd = command(run, args, devices, eval_gpu, free_port())
                env = os.environ.copy()
                env.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(egl[gpu]), HYDRA_FULL_ERROR="1",
                           NUMBA_DISABLE_JIT="1", TOKENIZERS_PARALLELISM="false")
                log_dir = args.out / "launch_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{run.exp_name}.log"
                with log_path.open("w") as log_f:
                    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
                reserved[gpu] = reserved.get(gpu, 0) + args.mem_per_run_mb
                active[proc.pid] = (run, proc, gpu, time.time())
                state[run.key] = dict(status="running", gpu=gpu, egl_device=egl[gpu], pid=proc.pid,
                                      log=str(log_path), started_at=now(), command=cmd)
                save_state(state_path, state)
                print(f"[{now()}] launched {run.key} on GPU {gpu} (log: {log_path})", flush=True)

        if time.time() - last_status > args.status_sec:
            running = ", ".join(f"{r.key}@gpu{g} {(time.time() - t) / 3600:.1f}h" for r, _, g, t in active.values())
            print(f"[{now()}] queued={len(queue)} running={len(active)} [{running}] done={n_ok} failed={n_fail}", flush=True)
            last_status = time.time()
        time.sleep(args.poll_sec)

    print(f"[{now()}] finished: completed={n_ok} failed={n_fail} remaining={len(queue)}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
