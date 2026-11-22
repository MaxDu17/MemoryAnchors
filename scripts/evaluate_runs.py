#!/usr/bin/env python3
"""Evaluate every training checkpoint under a folder in the LIBERO simulator.

Training writes `checkpoints/<global_step>.ckpt` at every evaluation point (the untrained step-0
model and the end of every task). This script finds all of them, rolls each one out on all tasks
of its suite (50 benchmark init states per task) and writes `<global_step>.ckpt.json` next to it.
Checkpoints that already have a json are skipped, so it is safe to re-run (or to leave running
with --watch while training is still in progress).

    python scripts/evaluate_runs.py --root outputs/RandER_1000 --gpus 0,1 --watch

Per-checkpoint logs: <run_dir>/eval_logs/<global_step>.ckpt.eval.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gpu_utils import cuda_to_egl_map, free_memory_mb, parse_int_list, pick_gpu, visible_devices  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO / "memory_anchors" / "evaluate.py"


@dataclass
class Job:
    proc: subprocess.Popen
    ckpt: Path
    gpu: int
    log: Path
    started: float


def discover(root: Path, force: bool) -> list[Path]:
    """Numbered checkpoints (<global_step>.ckpt) without results, inside runs with a saved config."""
    pending = []
    for ckpt in sorted(root.rglob("checkpoints/*.ckpt")):
        if not ckpt.stem.isdigit():
            continue  # task-end checkpoints (task<i>_*.ckpt) are not evaluated
        if not (ckpt.parent.parent / ".hydra" / "config.yaml").is_file():
            continue
        if Path(str(ckpt) + ".json").exists() and not force:
            continue
        pending.append(ckpt.resolve())
    return pending


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="Folder to search (e.g. outputs/<name>).")
    p.add_argument("--gpus", required=True, type=parse_int_list, help="GPU indices, e.g. 0,1.")
    p.add_argument("--mem-per-eval-mb", type=int, default=7500, help="Free GPU memory required per evaluation.")
    p.add_argument("--max-parallel", type=int, default=4, help="Max concurrent evaluations (each uses ~10 CPU sim workers).")
    p.add_argument("--egl-map", default=None, help='Explicit CUDA->EGL map, e.g. "0:1,1:0".')
    p.add_argument("--force", action="store_true", help="Re-evaluate checkpoints that already have a json.")
    p.add_argument("--delete-checkpoints-after-success", action="store_true",
                   help="Delete <global_step>.ckpt once its json is written (saves ~0.7 GB per checkpoint).")
    p.add_argument("--watch", action="store_true", help="Keep polling for new checkpoints until Ctrl+C.")
    p.add_argument("--watch-sec", type=float, default=120.0)
    p.add_argument("--terminal", action="store_true", help="Stream eval output to the terminal instead of log files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    assert args.root.is_dir(), f"--root does not exist: {args.root}"
    egl = cuda_to_egl_map(args.gpus, args.egl_map)
    active: dict[int, Job] = {}
    reserved = {g: 0 for g in args.gpus}
    launched: set[Path] = set()
    n_ok = n_fail = 0
    try:
        while True:
            pending = [c for c in discover(args.root, args.force) if c not in launched]
            while pending or active:
                for pid, job in list(active.items()):
                    code = job.proc.poll()
                    if code is None:
                        continue
                    reserved[job.gpu] -= args.mem_per_eval_mb
                    result = Path(str(job.ckpt) + ".json")
                    if code == 0 and result.is_file():
                        n_ok += 1
                        print(f"[ok]   {job.ckpt} ({(time.time() - job.started) / 60:.0f} min)", flush=True)
                        if args.delete_checkpoints_after_success:
                            job.ckpt.unlink(missing_ok=True)
                    else:
                        n_fail += 1
                        print(f"[fail] {job.ckpt} (code {code}); see {job.log}", flush=True)
                    del active[pid]

                while pending and len(active) < args.max_parallel:
                    gpu = pick_gpu(args.gpus, free_memory_mb(), reserved, args.mem_per_eval_mb)
                    if gpu is None:
                        break
                    ckpt = pending.pop(0)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=visible_devices(gpu, egl[gpu]), MUJOCO_GL="egl",
                               MUJOCO_EGL_DEVICE_ID=str(egl[gpu]), TOKENIZERS_PARALLELISM="false",
                               NUMBA_DISABLE_JIT="1")
                    log = ckpt.parent.parent / "eval_logs" / f"{ckpt.name}.eval.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [sys.executable, str(EVAL_SCRIPT), f"--checkpoint_path={ckpt}"]
                    if args.terminal:
                        proc = subprocess.Popen(cmd, env=env, cwd=REPO)
                    else:
                        with log.open("a") as f:
                            proc = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
                    reserved[gpu] += args.mem_per_eval_mb
                    active[proc.pid] = Job(proc, ckpt, gpu, log, time.time())
                    launched.add(ckpt)
                    print(f"[run]  {ckpt} on GPU {gpu} ({len(pending)} pending)", flush=True)
                time.sleep(5)
            print(f"done: ok={n_ok} failed={n_fail}", flush=True)
            if not args.watch:
                return 0 if n_fail == 0 else 1
            time.sleep(args.watch_sec)
    except KeyboardInterrupt:
        for job in active.values():
            job.proc.terminate()
        print("\ninterrupted", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
