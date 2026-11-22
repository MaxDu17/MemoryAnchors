#!/usr/bin/env python3
"""Add absolute end-effector actions to the LIBERO datasets.

LIBERO stores *delta* end-effector actions. All MemoryAnchors policies (and every result in the
paper) are trained on and output ABSOLUTE end-effector poses instead. This script replays every
demonstration in the simulator, reads the OSC controller's goal pose for each delta action and
stores it as `data/demo_<i>/abs_actions` (position (3), axis-angle rotation (3), gripper (1)).
The original `actions` key is left untouched. Each hdf5 is copied, not modified in place.

    python scripts/convert_to_abs_actions.py                  # data/libero/<suite> -> data/libero_abs/<suite>
    python scripts/convert_to_abs_actions.py --suites goal --num-workers 16

Takes on the order of an hour per suite (CPU only, parallel over demonstrations).
"""
import argparse
import multiprocessing
import os
import shutil
from pathlib import Path

import h5py
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
SUITES = {"goal": "libero_goal", "10": "libero_10", "long": "libero_10", "object": "libero_object", "spatial": "libero_spatial"}


def _worker(job):
    path, idx = job
    from memory_anchors.common.robomimic_util import RobomimicAbsoluteActionConverter

    return RobomimicAbsoluteActionConverter(path).convert_idx(idx)


def convert_file(src: Path, dst: Path, num_workers: int):
    from memory_anchors.common.robomimic_util import RobomimicAbsoluteActionConverter

    n_demos = len(RobomimicAbsoluteActionConverter(str(src)))
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(_worker, [(str(src), i) for i in range(n_demos)]),
                            total=n_demos, desc=src.name, leave=False))
    tmp = dst.with_suffix(".tmp")
    shutil.copyfile(src, tmp)
    with h5py.File(tmp, "r+") as f:
        for i, abs_actions in enumerate(results):
            f[f"data/demo_{i}"].create_dataset("abs_actions", data=abs_actions, dtype=abs_actions.dtype)
    tmp.replace(dst)


def main():
    data = Path(os.environ.get("MEMORY_ANCHORS_DATA", REPO / "data"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="goal,10,object,spatial")
    p.add_argument("--src", default=str(data / "libero"), help="Folder with the downloaded <suite>/ folders.")
    p.add_argument("--dst", default=str(data / "libero_abs"), help="Output folder (training reads <dst>/<suite>/).")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    for alias in [s.strip() for s in args.suites.split(",") if s.strip()]:
        suite = SUITES.get(alias, alias)
        files = sorted(Path(args.src, suite).glob("*.hdf5"))
        assert files, f"No hdf5 files in {Path(args.src, suite)}; run scripts/download_libero.py first."
        out_dir = Path(args.dst, suite)
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / src.name
            if dst.exists():
                print(f"skip (exists): {dst}")
                continue
            print(f"converting {src} -> {dst}")
            convert_file(src, dst, args.num_workers)
    print("Done.")


if __name__ == "__main__":
    main()
