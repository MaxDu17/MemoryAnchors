#!/usr/bin/env python3
"""Download the original LIBERO demonstration datasets from Hugging Face.

    python scripts/download_libero.py                       # all four suites -> data/libero/<suite>/
    python scripts/download_libero.py --suites goal,10      # a subset

Files are fetched from the `yifengzhu-hf/LIBERO-datasets` dataset repo (the official LIBERO
release, ~49 GB for the four suites). Afterwards run scripts/convert_to_abs_actions.py.
"""
import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITES = {"goal": "libero_goal", "10": "libero_10", "long": "libero_10", "object": "libero_object", "spatial": "libero_spatial"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="goal,10,object,spatial")
    p.add_argument("--out", default=os.path.join(os.environ.get("MEMORY_ANCHORS_DATA", REPO / "data"), "libero"))
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    suites = [SUITES.get(s.strip(), s.strip()) for s in args.suites.split(",") if s.strip()]
    for suite in suites:
        print(f"Downloading {suite} -> {args.out}/{suite}")
        snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=args.out, allow_patterns=f"{suite}/*")
        n = len(list(Path(args.out, suite).glob("*.hdf5")))
        assert n == 10, f"expected 10 hdf5 files for {suite}, found {n}"
    print("Done. Next: python scripts/convert_to_abs_actions.py")


if __name__ == "__main__":
    main()
