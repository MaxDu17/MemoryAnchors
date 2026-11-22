"""Filesystem locations used by MemoryAnchors.

Everything is resolved relative to the repository (not the current working
directory), so commands can be launched from anywhere. Override with:
  MEMORY_ANCHORS_DATA   root folder containing ``libero_abs/<suite>/*.hdf5`` (default: <repo>/data)
  MEMORY_ANCHORS_CACHE  folder for the zarr dataset cache (default: <repo>/cache)
"""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def repo_root() -> str:
    return str(REPO_ROOT)


def data_root() -> str:
    return os.environ.get("MEMORY_ANCHORS_DATA", str(REPO_ROOT / "data"))


def cache_root() -> str:
    return os.environ.get("MEMORY_ANCHORS_CACHE", str(REPO_ROOT / "cache"))


def libero_bddl_root() -> str:
    """BDDL folder of the vendored LIBERO copy (third_party/LIBERO)."""
    from libero.libero import get_libero_path

    return get_libero_path("bddl_files")


def evaluate_script() -> str:
    return str(PACKAGE_ROOT / "evaluate.py")
