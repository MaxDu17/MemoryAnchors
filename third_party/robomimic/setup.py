# Minimal installer for the vendored robomimic 0.2.0 (see PATCHES.md).
# Dependencies are managed by the top-level environment.yml (egl_probe is NOT required).
from setuptools import find_packages, setup

setup(
    name="robomimic",
    version="0.2.0+memoryanchors",
    description="robomimic 0.2.0 (vendored copy for MemoryAnchors; see PATCHES.md)",
    packages=find_packages(include=["robomimic", "robomimic.*"]),
    python_requires=">=3.8",
)
