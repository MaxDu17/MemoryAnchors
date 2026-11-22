# Minimal installer for the vendored subset of diffusion_policy (see PATCHES.md).
from setuptools import find_packages, setup

setup(
    name="diffusion_policy",
    version="0.0.0+memoryanchors",
    description="Subset of diffusion_policy (UMI fork) used by MemoryAnchors; see PATCHES.md",
    packages=find_packages(include=["diffusion_policy", "diffusion_policy.*"]),
    python_requires=">=3.8",
)
