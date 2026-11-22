# Minimal installer for the vendored, trimmed copy of LIBERO (environment side only).
# Install with:  pip install --config-settings editable_mode=compat -e third_party/LIBERO
from setuptools import find_packages, setup

setup(
    name="libero",
    version="0.1.0+memoryanchors",
    description="LIBERO benchmark environments (vendored copy for MemoryAnchors; see PATCHES.md)",
    packages=find_packages(include=["libero", "libero.*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
