from setuptools import find_packages, setup

setup(
    name="memory_anchors",
    version="1.0.0",
    description="Memory Anchors for Continual Robot Learning (CoRL 2026): AnchorER and baselines on LIBERO",
    packages=find_packages(include=["memory_anchors", "memory_anchors.*"]),
    package_data={"memory_anchors": ["config/**/*.yaml", "config/*.yaml", "task_name_mappings/*.json"]},
    include_package_data=True,
    python_requires=">=3.10",
)
