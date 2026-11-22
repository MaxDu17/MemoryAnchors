"""Train a diffusion policy on a LIBERO suite, one task after another, with experience replay.

    python memory_anchors/train.py task=libero_goal method=anchor_removal task.task_order_index=0

Config: memory_anchors/config/train.yaml (+ task/, model/, learning_algorithm/, method/ groups).
Usually launched through scripts/launch.py, which runs a grid of suites x task orders on local GPUs.
"""
import pathlib

import hydra
from omegaconf import OmegaConf

import memory_anchors  # noqa: F401  (registers OmegaConf resolvers)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors.utils.libero_task_order import register_libero_task_order_resolver

register_libero_task_order_resolver()


@hydra.main(
    version_base="1.2",
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
    config_name="train",
)
def main(cfg: OmegaConf):
    # resolve immediately so all ${now:} resolvers use the same time
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
