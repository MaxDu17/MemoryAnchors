"""
Usage:
Training:
python train.py --config-name=train_policy_workspace
"""

import memory_anchors  # noqa: F401  (registers OmegaConf resolvers)

import hydra
from omegaconf import OmegaConf
import pathlib

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors.utils.libero_task_order import (
    register_libero_task_order_resolver,
)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
register_libero_task_order_resolver()

@hydra.main(
    version_base='1.2',
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    print("[Train.py] Starting!")
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print("[Train.py] Class created! Ready to start workspace!")
    workspace.run()

if __name__ == "__main__":
    main()
