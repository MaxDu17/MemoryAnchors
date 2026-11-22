"""Evaluate a single checkpoint in the LIBERO simulator.

Usage (normally launched for you by ``scripts/evaluate_runs.py``):

    MUJOCO_GL=egl python memory_anchors/evaluate.py \
        --checkpoint_path <run_dir>/checkpoints/<step>.ckpt \
        [--hydra-yaml <run_dir>/.hydra/config.yaml]

The run's saved Hydra config is reloaded and merged with ``config/eval/parallel_rollout.yaml``,
which switches the workspace to rollout mode. Results are written to ``<checkpoint>.json``.
"""
import argparse
import pathlib

import hydra
from hydra.core.utils import setup_globals
from omegaconf import OmegaConf

import memory_anchors  # noqa: F401  (registers OmegaConf resolvers)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors.utils.libero_task_order import register_libero_task_order_resolver

register_libero_task_order_resolver()
setup_globals()  # same resolvers as a normal Hydra run (${now:}, ${hydra:}, ...)

EVAL_OVERLAY = pathlib.Path(__file__).parent / "config" / "eval" / "parallel_rollout.yaml"


def build_eval_cfg(checkpoint_path: str, hydra_yaml: str = None) -> OmegaConf:
    ckpt = pathlib.Path(checkpoint_path).expanduser().resolve()
    assert ckpt.is_file(), f"Checkpoint does not exist: {ckpt}"
    if hydra_yaml is None:
        hydra_yaml = ckpt.parent.parent / ".hydra" / "config.yaml"
    hydra_yaml = pathlib.Path(hydra_yaml).expanduser().resolve()
    assert hydra_yaml.is_file(), f"Config yaml does not exist: {hydra_yaml}"
    cfg = OmegaConf.merge(OmegaConf.load(hydra_yaml), OmegaConf.load(EVAL_OVERLAY))
    OmegaConf.resolve(cfg)  # resolve now so all ${now:} resolvers agree
    cfg.rollout.checkpoint_path = str(ckpt)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--hydra-yaml", type=str, default=None,
        help="Run config (default: <run_dir>/.hydra/config.yaml next to the checkpoint).",
    )
    args = parser.parse_args()

    cfg = build_eval_cfg(args.checkpoint_path, args.hydra_yaml)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
