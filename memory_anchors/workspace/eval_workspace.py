"""Evaluate one training checkpoint in the LIBERO simulator.

Launched (one process per checkpoint) by ``memory_anchors/evaluate.py``. Writes the
rollout results next to the checkpoint as ``<checkpoint>.json``.
"""
import copy
import logging
import numbers
import os
import pathlib
import random
import time

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.utils.load_env import dynamically_loaded_env_rollout

OmegaConf.register_new_resolver("eval", eval, replace=True)

# robosuite writes to /tmp/robosuite.log; on shared machines that file may be owned by
# another user. Redirect it to ~/.robosuite/robosuite.log in that case.
_DEFAULT_ROBOSUITE_LOG = "/tmp/robosuite.log"
try:
    with open(_DEFAULT_ROBOSUITE_LOG, "a"):
        pass
except (PermissionError, OSError):
    _user_log_dir = os.path.expanduser("~/.robosuite")
    os.makedirs(_user_log_dir, exist_ok=True)
    _user_log_path = os.path.join(_user_log_dir, "robosuite.log")
    _original_file_handler_init = logging.FileHandler.__init__

    def _patched_file_handler_init(self, filename, *args, **kwargs):
        if filename == _DEFAULT_ROBOSUITE_LOG:
            filename = _user_log_path
        return _original_file_handler_init(self, filename, *args, **kwargs)

    logging.FileHandler.__init__ = _patched_file_handler_init


def _log(msg):
    print(f"\033[94m[Eval workspace] {msg}\033[0m", flush=True)


class ParallelizedRolloutPolicyWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        ckpt_path = pathlib.Path(cfg.rollout.checkpoint_path).resolve()
        # <run_dir>/checkpoints/<step>.ckpt -> <run_dir>
        assert ckpt_path.parent.name == "checkpoints", (
            f"Expected checkpoint inside a run's checkpoints/ folder, got {ckpt_path}"
        )
        self._output_dir = str(ckpt_path.parent.parent)
        super().__init__(cfg, output_dir=self._output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.checkpoint_name = ckpt_path.stem
        self.model: BasePolicy = hydra.utils.instantiate(cfg.model)
        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

    def run(self):
        beg = time.time()
        cfg = copy.deepcopy(self.cfg)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        accelerator = Accelerator(log_with=None, mixed_precision=cfg.training.mixed_precision)

        _log(f"Evaluating checkpoint {cfg.rollout.checkpoint_path} (run dir: {self.output_dir})")
        saved_output_dir = self.output_dir
        self.load_checkpoint(path=cfg.rollout.checkpoint_path)
        self._output_dir = saved_output_dir

        policy = self.ema_model if cfg.training.use_ema else accelerator.unwrap_model(self.model)
        policy.eval()
        policy.to(accelerator.device)

        checkpoint_step = None
        try:
            checkpoint_step = int(pathlib.Path(cfg.rollout.checkpoint_path).stem)
        except (ValueError, TypeError):
            _log("Checkpoint name is not a global step; using default labels.")

        step_log = dynamically_loaded_env_rollout(
            cfg, policy, output_dir=self.output_dir, video_format="path", checkpoint_step=checkpoint_step,
        )

        # Keep numbers/strings plus dicts (multi_bddl_success_* maps bddl path -> bool).
        def _rollout_filter(k, v):
            return isinstance(v, (numbers.Number, str, dict))

        json_path = cfg.rollout.checkpoint_path + ".json"
        if os.path.exists(json_path):
            os.remove(json_path)  # JsonLogger appends; start fresh so the file holds one record
        with JsonLogger(json_path, filter_fn=_rollout_filter) as json_logger:
            json_logger.log(step_log)

        accelerator.end_training()
        _log(f"Wrote {json_path} in {time.time() - beg:.0f}s")
