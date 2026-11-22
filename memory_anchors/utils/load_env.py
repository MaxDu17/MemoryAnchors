import glob
import os

import hydra
import numpy as np
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


def _task_hdf5_files(cfg):
    """All task hdf5 files of the suite, in a deterministic (sorted) order.

    Every checkpoint is evaluated on *all* tasks of the suite (seen and unseen);
    the metrics code later uses task_boundaries.json to decide what was seen.
    """
    hdf5_files = sorted(glob.glob(os.path.join(cfg.task.dataset.dataset_path, "*.hdf5")))
    assert len(hdf5_files) > 0, f"No hdf5 files found in {cfg.task.dataset.dataset_path}"
    if cfg.task.env_runner.include_file_filters:
        hdf5_files = [
            x for x in hdf5_files
            if os.path.basename(x).replace(".hdf5", "") in cfg.task.env_runner.include_file_filters
        ]
    return hdf5_files


def _add_mean_scores(cfg, step_log):
    if cfg.checkpoint.topk.monitor_key == "test_mean_score":
        assert "test_mean_score" not in step_log
        test_scores = [v for k, v in step_log.items() if "test/" in k and "_mean_score" in k]
        # NOTE: averages over all suite tasks, including ones not trained yet.
        step_log["test_mean_score"] = np.mean(test_scores)
        train_scores = [v for k, v in step_log.items() if "train/" in k and "_mean_score" in k]
        step_log["train_mean_score"] = np.mean(train_scores) if train_scores else float("nan")
    return step_log


def load_env_runner(cfg, output_dir):
    """Instantiate one env runner per task (memory heavy; used for in-process eval)."""
    assert "libero" in cfg.task.name, "Only LIBERO tasks are supported."
    env_runners = []
    for file in _task_hdf5_files(cfg):
        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, task_dir=file, output_dir=output_dir
        )
        assert isinstance(env_runner, BaseImageRunner)
        env_runners.append(env_runner)
    return env_runners


def dynamically_loaded_env_rollout(cfg, policy, output_dir, video_format="wandb", checkpoint_step=None, cl_policy=None):
    """Evaluate a policy task-by-task, creating each task's envs only when needed (low RAM)."""
    assert "libero" in cfg.task.name, "Only LIBERO tasks are supported."
    step_log = {}
    for file in _task_hdf5_files(cfg):
        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, task_dir=file, output_dir=output_dir
        )
        assert isinstance(env_runner, BaseImageRunner)
        runner_log = env_runner.run(
            policy,
            video_format=video_format,
            close_afterward=True,
            checkpoint_step=checkpoint_step,
        )
        step_log.update(runner_log)
    return _add_mean_scores(cfg, step_log)


def env_rollout(cfg, env_runners, policy, video_format="wandb", save_sim_states=False, checkpoint_step=None, cl_policy=None):
    """Evaluate with pre-instantiated env runners (see load_env_runner)."""
    assert "libero" in cfg.task.name, "Only LIBERO tasks are supported."
    step_log = {}
    for env_runner in env_runners:
        runner_log = env_runner.run(
            policy,
            video_format=video_format,
            save_sim_states=save_sim_states,
            checkpoint_step=checkpoint_step,
        )
        step_log.update(runner_log)
    return _add_mean_scores(cfg, step_log)
