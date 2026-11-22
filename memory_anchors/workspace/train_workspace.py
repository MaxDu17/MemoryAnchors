import os
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
import copy
import random
import pickle
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from memory_anchors.utils.load_env import load_env_runner, env_rollout
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import accelerate
from accelerate import Accelerator
from accelerate.utils import send_to_device
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from memory_anchors.utils.training_utils import get_gradient_norm
from memory_anchors.dataset.base_dataset import BaseDataset
from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.utils.prompt_util import collate_prompts
import time 
import wandb 
import threading
import shutil

import subprocess
import sys
import pathlib
from multiprocessing import Queue
import json
import re

from memory_anchors import paths

OmegaConf.register_new_resolver("eval", eval, replace=True)

import logging


def _count_module_parameters(module: torch.nn.Module) -> tuple:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def _append_task_boundary_record(output_dir: str, task_name: str, end_step: int) -> None:
    """
    Append one task record to task_boundaries.json in the run directory.
    Uses a temp file + replace so partial writes from crashes do not corrupt the file.
    """
    path = os.path.join(output_dir, "task_boundaries.json")

    if os.path.isfile(path):
        with open(path, "r") as f:
            records = json.load(f)
        assert isinstance(records, list)
    else:
        records = []
    records.append({"task_name": task_name, "end_step": int(end_step)})
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _sanitize_task_name_for_ckpt(task_name: str) -> str:
    """
    Convert a task/dataset name into a filesystem-safe string suitable for checkpoint filenames.
    """
    s = str(task_name).strip()
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s


def _within_task_rollout_epochs(epochs_per_task: int, evals_per_task: int) -> frozenset[int]:
    """
    End-of-epoch indices t within a task (0 .. epochs_per_task-1) at which to run rollout.
    Exactly evals_per_task evals per task, including the last epoch of the task.
    """
    assert evals_per_task >= 1
    # assert epochs_per_task >= evals_per_task
    out: set[int] = set()
    for k in range(evals_per_task):
        t = (k + 1) * epochs_per_task // evals_per_task - 1
        t = min(max(t, 0), epochs_per_task - 1)
        out.add(t)
    return frozenset(out)


def _rollout_schedule_active(cfg) -> bool:
    """Whether any env rollout / separate-process eval is scheduled for this run."""
    if bool(cfg.task.sequential):
        return int(cfg.task.evals_per_task) >= 1
    return cfg.training.rollout_every != -1


def _resolve_eval_policies(model, ema_model, cfg, accelerator):
    """
    Returns (inference_policy, cl_policy).

    inference_policy exposes BasePolicy APIs (scalar loss, predict_action, reset).
    cl_policy is the CL wrapper (e.g. PackNet) used for mask application during rollouts.
    """
    cl_policy = accelerator.unwrap_model(model)
    if cfg.training.use_ema and ema_model is not None:
        return ema_model, cl_policy
    if hasattr(cl_policy, "base_model"):
        return cl_policy.base_model, cl_policy
    return cl_policy, cl_policy


def _should_run_rollout_at_epoch_end(cfg, epoch: int) -> bool:
    """
    Rollout at end of this global epoch (after training), before epoch counter increments.
    Non-sequential: same as legacy epoch % rollout_every with epoch > 0.
    Sequential: epoch % epochs_per_task in the per-task schedule from evals_per_task.
    """
    if not cfg.training.rollout_enabled:
        return False
    if bool(cfg.task.sequential):
        ept = int(cfg.task.epochs_per_task)
        n_eval = int(cfg.task.evals_per_task)
        within = epoch % ept
        return within in _within_task_rollout_epochs(ept, n_eval)
    if cfg.training.rollout_every == -1:
        return False
    return epoch > 0 and (epoch % cfg.training.rollout_every) == 0


# cursed fix to robotsuite setup 
# Fix robosuite log file permission issue
default_log_path = "/tmp/robosuite.log"
try:
    if os.path.exists(default_log_path):
        with open(default_log_path, "a") as f:
            pass
    else:
        with open(default_log_path, "w") as f:
            pass
        os.chmod(default_log_path, 0o666)
except (PermissionError, OSError) as e:
    user_log_dir = os.path.expanduser("~/.robosuite")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_path = os.path.join(user_log_dir, "robosuite.log")
    
    original_file_handler_init = logging.FileHandler.__init__
    
    def patched_file_handler_init(self, filename, *args, **kwargs):
        if filename == default_log_path:
            filename = user_log_path
        return original_file_handler_init(self, filename, *args, **kwargs)
    
    logging.FileHandler.__init__ = patched_file_handler_init


class EvaluationQueue:
    def __init__(self, accelerator, cfg, output_dir, top_k_manager = None, eval_program_path = None):
        self.top_k_manager = top_k_manager 
        self.eval_program_path = eval_program_path 
        self.accelerator = accelerator 
        self.cfg = cfg 
        self.output_dir = output_dir
        self.process_limit = 1
        self._eval_slots = threading.Semaphore(self.process_limit)

        self.log_queue = Queue()  # Add this
        self.eval_threads = []
        self.thread_failures = []


        if eval_program_path is None:
            self.eval_program_path = paths.evaluate_script()
    
    def process_eval_results(self, returncode, checkpoint, train_step, epoch, expected_json_file):
        if returncode != 0:
            self.thread_failures.append(
                f"epoch={epoch}, step={train_step}, checkpoint={checkpoint}, "
                f"error=eval subprocess returncode={returncode}"
            )
            return 
        print(f"\t [Workspace - process eval results] Reading from {expected_json_file}")
        try:
            with open(expected_json_file, "r") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            log = json.loads(lines[-1])  # JSON-lines file; the last record is the latest eval
        except:
            self.thread_failures.append(
                f"epoch={epoch}, step={train_step}, checkpoint={checkpoint}, "
                f"error=unable to read eval json {expected_json_file}"
            )
            print("\t [Workspace - process eval results] Unable to load the step log file!")
            return 
       
        # replace all video files with the wandb video version 
        for k, v in log.items():
            if isinstance(v, str) and ".mp4" in v: # hacky but catches all videos
                print(k)
                log[k] = wandb.Video(v)
        
        # tagging the log with the relevant localization values so we can graph it 
        log.update({"epoch" : epoch, "global_step" : train_step, "Steps" : train_step}) # this keeps 

        if self.top_k_manager is not None:
            try:
                topk_ckpt_path = self.top_k_manager.get_ckpt_path(log)
                # copy the current checkpoint to the topk ckpt path
                if topk_ckpt_path is not None:
                    print("[Workspace - process eval results] Saving checkpoint!")
                    shutil.copyfile(checkpoint, topk_ckpt_path) # this one might take a little time because it's 2gb 
            except:
                print("[Workspace] Topk reading failed! (this might be expected for the first run)")
        self.log_queue.put(log) 

        if os.path.exists(checkpoint):
            os.remove(checkpoint)

    def check_and_log_results(self):
        """Call this from main thread to process queued logs"""
        while not self.log_queue.empty():
            log = self.log_queue.get_nowait()
            epoch = log["epoch"]
            step = log["Steps"]
            print(f"[Workspace Main Process] Found log queue element, logging at step {step}, epoch {epoch}")
            self.accelerator.log(log) 

    def run_process(self, cmd, checkpoint, train_step, epoch, env, expected_json_file, output_mode, output_log_path):
        try:
            if output_mode == "file":
                assert output_log_path is not None, "output_log_path must be set when output_mode=file."
                with open(output_log_path, "a", buffering=1) as log_f:
                    eval_proc = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )
                    eval_proc.wait()
            else:
                eval_proc = subprocess.Popen(cmd, env=env)
                eval_proc.wait()  # This blocks the thread, not the main process
            print("\t [Workspace Main Process, Separate Thread] Evaluation Process Done")
            self.process_eval_results(eval_proc.returncode, checkpoint, train_step, epoch, expected_json_file)  # Execute callback when done
        except Exception as e:
            self.thread_failures.append(
                f"epoch={epoch}, step={train_step}, checkpoint={checkpoint}, error={repr(e)}"
            )
        finally:
            self._eval_slots.release()
        

    def set_off_run(self, checkpoint, train_step, epoch):
        self._eval_slots.acquire()
        expected_json_file = checkpoint + ".json"
        hydra_yaml = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), ".hydra", "config.yaml")
        assert os.path.isfile(hydra_yaml), f"Hydra YAML config does not exist: {hydra_yaml}"
        
        cmd = [
            sys.executable,  # python executable
            str(self.eval_program_path),
            f"--checkpoint_path={checkpoint}",
            f"--hydra-yaml={hydra_yaml}"# this is the actual checkpoint path 
        ]
        print(f"[Workspace Main Process] Epoch {epoch} Running command: ")
        print(cmd)
        print("--------------------------------")
        # Launch the process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.training.separate_process_gpu)
        env["MUJOCO_GL"] = "egl" # "osmesa"
        output_mode = str(self.cfg.training.separate_process_eval_output).strip().lower()
        assert output_mode in {"terminal", "file"}, (
            "training.separate_process_eval_output must be one of ['terminal', 'file']."
        )
        output_log_path = None
        if output_mode == "file":
            run_dir = os.path.dirname(os.path.dirname(checkpoint))
            eval_log_dir = os.path.join(run_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)
            output_log_path = os.path.join(
                eval_log_dir,
                f"eval_epoch={int(epoch):04d}_step={int(train_step):08d}.log",
            )
            print(f"[Workspace Main Process] Redirecting eval subprocess output to {output_log_path}")

        thread = threading.Thread(
            target=self.run_process,
            args=(
                cmd,
                checkpoint,
                train_step,
                epoch,
                env,
                expected_json_file,
                output_mode,
                output_log_path,
            ),
            name=f"eval-epoch{epoch}-step{train_step}",
        )
        thread.start()
        self.eval_threads.append(thread)

    def write_thread_failure_log(self):
        failure_log_path = os.path.join(self.output_dir, "eval_thread_failures.log")
        with open(failure_log_path, "w") as f:
            if len(self.thread_failures) == 0:
                f.write("No eval thread failures detected.\n")
                return
            f.write("Evaluation thread failures detected:\n")
            for idx, failure in enumerate(self.thread_failures):
                f.write(f"{idx + 1}. {failure}\n")

    def wait_for_all(self):
        """
        Wait for all launched evaluation threads to finish, then flush logs.
        Intended to be called once at the end of training on main process.
        """
        for t in self.eval_threads:
            t.join()

        alive_threads = [t.name for t in self.eval_threads if t.is_alive()]
        assert len(alive_threads) == 0, f"Evaluation threads still alive after join: {alive_threads}"
        if len(self.thread_failures) > 0:
            print("Evaluation thread failures detected:\n" + "\n".join(self.thread_failures))

        # final drain so the last finished eval logs before shutdown
        self.check_and_log_results()
        self.write_thread_failure_log()


def _write_done_file(output_dir: str) -> None:
    done_path = os.path.join(output_dir, "done.txt")
    with open(done_path, "w") as f:
        f.write("run_completed=true\n")
        f.write(f"timestamp_unix={time.time():.3f}\n")


class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', "training_split_info"]
    exclude_keys = ["_output_dir"] # I want to use a new directory every time I run # tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.learning_algorithm)

        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model.base_model)

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']


 # compute action MSE error on both training and validation datasets (and for each task if multi-task training)
    def compute_action_mse(self, category, task, pred_action, gt_action, action_components):
        log = {}
        
        # Overall MSE
        log[f'{category}/action_mse_error/{task}'] = torch.nn.functional.mse_loss(pred_action, gt_action).item()

        # Per-component MSE from shape_meta (works for single-arm and bimanual).
        current_action_dim = 0
        for component in action_components:
            component_dim = component.size
            component_name = component.type
            log[f'{category}/action_mse_error_{component_name}/{task}'] = torch.nn.functional.mse_loss(
                pred_action[..., current_action_dim:current_action_dim + component_dim],
                gt_action[..., current_action_dim:current_action_dim + component_dim],
            ).item()
            current_action_dim += component_dim
        assert current_action_dim == pred_action.shape[-1], (
            f"action_components sum to dim {current_action_dim} but pred_action has dim {pred_action.shape[-1]}"
        )
        
        return log

    def compute_action_mse_on_dataset(self, split_name: str, dataset: BaseDataset, dataloader_cfg, accelerator, policy, cfg, step_log):
        """Handles logging per-task MSE if doing multi-task training. Also handles only computing loss on non-prompt actions if prompting policy."""
        # Do not accelerator.prepare these loaders: under multi-GPU that can yield None
        # batches when only the main process iterates. Match the bimanual reference path.
        # Convert to a plain dict so we can add keys (e.g. drop_last) even when the Hydra
        # config is in struct mode and the key was never declared in the YAML.
        dataloader_cfg = OmegaConf.to_container(dataloader_cfg, resolve=True)
        dataloader_cfg['shuffle'] = False
        dataloader_cfg['persistent_workers'] = False
        dataloader_cfg['drop_last'] = False

        action_components = cfg.task.shape_meta.action.components

        if dataset.is_multi_task():
            unique_task_name_to_dataset_indices = dataset.get_unique_task_name_to_dataset_indices()
        else:
            unique_task_name_to_dataset_indices = {}
        unique_task_name_to_dataset_indices['all'] = list(range(len(dataset)))
        for unique_task_name, dataset_indices in unique_task_name_to_dataset_indices.items():
            if len(dataset_indices) == 0:
                continue
            subset = Subset(dataset, dataset_indices)
            subset_dataloader = accelerator.prepare(DataLoader(subset, **dataloader_cfg, collate_fn=collate_prompts))
            # subset_dataloader = DataLoader(subset, **dataloader_cfg, collate_fn=collate_prompts)

            num_task_observed = 0
            pred_actions = []
            gt_actions = []
            for batch in subset_dataloader:
                assert batch is not None and batch.get('action') is not None, (
                    f"Got invalid batch while computing action MSE on split={split_name} "
                    f"task={unique_task_name}"
                )
                batch = send_to_device(batch, device=policy.device)
                B = batch['action'].shape[0]
                num_task_observed += B

                pred_action = policy.predict_action_training(batch['obs'])['action_pred'].cpu() # (B, T [optional; only if sequence prompting], num_pred_steps, action_dim)
                gt_action = batch['action'].cpu() # (B, T [optional; only if sequence prompting], num_pred_steps, action_dim)

                if 'action_mask' in batch.get('metadata', {}):
                    # action_mask is (B,) and contains the indices of the first valid action for each batch entry
                    for batch_i in range(B):
                        first_valid_action_index = batch['metadata']['action_mask'][batch_i]
                        pred_actions.append(pred_action[batch_i, first_valid_action_index:])
                        gt_actions.append(gt_action[batch_i, first_valid_action_index:])
                else:
                    pred_actions.append(pred_action)
                    gt_actions.append(gt_action)

                if num_task_observed >= cfg.training.sample_min_size:
                    break

            assert num_task_observed >= cfg.training.sample_min_size, f'for task {unique_task_name}, num_task_observed: {num_task_observed}, cfg.training.sample_min_size: {cfg.training.sample_min_size} on split {split_name}'

            pred_actions = torch.cat(pred_actions, dim=0)
            gt_actions = torch.cat(gt_actions, dim=0)
            
            # now average over the entire batch
            mse_log = self.compute_action_mse(
                split_name,
                unique_task_name.replace(' ', '_'),
                pred_actions,
                gt_actions,
                action_components,
            )

            step_log.update(mse_log)

    def configure_checkpoint_settings(self, val_dataloader, cfg):
        # configure checkpoint and make sure the monitor_key is valid
        old_monitor_key = cfg.checkpoint.topk.monitor_key
        old_mode = cfg.checkpoint.topk.mode
        if cfg.checkpoint.topk.monitor_key == 'train_mean_score' or cfg.checkpoint.topk.monitor_key == 'test_mean_score':
            if bool(cfg.task.sequential) and int(cfg.task.evals_per_task) >= 1:
                checkpoint_every = max(1, int(cfg.task.epochs_per_task) // int(cfg.task.evals_per_task))
            elif cfg.training.rollout_every == -1:
                # invalid, try action MSE error
                split = 'val' if len(val_dataloader) > 0 else 'train'
                cfg.checkpoint.topk.monitor_key = f'{split}_action_mse_error_all'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use rollout every
                checkpoint_every = cfg.training.rollout_every
        if 'action_mse_error' in cfg.checkpoint.topk.monitor_key:
            if cfg.training.sample_every == -1:
                # invalid, try val loss
                cfg.checkpoint.topk.monitor_key = 'val_loss'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use sample every
                checkpoint_every = cfg.training.sample_every
        if cfg.checkpoint.topk.monitor_key == 'val_loss':
            if cfg.training.val_every == -1:
                # invalid, try train loss
                cfg.checkpoint.topk.monitor_key = 'train_loss'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use val every
                checkpoint_every = cfg.training.val_every
        if cfg.checkpoint.topk.monitor_key == 'train_loss':
            # train_loss is always a valid metric to checkpoint on
            checkpoint_every = 1

        if cfg.checkpoint.topk.monitor_key != old_monitor_key:
            print(f'NOTE: updating checkpoint monitor key from `{old_monitor_key}` ({old_mode}) to `{cfg.checkpoint.topk.monitor_key}` ({cfg.checkpoint.topk.mode})')
            cfg.checkpoint.topk.format_str = cfg.checkpoint.topk.format_str.replace(old_monitor_key, cfg.checkpoint.topk.monitor_key)
        return checkpoint_every

    def run(self):
        start_time = time.time()
        cfg = copy.deepcopy(self.cfg) 
        print("[Workspace] Started run")
        if cfg.training.debug: # settings for debug development 
            cfg.training.num_epochs = 10 # only train for 2 epochs 
            cfg.training.max_train_steps = 3 # only train for 3 steps per epoch 
            cfg.training.max_val_steps = 3 # only use 3 steps for validation 
            cfg.training.rollout_every = 10 # rollout every epoch 
            cfg.training.val_every = 1 # valid every epoch 
            cfg.training.sample_every = 1

            # uncomment to test task switching 
            cfg.task.epochs_per_task = 2 # should progress through a few tasks (not valid for everything)

            # this makes the rollouts short 
            # cfg.training.rollout_enabled=True # if you just wnat to test training 
            cfg.training.rollout_enabled=False # if you just wnat to test training 
            # cfg.task.env_runner.n_test = 1 # only one rollout 
            # cfg.task.env_runner.n_test_vis = 1 
            # cfg.task.env_runner.n_train = 1 
            # cfg.task.env_runner.n_train_vis = 1 


        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes
        print("[Workspace] Starting to Initialize Accelerator ", round(time.time() - start_time, 2))

        ### INITIALIZING WANDB AND ACCELERATORS AND LOGGING ### 
        timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=120)) # two hour timeout on multi GPU NCCL (default is 30 min). This helps with long setup before training where processes have to wait for env_runners to initialize
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[timeout], mixed_precision=cfg.training.mixed_precision)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        # ensure all processes use the same output directory
        output_dirs = accelerate.utils.gather_object([self.output_dir] if accelerator.is_main_process else [''])
        main_output_dir = [x for x in output_dirs if x][0]
        self._output_dir = main_output_dir
        accelerator.print(f'[Workspace] Started training. Run dir: {self.output_dir}. {round(time.time() - start_time, 2)}')

        self.model.register_accelerator(accelerator) # this is so we have access to the accelerator object 
        
        ### INITIALIZING ROLLOUT DEPENDENCIES ### 
        accelerator.print(f'[Workspace] Loading env runner. {round(time.time() - start_time, 2)}')
        # configure env runners (rollout) if necessary 
        if cfg.training.rollout_enabled and not cfg.training.separate_process_eval and _rollout_schedule_active(cfg) and accelerator.is_main_process:
            env_runners = load_env_runner(cfg, self.output_dir)

        # configure optimizer
        for key, value in cfg.optimizer.items():
            if key == 'lr' or key == 'obs_encoder_lr':
                cfg.optimizer[key] *= accelerator.num_processes # scale LR by num GPUs; see see https://huggingface.co/docs/accelerate/concept_guides/performance

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.num_grad_steps = 0
        self.epoch = 0
        self.training_split_info = None 


        self.event_table = wandb.Table(columns=["step", "event"])
        accelerator.print(f'[Workspace] Loading Checkpoints. {round(time.time() - start_time, 2)}')

        #### LOADING CHECKPOINTS ### 
        # Task-boundary resume: load weights/epoch/split from an explicit ckpt (preferred)
        # or from checkpoints/latest.ckpt when training.resume=true.
        resume_checkpoint_path = cfg.training.get("resume_checkpoint_path", None)
        is_resume = False
        if resume_checkpoint_path is not None:
            ckpt_path = pathlib.Path(resume_checkpoint_path)
            assert ckpt_path.is_file(), f"[Workspace] resume_checkpoint_path not found: {ckpt_path}"
            accelerator.print(f"[Workspace] Resuming from checkpoint {ckpt_path}")
            # Keep the current Hydra run dir; do not restore the original run's _output_dir.
            self.load_checkpoint(path=ckpt_path, exclude_keys=["_output_dir"])
            is_resume = True
        elif cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            assert latest_ckpt_path.is_file(), (
                f"[Workspace] training.resume=true but latest.ckpt missing: {latest_ckpt_path}. "
                f"Set training.resume_checkpoint_path to a task-end checkpoint instead."
            )
            accelerator.print(f"[Workspace] Resuming from checkpoint {latest_ckpt_path}")
            self.load_checkpoint(path=latest_ckpt_path, exclude_keys=["_output_dir"])
            is_resume = True

        if is_resume and cfg.task.get("sequential", False):
            epochs_per_task = int(cfg.task.epochs_per_task)
            assert self.epoch % epochs_per_task == 0, (
                f"[Workspace] Resume only supported at task boundaries "
                f"(epoch={self.epoch} is not divisible by epochs_per_task={epochs_per_task}). "
                f"Use a task-end checkpoint (e.g. task1_*.ckpt)."
            )
        
        if cfg.training.pretrained_checkpoint_path is not None: # don't assume we're using the same dataset 
            print("[Workspace] Loading from checkpoint") 
            # but still train on a new checkpoint 
            # self.load_checkpoint(path=cfg.training.pretrained_checkpoint_path, exclude_keys = ["_output_dir", "epoch", "training_split_info", "global_step"] )
            self.load_checkpoint(path=cfg.training.pretrained_checkpoint_path, exclude_keys = ["_output_dir", "epoch", "global_step"] )

        ### CONFIGURING DATA ### 
        # configure train and validation datasets
        dataset_generator = hydra.utils.instantiate(cfg.task.dataset)
        if cfg.training.pretrained_checkpoint_path is not None or is_resume: # use loaded split from the checkpoint 
            dataset_generator.load_custom_training_split_info(self.training_split_info)

        if dataset_generator.is_sequential: 
            accelerator.print(f'[Workspace] Loading full set for split calculations and normalization calculations if needed. Only for sequential. {round(time.time() - start_time, 2)}')
            dataset = dataset_generator.load_full_set() # loading after we load the custom split info 
        else:
            # SLIGHT BUG: we're not normalizing on the non-trained split, but that's ok. It's more realistic. 
            accelerator.print("[Workspace] Loading the pretraining split of data.")
            dataset = dataset_generator.load_pretrain_subset() 
        # it's critical to use the validation set split to compute the baseline 
        # for our method, which is why we register the training split and save it 
        self.training_split_info = dataset.get_training_split_info() # reload because the original split info may be a subset (if the old model was trained on a subset)

        self.model.register_training_split_info(self.training_split_info)
        if cfg.training.use_ema:
            self.ema_model.register_training_split_info(self.training_split_info)

        train_dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_prompts)
        assert isinstance(dataset, BaseDataset)
        val_dataset: BaseDataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader, collate_fn=collate_prompts)
        accelerator.print('train dataset length:', len(dataset), 'dataloader length:', len(train_dataloader))
        accelerator.print('val dataset length:', len(val_dataset), 'dataloader length:', len(val_dataloader))
        accelerator.print(f'[Workspace] Loaded datasets. {time.time() - start_time}')

        # only load the normalizer from the dataset if we are training from scratch, otherwise use the existing 
        # normalizer to prevent distribution shift 
        # good pratice is that the pretraining normalizer should work on the fine-tuning 
        if cfg.training.pretrained_checkpoint_path is None and not is_resume:
            print("[Workspace] Loading and registering normalizer from the dataset. This is expected for scratch training and NOT expected for resuming or fine-tuning")
            # compute normalizer on the main process and save to disk
            normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
            if accelerator.is_main_process:
                normalizer = dataset.get_normalizer()
                pickle.dump(normalizer, open(normalizer_path, 'wb'))

            # load normalizer on all processes
            accelerator.wait_for_everyone()
            normalizer = pickle.load(open(normalizer_path, 'rb'))
            self.model.set_normalizer(normalizer)
            
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        self.model.register_accelerator(accelerator) # this gives the model access to the parallelizer for internal calculations 
        
        ### CONFIGURING MODELS ### 
        # This is only accurate if we're doing pretraining 
        total_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_proportion * total_steps),
            num_training_steps=total_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        accelerator.wait_for_everyone() # since loading envs can take a while, halt all processes until it's finished to prevent timeout

        checkpoint_every = self.configure_checkpoint_settings(val_dataloader, cfg)
        
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        ) if cfg.checkpoint.topk.monitor_key != 'train_loss' else None
        topk_manager_train_loss = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='train_loss',
            mode='min',
            k=cfg.checkpoint.topk.k,
            format_str='epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
        )

        # accelerator
        train_dataloader, val_dataloader, self.model.base_model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model.base_model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        policy_for_count = accelerator.unwrap_model(self.model.base_model)
        n_total, n_trainable = _count_module_parameters(policy_for_count)
        accelerator.print(
            f"[Workspace] Model parameter count (policy / base_model) — total={n_total:,} trainable={n_trainable:,}"
        )
        full_total, full_trainable = _count_module_parameters(self.model)
        if full_total != n_total or full_trainable != n_trainable:
            accelerator.print(
                f"[Workspace] Learning algorithm (incl. CL extras) — total={full_total:,} trainable={full_trainable:,}"
            )
        if self.ema_model is not None:
            ema_total, ema_trainable = _count_module_parameters(self.ema_model)
            accelerator.print(
                f"[Workspace] EMA model parameter count — total={ema_total:,} trainable={ema_trainable:,}"
            )

        eval_engine = EvaluationQueue(accelerator, cfg, output_dir=self.output_dir, top_k_manager = topk_manager, 
            eval_program_path = paths.evaluate_script())

        if cfg.training.separate_process_eval and  accelerator.is_main_process: 
            wandb.define_metric('Steps')
            wandb.define_metric("*", step_metric="Steps")   
            
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        accelerator.print(f'[Workspace] Ready to go! {time.time() - start_time}')


        # initial evaluation (step 0 / untrained; decoupled from rollout_every for sequential CL)
        if not cfg.training.separate_process_eval and cfg.training.rollout_enabled and _rollout_schedule_active(cfg) and accelerator.is_main_process:
            beg = time.time()
            policy, cl_policy = _resolve_eval_policies(self.model, self.ema_model, cfg, accelerator)
            policy.eval()
            runner_log = env_rollout(cfg, env_runners, policy, cl_policy=cl_policy)
            policy.reset() # reset the policy to clear any cached data for stateful models
            print(f"Rollout time: {time.time() - beg}")
            accelerator.log(runner_log, step=0)

        elif cfg.training.separate_process_eval and cfg.training.rollout_enabled and _rollout_schedule_active(cfg) and accelerator.is_main_process:
            # save the checkpoint 
            # unwrap the model to save ckpt
            model_ddp = self.model.base_model
            self.model.base_model = accelerator.unwrap_model(self.model.base_model)
            step_log = {"epoch" : 0, "train_loss" : 0} 
            ckpt_dir = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{self.global_step}.ckpt')
            beg = time.time() 
            filepath = self.save_checkpoint(ckpt_dir, use_thread = False) # this is so we can assure that we are 100% using the untouched model 
            print("[Workspace] Time to save model for initial eval: ", round(time.time() - beg, 2))
            self.model.base_model = model_ddp
            
            if cfg.training.spawn_eval_process:
                eval_engine.set_off_run(filepath, self.global_step, self.epoch) #(filepath, self.global_step, step_log)


        # Task-boundary resume: replay finished tasks so SelectiveER history
        # (last_dataset / full_datasets) matches a normal run at this epoch.
        # The training loop then enters the next task and recomputes selection.
        if is_resume and dataset_generator.is_sequential and self.epoch > 0:
            n_finished = self.epoch // int(cfg.task.epochs_per_task)
            accelerator.print(
                f"[Workspace] Resume bootstrap: replaying {n_finished} finished task(s) "
                f"before continuing from epoch={self.epoch}"
            )
            for task_i in range(n_finished):
                accelerator.print(
                    f"[Workspace] Resume bootstrap: entering finished task "
                    f"{task_i + 1}/{n_finished}"
                )
                dataset = dataset_generator.next_task()
                train_dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_prompts)
                train_dataloader = accelerator.prepare(train_dataloader)
                self.model.start_task(current_dataset=dataset, info_cache_dir=self.output_dir)
                accelerator.wait_for_everyone()

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                accelerator.wait_for_everyone() # since some of the evaluations can take a while (env rollout happens only on main process), halt all processes until it's finished to prevent timeout

                self.model.train()
                if self.ema_model is not None:
                    self.ema_model.train()

                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    # TODO: model specific assumption that this variable exists. Can put this variable in the BaseModel class to address this
                    self.model.base_model.obs_encoder.eval()
                    self.model.base_model.obs_encoder.requires_grad_(False)
                
                
                if dataset_generator.is_sequential and (self.epoch) % cfg.task.epochs_per_task == 0:
                    print("[Workspace] Entering next task.")
                    dataset = dataset_generator.next_task()
                    train_dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_prompts)
                    total_steps = (len(train_dataloader) * cfg.task.epochs_per_task) // cfg.training.gradient_accumulate_every

                    lr_scheduler = get_scheduler(
                        cfg.training.lr_scheduler,
                        optimizer=self.optimizer,
                        num_warmup_steps=int(cfg.training.lr_warmup_proportion * total_steps),
                        num_training_steps=total_steps,
                        last_epoch=-1
                    )
                    if cfg.training.reset_optimizer_per_task: 
                        # reset the optimizer here
                        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
                        self.optimizer = accelerator.prepare(self.optimizer)

                    train_dataloader, lr_scheduler = accelerator.prepare(
                        train_dataloader,lr_scheduler
                    )
                    self.model.start_task(current_dataset = dataset, info_cache_dir=self.output_dir)
                    accelerator.wait_for_everyone() # since some of the evaluations can take a while (env rollout happens only on main process), halt all processes until it's finished to prevent timeout
                    # this is critical because 
                

                if dataset.requires_epoch_shuffle() and self.epoch != 0:
                    # shuffle data ordering (we need to recreate dataloaders as data length may change after shuffling so we recreate to be safe). Data order shuffling is specifically important for prompt based policies
                    dataset.shuffle_data_ordering(seed=self.epoch)
                    val_dataset.shuffle_data_ordering(seed=self.epoch)
                    train_dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_prompts)
                    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader, collate_fn=collate_prompts)
                    train_dataloader, val_dataloader = accelerator.prepare(
                        train_dataloader, val_dataloader
                    )
                    accelerator.print(f'Epoch {local_epoch_idx}. Recreated dataloaders')
                    accelerator.print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
                    accelerator.print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

                step_log = dict()
                train_losses = list()
    
                steps_per_task = len(train_dataloader) * cfg.task.epochs_per_task

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=True, 
                    mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # DEBUGGING, REMOVE PLS 
                        # if batch_idx > 100:
                        #     break 
                        step_log = {}
                        # beg = time.time() 
                        losses, info = self.model(batch)
                        # print("Forward time", time.time() - beg)
                        raw_loss = losses["raw_loss"]
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        beg = time.time() 
                        accelerator.backward(loss)
                        self.model.post_backward()
                        # print("backward time: ", time.time() - beg)
                        # beg = time.time() 
                        epoch_in_current_task = self.epoch % cfg.task.epochs_per_task 
                        progress = (batch_idx + epoch_in_current_task * len(train_dataloader)) / steps_per_task 
                        self.model.register_training_progress(progress)

                        # this is logging additional information 
                        scalars = info.get("scalars", None)
                        if scalars is not None:
                            step_log.update(scalars) # add these scalars to the 
                        distrs = info.get("distrs", None)
                        if distrs is not None:
                            # Only log histogram if wandb available and we're main process
                            if accelerator.is_main_process:
                                # Best-effort: handle 1. wandb logger 2. differentiate between different shapes
                                for dk, dv in distrs.items():
                                    arr = dv.detach().cpu().numpy()
                                    key = f"hist_{dk}"
                                    # Log histogram via accelerator
                                    accelerator.log({key: wandb.Histogram(arr)}, step=self.global_step)
                                
                        
                        # log grad norms
                        if cfg.training.log_grad_norm_every != -1 and self.num_grad_steps % cfg.training.log_grad_norm_every == 0 and self.global_step % cfg.training.gradient_accumulate_every == 0:
                            step_log['grad_norm'] = get_gradient_norm(self.model.base_model)
                            if cfg.training.clip_grad_norm:
                                step_log['grad_norm_clipped'] = min(cfg.training.clip_grad_norm, step_log['grad_norm'])

                        if cfg.training.clip_grad_norm and accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad_norm)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.num_grad_steps += 1
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            # TODO: figure out the best way of wokring through the encapsuatlion 
                            ema.step(accelerator.unwrap_model(self.model.base_model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log.update({
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'grad_steps': self.num_grad_steps,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        })
                        step_log.update({k: (v.item() if hasattr(v, "item") else v) for k, v in losses.items()}) # this should log all relevant losses here 
                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        
                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        exclude_keys = ['test_mean_score', 'train', 'means_train', 'means_test', 'videos_test', 'train_mean_score', 'test', 'videos_train', 'test_mean_score', 'train_mean_score', 'val_loss']
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if cfg.training.separate_process_eval: # make sure we don't log conflicting things 
                                step_log = {k : v for k, v in step_log.items() if k.split("/")[0] not in exclude_keys}
                                step_log["Steps"] = self.global_step 
                      
                            accelerator.log(step_log, step=self.global_step)
                            if accelerator.is_main_process:
                                json_logger.log(step_log)
                            eval_engine.check_and_log_results() # logs test results if present 
                            self.global_step += 1
                        # print("The rest: ", time.time() - beg)

                if len(train_losses) > 0:
                    del raw_loss, loss, batch
                
                # at the end of each epoch log epoch average train loss
                train_loss_average = np.mean(train_losses)
                step_log['train_loss_epoch_average'] = step_log['train_loss'] = train_loss_average # also sets 'train_loss' to epoch average in case we checkpoint based on 'train_loss' we should be using the epoch average not just one batch
                
                


                # ========= eval for this epoch ==========
                self.optimizer.zero_grad() # ensure optimizer is not holding any gradients so that we free up GPU memory for rollout (we need this since we might not end on a gradient accumulation step where we step the model and then zero grads); this is here in the case that the final step is not a gradient update step (due to multiple steps of gradient accumulation)
                policy, cl_policy = _resolve_eval_policies(self.model, self.ema_model, cfg, accelerator)
                policy.eval()

                # run rollout
                
                # main process rollout 
                if not cfg.training.separate_process_eval and cfg.training.rollout_enabled and _should_run_rollout_at_epoch_end(cfg, self.epoch) and accelerator.is_main_process:
                    beg = time.time()
                    runner_log = env_rollout(
                        cfg, env_runners, policy,
                        checkpoint_step=self.global_step,
                        cl_policy=cl_policy,
                    )
                    step_log.update(runner_log)
                    policy.reset() # reset the policy to clear any cached data for stateful models
                    print(f"Rollout time: {time.time() - beg}")
                elif cfg.training.separate_process_eval and cfg.training.rollout_enabled and _should_run_rollout_at_epoch_end(cfg, self.epoch) and accelerator.is_main_process:
                    # save the checkpoint 
                    # unwrap the model to save ckpt
                    model_ddp = self.model.base_model
                    self.model.base_model = accelerator.unwrap_model(self.model.base_model)

                    ckpt_dir = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{self.global_step}.ckpt')
                    filepath = self.save_checkpoint(ckpt_dir, use_thread = False) 
                    self.model.base_model = model_ddp
                    
                    if cfg.training.spawn_eval_process:
                        eval_engine.set_off_run(filepath, self.global_step, self.epoch)
                   
                # run validation
                if cfg.training.val_every != -1 and self.epoch % cfg.training.val_every == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.inference_mode():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=True, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                loss = policy(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            del batch
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

               
                
                if cfg.training.sample_every != -1 and self.epoch % cfg.training.sample_every == 0 and accelerator.is_main_process:
                    with torch.inference_mode():
                        # TODO: verify if this is ok 
                        print("[Workspace] Sampling validations")
                        self.compute_action_mse_on_dataset('train', dataset, cfg.val_dataloader, accelerator, policy, cfg, step_log) # we use val_dataloader cfg here because we want to use the batch size of the val_dataloader since we are doing evaluations
                        self.compute_action_mse_on_dataset('val', val_dataset, cfg.val_dataloader, accelerator, policy, cfg, step_log)
                
                # checkpoint
                if accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model.base_model
                    self.model.base_model = accelerator.unwrap_model(self.model.base_model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    if self.epoch > 0 and cfg.checkpoint.save_every_n_epochs > 0 and (self.epoch % int(cfg.checkpoint.save_every_n_epochs)) == 0:
                        ckpt_path = pathlib.Path(self.output_dir).joinpath(
                            'checkpoints', f'periodic_epoch={self.epoch:04d}.ckpt'
                        )
                        self.save_checkpoint(path=ckpt_path, use_thread=False)

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # always save `train_loss` checkpoint
                    # MIGHT NOT WORK WITH DEBUG MODE 
                    if not cfg.training.debug:
                        try:
                            topk_ckpt_path = topk_manager_train_loss.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)

                            if not cfg.training.separate_process_eval and (self.epoch % checkpoint_every) == 0 and topk_manager is not None:
                                # We can't copy the last checkpoint here
                                # since save_checkpoint uses threads.
                                # therefore at this point the file might have been empty!
                                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                                if topk_ckpt_path is not None:
                                    self.save_checkpoint(path=topk_ckpt_path)
                        except:
                            # TODO: the initial eval creates a topk error because of incomplete data I think 
                            print("[Workspace] Topk error, sometimes expected! This is only an issue with saving.")

                    # recover the DDP model
                    self.model.base_model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                
                self.global_step += 1
                self.epoch += 1

                # this is when we are about to advance onto the next task
                if dataset_generator.is_sequential and self.epoch > 0 and self.epoch % cfg.task.epochs_per_task == 0: # 
                    self.event_table.add_data(self.global_step - 1, f"End of task {dataset_generator.current_task}")
                    step_log["task_end"] = dataset_generator.current_task

                    self.model.end_task()
                    if accelerator.is_main_process:
                        _append_task_boundary_record(
                            self.output_dir,
                            dataset_generator.current_task,
                            self.global_step - 1,
                        )
                    if cfg.checkpoint.save_ckpt_each_task_end:
                        # Ensure all ranks sync before we start a heavy (multi-GB) save.
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            task_name = dataset_generator.current_task 
                            task_count = dataset_generator.task_count 
                            task_slug = _sanitize_task_name_for_ckpt(task_name)
                            ckpt_path = (
                                pathlib.Path(self.output_dir)
                                .joinpath("checkpoints", f"task{task_count}_{task_slug}.ckpt")
                            )

                            # unwrap model to save a checkpoint that isn't tied to DDP wrappers
                            model_ddp = self.model.base_model
                            self.model.base_model = accelerator.unwrap_model(self.model.base_model)
                            self.save_checkpoint(path=ckpt_path, use_thread=True)
                            self.model.base_model = model_ddp

                            accelerator.print(
                                f"[Workspace] Saved per-task checkpoint: "
                                f"task_idx={task_count}, task_name={task_name} -> {ckpt_path}"
                            )
                        
                        accelerator.wait_for_everyone()
                        # if cfg.training.debug: # debugging 
                        #     print("DONE WITH THE MAIN PROCESS")
                        #     break 

                    self.model.log_info(self.output_dir)

                if cfg.training.separate_process_eval:
                    step_log = {k : v for k, v in step_log.items() if "test/" not in k}
                accelerator.log(step_log, step = self.global_step - 1) #  step=self.global_step - 1) # used to be -1 but now we actually evalutate at the start 
                if accelerator.is_main_process:
                    json_logger.log(step_log)


        if cfg.training.separate_process_eval:
            if accelerator.is_main_process:
                accelerator.print("[Workspace] Waiting for all background evals to finish before shutdown.")
                eval_engine.wait_for_all()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            _write_done_file(self.output_dir)

        accelerator.print(f'Finished training. Run dir: {self.output_dir}')
        accelerator.end_training()


