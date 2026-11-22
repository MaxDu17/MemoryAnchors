import os
import sys
from typing import Optional, Union
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import torch.nn.functional as F
from torch.utils.data import Subset
# import pytorch3d.transforms as pt
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
)
from transformers import CLIPTokenizer

from memory_anchors.utils.prompt_util import PromptActionChunker
from memory_anchors.utils.plot_util import combine_attention_and_rollout_video, vis_actions_rollout, vis_attention_map, vis_proprio_rollout
from memory_anchors.utils.video_recorder import VideoRecorder

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

## here we just use the same env wrapper as robomimic
from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from memory_anchors.env_runner.libero_bddl_mapping import bddl_file_name_dict

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from ..model.language.language_embedding import LanguageEmbedding
from memory_anchors.model.common.rotation_transformer import (
    RotationTransformer,
)
from libero.libero.envs.bddl_utils import get_problem_info
from memory_anchors.utils.libero_util import hdf5_to_bddl, hdf5_to_task, pad_action
from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.dataset.libero_replay_image_dataset import LiberoReplayImageDataset
from memory_anchors.common.torch_util import move_batch_to_device, move_batch_to_numpy, remove_batch_dim_from_prompt
from memory_anchors.common.transform_util import pose_6d_to_4x4
from memory_anchors.utils.prompt_util import collate_prompts

from libero.libero import get_libero_path
from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.utils.binding_utils import MjSimState

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)

    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    if env_meta["bddl_file"] not in bddl_file_name_dict.values():
        print("convert bddl filename")
        print(env_meta["bddl_file"])
        print(env_meta["env_kwargs"]["bddl_file_name"])
        env_meta["bddl_file"] = bddl_file_name_dict[env_meta["bddl_file"]]
        env_meta["env_kwargs"]["bddl_file_name"] = env_meta["bddl_file"]
    else:
        print("use existing bddl file")
        print(env_meta["bddl_file"])
        print(env_meta["env_kwargs"]["bddl_file_name"])
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class LiberoImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        task_dir,
        output_dir,
        dataset_path,
        language_embedding: LanguageEmbedding,
        shape_meta: dict,
        use_finetuned_language_encoder: bool=False,
        text_encoder_model_name: Optional[str]=None,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        render_obs_key=["agentview_image", "robot0_eye_in_hand_image"],
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        n_envs=None,
        include_file_filters:list[str]=[], # used in load_env.py which is why it's unused here
        exec_action_horizon:int=12,
        vis_prompt:bool=False,
        vis_actions_rollout:bool=False,
        vis_proprio_rollout:bool=False,
        vis_trajectory_video:bool=False,
        replay_dataset_action: bool=False,
        vis_attention_map: bool=False,
        vis_env_index: int=0,
        run_train_config: bool=False,
        save_sim_states: bool=False,
        multi_bddl_path: Optional[str]=None, # path to where the bddl files are 
    ):

        super().__init__(output_dir)

        max_steps += exec_action_horizon # at the start of the rollout after reset we first execute the open gripper action so we need to add the action horizon to the max steps to ensure we accoutn for these additional open gripper actions

        if n_envs is None:
            n_envs = n_train + n_test
        
        self.n_train = n_train
        self.n_test = n_test 

        dataset_path = task_dir
        robosuite_fps = 20
        self.steps_per_render = max(robosuite_fps // fps, 1)
        self.use_prompting = shape_meta['use_prompting']
        self.prompt_sample_mode = shape_meta['prompt_sample_mode'] if self.use_prompting else None
        self.action_rep = shape_meta['action']['rep']
        self.demo_name = os.path.basename(dataset_path).replace('.hdf5', '') # get the name of the hdf5 file without the extension

        # observation/action horizon
        self.key_horizon = dict()
        for key, attr in shape_meta['obs'].items():
            self.key_horizon[key] = shape_meta['obs'][key]['horizon']

        if self.prompt_sample_mode == 'sequence':
            # sequence prompting policies need to get all observations that occur during the rollout so we need to have the env runner recording observations for the entire batch of the past actions executed for each step
            max_obs_horizon = exec_action_horizon
        else:
            # for other policies (non-prompting or pair prompting) we need to record history of observations with length of the longest observation horizon
            max_obs_horizon = max(self.key_horizon.values())
        if vis_proprio_rollout:
            # if we want to visualize the proprio we need to record the entire batch of the past proprio for the entire duration of executed actions
            max_obs_horizon = max(max_obs_horizon, exec_action_horizon)
        assert exec_action_horizon <= shape_meta['action']['horizon'], 'number of steps to execute must not be greater than the action horizon predicted by the policy'

        # env_shape_meta should not have language
        env_shape_meta = shape_meta.copy()
        del env_shape_meta['obs']['task_language']

        self.rgb_keys = []
        self.lowdim_keys = []
        for key in env_shape_meta['obs']:
            if env_shape_meta['obs'][key]['type'] == 'rgb':
                self.rgb_keys.append(key)
            elif env_shape_meta['obs'][key]['type'] == 'low_dim':
                self.lowdim_keys.append(key)

        self.obs_key_to_runner_key = { # the names of the observations in simulation have different names (and potentially different formats) than the ones in the training dataset (which is the naming convention that the policy uses)
            'agentview_rgb': 'agentview_image', # just rename
            'eye_in_hand_rgb': 'robot0_eye_in_hand_image', # just rename
            'ee_pos': 'robot0_eef_pos', # just rename
            'ee_ori': 'robot0_eef_quat', # sim gives xyzw quat (note real part last!), need to convert to xyz euler axis-angle
            'gripper_states': 'robot0_gripper_qpos' # just rename
        }
        self.runner_key_to_obs_key = {v: k for k,v in self.obs_key_to_runner_key.items()}

        # update rgb keys in env_shape_meta
        for rgb_key in self.rgb_keys:
            runner_key = self.obs_key_to_runner_key[rgb_key]
            env_shape_meta["obs"][runner_key] = env_shape_meta['obs'].pop(rgb_key)
            env_shape_meta["obs"][runner_key]['shape'] = [3, 128, 128] # the simulator provides (128, 128), while the policy input is (224, 224)

        # proprio keys
        proprio_key_names = []
        proprio_key_sizes = []

        # update lowdim keys in env_shape_meta
        for lowdim_key in sorted(self.lowdim_keys):
            runner_key = self.obs_key_to_runner_key[lowdim_key]
            env_shape_meta["obs"][runner_key] = env_shape_meta['obs'].pop(lowdim_key)
            if runner_key == 'robot0_eef_quat':
                env_shape_meta["obs"][runner_key]['shape'] = [4] # axis angle to quat is change from [3] to [4]

            if shape_meta["obs"][lowdim_key]['prompt_type'] == 'proprioception':
                proprio_key_names.append(lowdim_key)
                shape = shape_meta["obs"][lowdim_key]['shape']
                assert len(shape) == 1, f'proprio value must be 1D'
                proprio_key_sizes.append(shape[0])
        
        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = self.action_rep == 'delta'
        # env_meta["env_kwargs"]["render_gpu_device_id"] = 1
        # parallel_evaluation subprocess sets CUDA_VISIBLE_DEVICES to a single GPU; robosuite EGL uses device index 0 there.

        self.task = hdf5_to_task(dataset_path)
        self.dataset_name = os.path.basename(os.path.dirname(dataset_path))
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.task.problem_folder,
            self.task.init_states_file,
        )
        print(f"[ImageRunner] Loading init states from {init_states_path}")

        init_states = torch.load(
            init_states_path, map_location="cpu", weights_only=False
        )
        assert n_test <= len(init_states), (
            f"n_test ({n_test}) must be <= number of benchmark init states ({len(init_states)})"
        )

        def env_fn():
            libero_env = create_env(env_meta=env_meta, shape_meta=env_shape_meta)
            libero_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=libero_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=self.steps_per_render,
                ),
                n_obs_steps=max_obs_horizon,
                n_action_steps=exec_action_horizon,
                max_episode_steps=max_steps,
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            libero_env = create_env(
                env_meta=env_meta, shape_meta=env_shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=libero_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=self.steps_per_render,
                ),
                n_obs_steps=max_obs_horizon,
                n_action_steps=exec_action_horizon,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        pre_collected_actions = None
        pre_collected_obs = None


        # train config (rollout but initialize with a training state)
        if run_train_config:
            with h5py.File(dataset_path, "r") as f:
                for i in range(n_train):
                    train_idx = train_start_idx + i
                    enable_render = i < n_train_vis
                    init_state = f[f"data/demo_{train_idx}/states"][0]
                    env_seed = train_idx

                    if i == 0:
                        action_key = 'actions' if self.action_rep == 'delta' else 'abs_actions'
                        pre_collected_actions = f[f"data/demo_{train_idx}/{action_key}"][:]
                        pre_collected_obs = {k: v[:] for k, v in f[f"data/demo_{train_idx}/obs"].items()}
                    
                    def init_fn(
                        env,
                        init_state=init_state,
                        enable_render=enable_render,
                        demo_name=self.demo_name,
                        env_seed=env_seed,
                    ):
                        # setup rendering
                        # video_wrapper
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = None
                        if enable_render:
                            filename = pathlib.Path(output_dir).joinpath(
                                "media",
                                f"train_{demo_name}_{env_seed}_{wv.util.generate_id()}.mp4",
                            )
                            filename.parent.mkdir(parents=False, exist_ok=True)
                            filename = str(filename)
                            env.env.file_path = filename

                        # Seed, full sim reset, then apply dataset init state (see LIBERO README).
                        assert isinstance(env.env.env, RobomimicImageWrapper)
                        env.env.env.register_init_state(None)
                        env.seed(7)
                        env.reset()
                        env.env.env.register_init_state(init_state)

                    env_seeds.append(train_idx)
                    env_prefixs.append(
                        f"train/{self.dataset_name}/"
                        + env_meta["bddl_file"].split("/")[-1][:-5]
                        + "_"
                    )
                    env_init_fn_dills.append(dill.dumps(init_fn))
        # test config: benchmark pruned init states (same files as LIBERO lifelong / evaluate)
        for i in range(n_test):
            enable_render = i < n_test_vis
            env_seed = test_start_seed + i
            init_state = init_states[i]

            def init_fn(
                env,
                init_state=init_state,
                enable_render=enable_render,
                demo_name=self.demo_name,
                env_seed=env_seed,
            ):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media",
                        f"test_{demo_name}_{env_seed}_{wv.util.generate_id()}.mp4",
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                w = env.env.env
                assert isinstance(w, RobomimicImageWrapper)
                w.register_init_state(None)
                env.seed(7)
                env.reset()
                w.register_init_state(init_state)

            env_seeds.append(env_seed)
            env_prefixs.append(
                f"test/{self.dataset_name}/"
                + env_meta["bddl_file"].split("/")[-1][:-5]
                + "_"
            )
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)
        
        self.shape_meta = shape_meta
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.max_steps = len(pre_collected_actions) if replay_dataset_action else max_steps
  
        self.rotation_transformer_axisangle_to_rot6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
        self.rotation_transformer_quatwxyz_to_rot6d = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
        self.rotation_transformer_quatwxyz_to_axisangle = RotationTransformer(from_rep='quaternion', to_rep='axis_angle')
        self.rotation_transformer_axis_angle_to_matrix = RotationTransformer(from_rep='axis_angle', to_rep='matrix')
        self.tqdm_interval_sec = tqdm_interval_sec
        self.dataset_path = dataset_path
        self.language_embedding = language_embedding
        self.prompt_sequence_length = shape_meta.prompt_sequence_length
        self.exec_action_horizon = exec_action_horizon
        self.vis_prompt = vis_prompt
        self.vis_actions_rollout = vis_actions_rollout
        self.vis_proprio_rollout = vis_proprio_rollout
        self.vis_trajectory_video = vis_trajectory_video
        self.replay_dataset_action = replay_dataset_action
        self.proprio_key_names = proprio_key_names
        self.proprio_key_sizes = proprio_key_sizes
        self.pre_collected_actions = pre_collected_actions
        self.pre_collected_obs = pre_collected_obs
        self.vis_attention_map = vis_attention_map
        self.vis_env_index = vis_env_index
        self.save_sim_states = save_sim_states

        self.multi_bddl_paths = None 
        if multi_bddl_path is not None:
            bddl_paths = list() 
            for fname in os.listdir(multi_bddl_path):
                if fname.endswith(".bddl"):
                    bddl_paths.append(os.path.abspath(os.path.join(multi_bddl_path, fname)))
                    bddl_paths.sort()
            self.multi_bddl_paths = bddl_paths 


        self.use_finetuned_language_encoder = use_finetuned_language_encoder
        cur_bddl_path = hdf5_to_bddl(dataset_path)
        problem_info = get_problem_info(cur_bddl_path)
        self.language_goal = problem_info["language_instruction"]

        if self.use_finetuned_language_encoder:
            # Initialize CLIP tokenizer and tokenize task name
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
            tokens = self.clip_tokenizer(
                self.language_goal,
                padding='max_length',
                truncation=True,
                max_length=77,  # CLIP's default max length
                return_tensors='np'
            )
            self.task_language_token_ids = tokens['input_ids'].astype(np.int64)  # (1, seq_len)
        else:
            # Use precomputed embedding
            self.language_goal_embedding = language_embedding.embed(self.language_goal)

        # compute language goal embedding
        # self.language_goal_embedding = language_embedding.embed(self.language_goal)

    def run(self, policy: Optional[BasePolicy], video_format = "wandb", close_afterward = False,  checkpoint_step = None, **kwargs):
        # this is the code that interfaces with the evaluations 
        if self.replay_dataset_action:
            device: torch.device = torch.device('cpu')
        else:
            device = policy.device
            assert not policy.training, "policy must be in eval mode"

        if self.vis_attention_map or self.vis_trajectory_video:
            raise NotImplementedError("vis_attention_map / vis_trajectory_video were removed in the MemoryAnchors release.")

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_multi_bddl_success = [None] * n_inits if self.multi_bddl_paths else None

        print("env_runner: ", self.language_goal)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # init policy
            if policy is not None:
                policy.reset(action_exec_horizon=self.exec_action_horizon)

            # if policy is prompt based model, then prompt
            if self.use_prompting:
                raise Exception("Prompting is not used in this codebase! Check your configs!")

            # env_name = self.env_meta['env_name']
            env_name = self.env_meta["bddl_file"].split("/")[-1][:-5]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=True,
                mininterval=self.tqdm_interval_sec,
            )

            def prepare_obs_dict(obs):
                """Prepares the observations for the policy inference.
                Returns two versions of the observation dict. The first is the numpy version which is used for visualization (contains full observation history). The second is the torch version which is used for policy inference (contains the format the policy expects).
                """
                visualization_obs_dict = dict(obs)

                # rename simulator keys to match policy keys
                for obs_key in list(visualization_obs_dict.keys()):
                    policy_key_name = self.runner_key_to_obs_key[obs_key]
                    # rename observation from sim name to policy name
                    visualization_obs_dict[policy_key_name] = visualization_obs_dict.pop(obs_key)
                
                # convert rotation representation for ee_ori from quat xyzw from simulator to rot6d for policy
                if 'ee_ori' in visualization_obs_dict:
                    quat_xyzw = visualization_obs_dict['ee_ori']
                    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]] 
                    rot6d = self.rotation_transformer_quatwxyz_to_rot6d.forward(quat_wxyz)
                    visualization_obs_dict['ee_ori'] = rot6d

                # device transfer
                policy_obs_dict = dict_apply(
                    visualization_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # trim keys to the correct horizon
                for obs_key in policy_obs_dict.keys():
                    horizon = self.key_horizon[obs_key] if self.prompt_sample_mode != 'sequence' else self.exec_action_horizon
                    policy_obs_dict[obs_key] = policy_obs_dict[obs_key][:, -horizon:]

                # resize the rgb keys to the appropriate size
                for rgb_key in self.rgb_keys:
                    # resize image to be 224x224
                    resize = 224
                    B, T, C, H, W = policy_obs_dict[rgb_key].shape
                    resized_tensor = F.interpolate(
                        policy_obs_dict[rgb_key].reshape(B * T, C, H, W),
                        size=(resize, resize),
                        mode="bilinear",
                        align_corners=False,
                    )
                    policy_obs_dict[rgb_key] = resized_tensor.view(
                        B, T, C, resize, resize
                    )

                # insert language goal
                if self.use_finetuned_language_encoder:
                    # Pass tokenized task name (token IDs) - will be encoded by model
                    horizon = self.key_horizon['task_language']
                    # Repeat token IDs for each time step and each env
                    # Shape: (n_envs, horizon, seq_len)
                    token_ids = np.tile(self.task_language_token_ids, (n_envs, horizon, 1))
                    policy_obs_dict['task_language'] = torch.from_numpy(token_ids).to(device=device)  # (B, T, seq_len)
                elif 'task_language' in self.shape_meta['obs']:
                    # Use precomputed embedding
                    policy_obs_dict['task_language'] = torch.from_numpy(self.language_goal_embedding).unsqueeze(0).unsqueeze(0).repeat((n_envs, self.key_horizon['task_language'], 1)).to(device=device) # (B, T, D)

                # if 'task_language' in self.shape_meta['obs'] and not self.shape_meta['obs']['task_language'].get('ignore_by_policy', False):
                #     policy_obs_dict['task_language'] = torch.from_numpy(self.language_goal_embedding).unsqueeze(0).unsqueeze(0).repeat((n_envs, self.key_horizon['task_language'], 1)).to(device=device) # (B, T, D)

                return visualization_obs_dict, policy_obs_dict

            # reset env
            obs = env.reset()

            # weirdly it seems during data collection the gripper is fully open, but in the test environments the gripper can be initialized to be partially closed. This can lead to out of distribution issues, so we just open the gripper all the way
            if self.action_rep == 'delta':
                open_gripper_action = np.zeros((n_envs, self.exec_action_horizon, 7))
                open_gripper_action[:, :, 6] = -1
            else: # absolute
                # we command the robot to its current position with a gripper open action
                pos = obs['robot0_eef_pos'][:, -1] # (B, 3)
                quat_xyzw = obs['robot0_eef_quat'][:, -1] # (B, 4)
                quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]] # (B, 4)
                axis_angle = self.rotation_transformer_quatwxyz_to_axisangle.forward(quat_wxyz) # (B, 3)
                rot_mat = self.rotation_transformer_axis_angle_to_matrix.forward(axis_angle) # (B, 3, 3)

                # rotate -90 degrees about z (for some reason eef_quat needs this rotation)
                rot_z_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
                                   [np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
                                   [0, 0, 1]])
                rot_mat = rot_mat @ rot_z_90
                axis_angle = self.rotation_transformer_axis_angle_to_matrix.inverse(rot_mat) # (B, 3)

                gripper_action = np.full((n_envs, 1), -1)
                open_gripper_action = np.concatenate([pos, axis_angle, gripper_action], axis=-1) # (B, 7)

                # insert horizon
                open_gripper_action = np.expand_dims(open_gripper_action, axis=1) # (B, 1, 7)
                open_gripper_action = np.tile(open_gripper_action, (1, self.exec_action_horizon, 1)) # (B, exec_action_horizon, 7)

            obs, reward, done, info = env.step(open_gripper_action)
            # obs, reward, done, info = env.step(np.zeros((n_envs, self.exec_action_horizon, 7))) # I observe the robot moves a little bit even with zero actions when doing action, perhaps since the initial state has some initial velocity. Add this line back in to observe this effect.

            visualization_obs_dict, policy_obs_dict = prepare_obs_dict(obs)

            # store initial proprioception
            cur_proprio_rollout = np.concatenate([visualization_obs_dict[key] for key in self.proprio_key_names], axis=-1) # (B, T, proprio_dim)
            all_proprio_rollout = cur_proprio_rollout[:, :1] # (B, 1, proprio_dim); only take 1 step of the initial proprio since we are at the start of the rollout and only have observed a single state

            # start rollout
            all_execed_actions = None # (B, T, action_dim)
            done = False
            pred_steps = 0
            if self.vis_attention_map:
                all_cross_attn_weights = []
            while not done:
                # get action either from policy or from replay dataset
                if self.replay_dataset_action:
                    action = self.pre_collected_actions[:self.exec_action_horizon] # self.pre_collected_actions is (T_demonstration, action_dim) and contains actions for just the first training env
                    action = np.expand_dims(action, axis=0) # (1, T_demonstration, action_dim)
                    
                    # pad with zeros for rest of the envs
                    n_pad = n_envs - action.shape[0]
                    if n_pad > 0:
                        action = np.pad(action, ((0, n_pad), (0,0), (0,0)), mode='constant')

                    # pad horizon if necessary
                    if action.shape[1] < self.exec_action_horizon:
                        action = self.transform_action_for_policy(action, undo=False) # axis-angle -> rot6d (B, num_pred_steps, 10)
                        action = pad_action(action, self.action_rep, self.exec_action_horizon)
                        action = self.transform_action_for_policy(action, undo=True) # rot6d -> axis-angle (B, num_pred_steps, 7)

                    # check that the shape is correct
                    assert action.shape[0] == n_envs and action.shape[1] == self.exec_action_horizon, f"cur_action.shape: {action.shape} is not expected shape"

                    self.pre_collected_actions = self.pre_collected_actions[self.exec_action_horizon:]
                else:
                    with torch.inference_mode():
                        if self.vis_attention_map:
                            kwargs['need_weights'] = True
                            kwargs['average_attn_weights'] = True
                        action_dict = policy.predict_action(
                            policy_obs_dict,
                            **kwargs,
                        )
                    
                    action = action_dict["action"].detach().to("cpu").numpy()  # (B, num_pred_steps, 10)
                    action = self.transform_action_for_policy(action, undo=True) # rot6d -> axis-angle (B, num_pred_steps, 7)
                    action = action[:, :self.exec_action_horizon] # exec only part of the full action trajectory (B, exec_action_horizon, 7)

                    if self.vis_attention_map:
                        cross_attn_weights = action_dict['cross_attn_weights'].detach().to("cpu").numpy() # (num_diffusion_steps, num_layers, batch, action_pred_steps, cross_attn_dim)
                        cross_attn_weights = cross_attn_weights[:,:,self.vis_env_index] # (num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim) select the first env only
                        all_cross_attn_weights.append(cross_attn_weights)

                # step env
                obs, reward, done, info = env.step(action)

                for i in range(len(reward)):
                    if reward[i] == 1:
                        done[i] = True

                done = np.all(done)

                # prepare obs for next iteration
                visualization_obs_dict, policy_obs_dict = prepare_obs_dict(obs)

                if self.vis_proprio_rollout or self.vis_trajectory_video:
                    cur_proprio_rollout = np.concatenate([visualization_obs_dict[key] for key in self.proprio_key_names], axis=-1) # (B, T, proprio_dim)
                    all_proprio_rollout = np.concatenate([all_proprio_rollout, cur_proprio_rollout[:, :self.exec_action_horizon]], axis=1) # concat along time dimension
                
                if self.vis_actions_rollout:
                    if all_execed_actions is None:
                        all_execed_actions = action[:, :self.exec_action_horizon]    
                    else:
                        all_execed_actions = np.concatenate([all_execed_actions, action[:, :self.exec_action_horizon]], axis=1)

                # update pbar
                pbar.update(action.shape[1])
                pred_steps += action.shape[1]

                # if self.use_prompting and total_available_action_steps_after_prompting is not None:
                #     # prompting models have a fixed sequence length so we have to stop once we reach the sequence length
                #     if pred_steps >= total_available_action_steps_after_prompting:
                #         done = True
                
                if self.replay_dataset_action:
                    if len(self.pre_collected_actions) == 0:
                        done = True
            
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
            if self.multi_bddl_paths is not None:
                multi_results = env.call("check_success_multi_bddl", self.multi_bddl_paths)
                all_multi_bddl_success[this_global_slice] = list(multi_results)[this_local_slice]

            # # save final sim states for offline multi-BDDL failure analysis
            # if self.save_sim_states:
            #     try:
            #         sim_states = env.call("get_sim_state")
            #         bddl_file = self.env_meta.get("bddl_file") or self.env_meta.get("env_kwargs", {}).get("bddl_file_name", "")
            #         task_name = self.env_meta["bddl_file"].split("/")[-1][:-5] if self.env_meta.get("bddl_file") else os.path.basename(self.dataset_path).replace(".hdf5", "")
            #         step_dir = f"step_{checkpoint_step}" if checkpoint_step is not None else "eval"
            #         save_root = pathlib.Path(self.output_dir) / "sim_states" / step_dir / task_name
            #         save_root.mkdir(parents=True, exist_ok=True)
            #         for k in range(len(sim_states)):
            #             # TODO: separate into training and testing splits here 
            #             # the first X are train splits, and the rest are test. Reflect that here! 
            #             global_idx = start + k # global idx 
            #             # if global_idx < n_inits:

            #             seed = self.env_seeds[global_idx]
            #             r = all_rewards[global_idx]
            #             reward = float(np.max(np.atleast_1d(r))) if r is not None else None
            #             data = {
            #                 "sim_state": np.array(sim_states[k]),
            #                 "bddl_file": bddl_file,
            #                 "task_name": task_name,
            #                 "seed": seed,
            #                 "rollout_idx": global_idx,
            #                 "reward": reward,
            #                 "eval_env": global_idx >= self.n_train # is this an evaluation split or a train split 
            #             }
            #             fpath = save_root / f"seed_{seed:06d}_rollout_{global_idx:04d}.pt"
            #             torch.save(data, fpath)
            #     except Exception as e:
            #         print(f"[LiberoImageRunner] Warning: failed to save sim states: {e}")

        # clear out video buffer
        _ = env.reset()

        # if self.use_prompting and total_available_action_steps_after_prompting is not None and total_available_action_steps_after_prompting < self.max_steps:
        #     assert policy.num_available_actions() < self.exec_action_horizon, 'if we are limited by the number of steps we can execute, then the number of steps we can execute should be less than the number of steps we can predict at the very end of execution'

        vis_video_path = all_video_paths[self.vis_env_index] if len(all_video_paths) > self.vis_env_index else None
        vis_demo_name = os.path.basename(vis_video_path).replace('.mp4', '') if vis_video_path else f"{self.demo_name}_env{self.vis_env_index}"

        # visualize action rollout
        if self.vis_actions_rollout:
            first_execed_action = all_execed_actions[self.vis_env_index]

            # convert rotation in first_execed_action from axis-angle to rot6d to match the prompt format
            pos = first_execed_action[:, :3]
            axisangle = first_execed_action[:, 3:6]
            gripper_action = first_execed_action[:, 6:]
            rot6d = self.rotation_transformer_axisangle_to_rot6d.forward(axisangle)
            first_execed_action = np.concatenate([pos, rot6d, gripper_action], axis=-1)

            vis_out_dir = os.path.join(self.output_dir, "vis")
            os.makedirs(vis_out_dir, exist_ok=True)
            prompt_for_vis = None #remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index) if self.use_prompting else None
            vis_actions_rollout(os.path.join(vis_out_dir, f"actions_rollout_{vis_demo_name}.png"), first_execed_action, prompt=prompt_for_vis)

        if self.replay_dataset_action:
            if 'ee_ori' in self.proprio_key_names:
                self.pre_collected_obs['ee_ori'] = self.rotation_transformer_axisangle_to_rot6d.forward(self.pre_collected_obs['ee_ori'])
            all_proprio_replay = np.concatenate([self.pre_collected_obs[key] for key in self.proprio_key_names], axis=-1) # (T, proprio_dim) for the first env only

        vis_out_dir = os.path.join(self.output_dir, "vis")
        os.makedirs(vis_out_dir, exist_ok=True)

        # visualize proprio rollout
        if self.vis_proprio_rollout:
            first_proprio = all_proprio_rollout[self.vis_env_index]
            proprio_list = [first_proprio]
            proprio_legend_names = ["Rollout observations"] if not self.use_prompting else ["Prompt then rollout observations"]

            if self.replay_dataset_action:
                proprio_list.append(all_proprio_replay)
                proprio_list[0] = proprio_list[0][:len(proprio_list[1])] # the proprio rollout is longer than the precollected actions because the full exec horizon happens each iteration which may exceed the length of the precollected proprio, so we just trim it
                proprio_legend_names.append("Replay dataset observations")

            prompt_for_vis = None # remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index) if self.use_prompting else None
            vis_proprio_rollout(
                os.path.join(vis_out_dir, f"proprio_rollout_{vis_demo_name}.png"),
                proprio_list=proprio_list,
                proprio_key_names=self.proprio_key_names, 
                proprio_key_sizes=self.proprio_key_sizes, 
                prompt=prompt_for_vis,
                proprio_legend_names=proprio_legend_names
            )

        # visualize attention map
        # if self.vis_attention_map:
        #     prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index)
        #     prompt_for_vis = move_batch_to_numpy(prompt_for_vis)
        #     prompt_len = prompt_for_vis['action'].shape[0]

        #     all_cross_attn_weights = np.stack(all_cross_attn_weights, axis=0) # (num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        #     attention_map_path = os.path.join(vis_out_dir, f"attention_map_{vis_demo_name}.mp4")
        #     vis_attention_map(attention_map_path, all_cross_attn_weights, self.exec_action_horizon, policy.get_cross_attn_dim_names(prompt_len), fps=20)

        #     out_path = os.path.join(vis_out_dir, f"attention_and_rollout_{vis_demo_name}.mp4")

        #     # combine attention map and rollout video
        #     combine_attention_and_rollout_video(attention_map_path, all_video_paths[self.vis_env_index], out_path, prompt_for_vis, self.rgb_keys, steps_per_render=self.steps_per_render)
        #     os.remove(attention_map_path)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward
            if all_multi_bddl_success is not None and all_multi_bddl_success[i] is not None:
                log_data[prefix + f"multi_bddl_success_{seed}"] = all_multi_bddl_success[i]
            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None: 
                # crucial modification to allow for passthrough 
                if video_format == "wandb":
                    sim_video = wandb.Video(video_path)
                else:
                    sim_video = video_path 
                log_data["videos_" + prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = "means_" + prefix + "mean_score" # add separate panel for aggregation 
            value = np.mean(value)
            log_data[name] = value

        if close_afterward: # this is critical if we don't want to keep around a ton of memory after run() is complete. Only if you're done running. 
            self.env.close() 
            # self.env.close_extras()
        return log_data

    def transform_action_for_policy(self, action, undo=False):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        if undo:
            rot = self.rotation_transformer_axisangle_to_rot6d.inverse(rot)
        else:
            rot = self.rotation_transformer_axisangle_to_rot6d.forward(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
