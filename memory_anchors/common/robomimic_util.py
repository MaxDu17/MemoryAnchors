# From Shaung Li

import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation
import collections
import os
import cv2

from robomimic.config import config_factory
import libero.libero.envs  # noqa: F401  (registers the LIBERO environments with robosuite)
from memory_anchors.env_runner.libero_bddl_mapping import bddl_file_name_dict

    

class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        ###################################################################################################
        # modality_mapping = collections.defaultdict(list)
        # shape_meta = {'action': {'shape': [10]}, 
        #               'obs': {
        #                     'agentview_image': {'shape': [3, 128, 128], 'type': 'rgb'},
        #                     'ee_ori': {'shape': [3]},
        #                     'ee_pos': {'shape': [3]},
        #                       }
        #               }
        # for key, attr in shape_meta['obs'].items():
        #     modality_mapping[attr.get('type', 'low_dim')].append(key)
        # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        ###################################################################################################
        
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        # if env_meta['bddl_file'] not in bddl_file_name_dict:
        #     print(env_meta['bddl_file'])
        #     print(dataset_path)
        #     pdb.set_trace()
        # exit()

        if env_meta['bddl_file'] not in bddl_file_name_dict.values():
            print('convert bddl filename')
            print(env_meta['bddl_file'])
            print(env_meta['env_kwargs']['bddl_file_name'])
            
            env_meta['bddl_file'] = bddl_file_name_dict[env_meta['bddl_file']]
            env_meta['env_kwargs']['bddl_file_name'] = env_meta['bddl_file']
        else:
            print('use existing bddl file')
            print(env_meta['bddl_file'])
            print(env_meta['env_kwargs']['bddl_file_name'])

        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        
        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')

        
        # self.vis_path = 'debug_convert_action2/%s'%env_meta['bddl_file'].split('/')[-1][:-5]
        # os.makedirs(self.vis_path, exist_ok=True)
        # self.convert_and_eval_idx(0)
    

    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)

        
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(stacked_actions.shape[:-1]+(3,), dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(stacked_actions.shape[:-1]+(3,), dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(controller.goal_ori).as_rotvec()


        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

        

    
