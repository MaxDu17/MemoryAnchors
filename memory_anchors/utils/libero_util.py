import os
import pickle
import cv2
import shutil
import numpy as np
from memory_anchors.env_runner.libero_bddl_mapping import bddl_path
from memory_anchors.common.transform_util import pose_6d_to_4x4
from memory_anchors.model.common.rotation_transformer import RotationTransformer


from libero.libero import get_libero_path
from libero.libero import benchmark
import robomimic.utils.file_utils as FileUtils

def hdf5_to_bddl(hdf5_path):
    libero_split = os.path.basename(os.path.dirname(hdf5_path))
    hdf5_name = os.path.basename(hdf5_path)
    bddl_name = hdf5_name.replace("_demo.hdf5", ".bddl")
    cur_bddl_path = os.path.join(bddl_path, libero_split, bddl_name)
    return cur_bddl_path

def hdf5_to_split(hdf5_path):
    libero_split = os.path.basename(os.path.dirname(hdf5_path))
    return libero_split

def hdf5_to_task(hdf5_path):
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    libero_split_name = hdf5_to_split(hdf5_path)
    bddl_path = hdf5_to_bddl(hdf5_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_split_name]()

    # find the corresponding task id
    task_id = None
    for cur_task_id, task in enumerate(task_suite.tasks):
        if task.bddl_file == os.path.basename(bddl_path):
            task_id = cur_task_id
            break
    assert task_id is not None, f"task id not found for bddl file: {env_meta['bddl_file']}"
    assert task.problem_folder == libero_split_name

    task = task_suite.get_task(task_id)
    return task

# def hdf5_to_task(hdf5_path):
#     """Map a LIBERO demonstration HDF5 path to benchmark `Task` (problem_folder, init_states_file, etc.)."""
#     from libero.libero.benchmark import task_maps

#     libero_split = os.path.basename(os.path.dirname(hdf5_path))
#     hdf5_name = os.path.basename(hdf5_path)
#     assert hdf5_name.endswith("_demo.hdf5"), f"expected *_demo.hdf5, got {hdf5_name!r}"
#     task_name = hdf5_name[: -len("_demo.hdf5")]
#     assert libero_split in task_maps, (
#         f"unknown LIBERO suite {libero_split!r}; expected one of {list(task_maps.keys())}"
#     )
#     assert task_name in task_maps[libero_split], (
#         f"unknown task {task_name!r} in suite {libero_split!r}"
#     )
#     return task_maps[libero_split][task_name]


def pad_action(action, action_rep, desired_length):
    """
    Pads the input action array of shape (B, T, D=10) according to the action_rep parameter, up to desired_length.
    Args:
        action (np.ndarray): Input action array of shape (B, T, 10).
        action_rep (str): Either 'delta' or 'absolute'.
        desired_length (int): Desired length along the time dimension (T).
    Returns:
        np.ndarray: Padded action array of shape (B, desired_length, 10).
    """
    assert action.shape[-1] == 10, "Action dimension D must be 10."
    B, T, D = action.shape
    # Split action into position, rot6d, gripper
    pos = action[..., :3]        # (B, T, 3)
    rot6d = action[..., 3:9]     # (B, T, 6)
    gripper = action[..., 9:]    # (B, T, 1)

    pad_count = max(0, desired_length - T)
    if pad_count == 0:
        return action

    if action_rep == 'delta':
        pad_pos = np.zeros((B, pad_count, 3), dtype=action.dtype)
        pad_rot6d = np.tile(np.array([[1,0,0,0,1,0]], dtype=action.dtype), (B, pad_count, 1)) # identity rotation
    elif action_rep == 'absolute':
        pad_pos = np.tile(pos[:, -1:, :], (1, pad_count, 1))
        pad_rot6d = np.tile(rot6d[:, -1:, :], (1, pad_count, 1))
    else:
        raise ValueError(f"Unknown action_rep: {action_rep}")

    pad_gripper = np.tile(gripper[:, -1:, :], (1, pad_count, 1)) # all rotation representations use the same gripper action and we chose to just repeat the last gripper action

    pad_action = np.concatenate([pad_pos, pad_rot6d, pad_gripper], axis=-1)  # (B, pad_count, 10)
    padded_action = np.concatenate([action, pad_action], axis=1)  # (B, T+pad_count, 10)
    return padded_action

