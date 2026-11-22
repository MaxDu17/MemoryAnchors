import hashlib
import json
from omegaconf import DictConfig, OmegaConf
from memory_anchors import paths
from typing import Dict, List, Tuple
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import glob
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import LinearNormalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from memory_anchors.common.replay_buffer import (
    ReplayBuffer,
    check_chunks_compatible,
    get_optimal_chunks,
)
from memory_anchors.common.sampler import SequenceSampler, get_train_mask_from_training_split_info, get_training_split_info_from_train_mask, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from typing import Optional
import torch.nn.functional as F

from transformers import CLIPTokenizer


register_codecs()
from ..model.language.language_embedding import LanguageEmbedding

from libero.libero.envs.bddl_utils import get_problem_info
from memory_anchors.utils.libero_util import hdf5_to_bddl
from memory_anchors.common.augmentation import ImageAugmentation
from memory_anchors.dataset.base_dataset import BaseDataset


def _cache_key(dataset_path, shape_meta) -> str:
    """Short hash identifying a (dataset location, observation/action config) pair."""
    if isinstance(shape_meta, DictConfig):
        shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
    blob = json.dumps({"path": os.path.abspath(dataset_path), "shape_meta": shape_meta}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]

class LiberoReplayImageDataset(BaseDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        language_embedding: LanguageEmbedding,
        action_padding: bool=False,
        use_cache=False,
        overwrite_cache=False,
        cache_dir: Optional[str]=None,
        seed=42,
        val_ratio=0.0,
        dataset_basepath = None,
        sample_type:str='episode',
        only_prompt:bool=False,
        max_segments:int=-1,
        name_suffix:str='',
        include_file_filters: list[str]=[],
        training_split_info: Optional[Dict[Tuple[str, str], bool]]=None,
        use_finetuned_language_encoder: bool=False,
        text_encoder_model_name: Optional[str]=None,
    ):
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )
        self.dataset_path = dataset_path

        if dataset_basepath is not None: # allows you to manually specify a basepath 
            self.dataset_path = os.path.join(dataset_basepath, self.dataset_path)

        self.use_cache = use_cache and len(include_file_filters) == 0 # keeps track for representation purposes
        if len(include_file_filters) > 0: 
            print("[Dataset] Using these following tasks in training: ", include_file_filters)
        # if you use file filters, you will incorrectly load a cache!  
        replay_buffer = None
        if self.use_cache:
            assert len(include_file_filters) == 0
            # raise Exception("NO longer using cache due to risky bug") # cache doesn't listen to new data splits 
            if cache_dir is None:
                cache_dir = os.path.join(paths.cache_root(), 'libero')
            os.makedirs(cache_dir, exist_ok=True)
            # Key the cache on the absolute dataset path and shape_meta so that two datasets
            # of the same suite (or a changed observation config) never share a stale cache.
            file_name = os.path.join(cache_dir, f"{os.path.basename(os.path.normpath(self.dataset_path))}_{_cache_key(self.dataset_path, shape_meta)}")
            if name_suffix:
                file_name = f"{file_name}_{name_suffix}"
            cache_zarr_path = file_name + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("[Dataset] Acquiring lock on cache.")
            print("[Dataset] Cache path:", cache_zarr_path)
            with FileLock(cache_lock_path):
                if overwrite_cache and os.path.exists(cache_zarr_path):
                    print("[Dataset] Overwriting cache.")
                    os.remove(cache_zarr_path)
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("[Dataset] Cache does not exist; creating cache.")

                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=self.dataset_path,
                            rotation_transformer=self.rotation_transformer,
                            include_file_filters=include_file_filters
                        )
                        print("[Dataset] Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise e
                else:
                    print("[Dataset] Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("[Dataset] Finished loading cached ReplayBuffer")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=self.dataset_path,
                rotation_transformer=self.rotation_transformer,
                include_file_filters=include_file_filters
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        assert replay_buffer.n_episodes == replay_buffer.n_tasks, 'we assume one episode corresponds to one task'
        if training_split_info is None:
            val_mask = get_val_mask(
                replay_buffer=replay_buffer, sample_type=sample_type, val_ratio=val_ratio, seed=seed
            )
            train_mask = ~val_mask
        else:
            print("[LiberoReplayImageDataset] Using supplied training split info")
            train_mask = get_train_mask_from_training_split_info(replay_buffer, training_split_info, sample_type)

        language_keys = list()
        for key, attr in obs_shape_meta.items():
            if attr.get('is_language', False):
                if not shape_meta.use_prompting or not attr.prompt_type != 'ignore':
                    language_keys.append(key)

        # compute language embedding
        self.using_language = not shape_meta['obs']['task_language'].get('ignore_by_policy', False)

        self.use_finetuned_language_encoder = use_finetuned_language_encoder and self.using_language
        self.text_encoder_model_name = text_encoder_model_name
        if self.use_finetuned_language_encoder:
            # Initialize CLIP tokenizer for tokenizing task names
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
        else:
            self.clip_tokenizer = None


        self.language_key_to_embedding = {}
        if self.using_language and not self.use_finetuned_language_encoder: 
            for language_key in language_keys:
                if language_key == 'task_language':
                    assert sample_type == 'task'
                    self.language_key_to_embedding[language_key] = language_embedding.embed(replay_buffer.task_names)
                else:
                    raise NotImplementedError(f'language key `{language_key}` not supported')

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key and not shape_meta['obs'][key].get('is_language', False):
                self.sampler_lowdim_keys.append(key)

        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.language_keys = language_keys
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.action_padding = action_padding
        self.train_mask = train_mask
        self.sample_type = sample_type
        self.only_prompt = only_prompt
        self.max_segments = max_segments
        self.include_file_filters = include_file_filters
        self.dataset_path = dataset_path
        self.action_rep = shape_meta['action']['rep']
        self.action_key = 'action' if self.action_rep == 'delta' else 'abs_action'

        self.use_prompting = self.shape_meta.use_prompting
    
        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': self.action_padding,
            'sample_type': self.sample_type,
            'only_prompt': self.only_prompt,
            'max_segments': self.max_segments,
            'seed': seed,
            'action_key': self.action_key
        }

        sampler = SequenceSampler(
            mask=self.train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler

    def __repr__(self):
        return f"LiberoReplayImageDataset(dataset_path={self.dataset_path}, include_file_filters={self.include_file_filters}), from cache: {self.use_cache}"

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(
            mask=~self.train_mask,
            **self.sampler_kwargs
        )

        return val_set
    
    def set_new_training_mask(self, mask):
        # this is used in rare cases when we want to re-adjust the training mask 
        # manually, like in a truncated dataset used in ER tests 
        self.train_mask = mask 
        sampler = SequenceSampler(
            mask=self.train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Note that unlike the UMI normalizer, we don't iterate over the dataset itself and instead just directly access values in the replay buffer. This works for LIBERO because we are just directly providing the values from the replay_buffer without any transformations (unlike UMI which does a bunch of conversions to relative trajectories). However, it's probably best to just follow what UMI does rather than making this assumption."""
        normalizer = LinearNormalizer()

        # TODO implement temporally indepent normalization (can follow umi_task_dataset.py). This will require creating a dataloader on this dataset and iterating over the data such that the horizons are included in the data

        # action normalizer
        stat = array_to_stats(self.replay_buffer[self.action_key])
        normalizer["action"] = robomimic_abs_action_only_normalizer_from_stat(stat) # this normalizer is only for position (makes it -1 to 1) and leaves the rotation (already -1 to 1 in rot6d format) and gripper (already -1 to 1). This normalizer works regardless of whether the action representation is delta or absolute.

        # lowdim normalizer (excluding language)
        for key in self.lowdim_keys:
            if key in self.language_keys:
                continue

            stat = array_to_stats(self.replay_buffer[key])

            if key == "ee_pos": # ee_pos is absolute position
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "ee_ori": # ee_ori is absolute rotation in rot6d format so already between -1 and 1
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == "gripper_states": # gripper_states is finger positions [0 to 0.04, -0.04 to 0]
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue  ## skip
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # if not self.use_finetuned_language_encoder:
        # language normalizer
        for key in self.language_keys:
            shape = self.shape_meta['obs'][key]['shape']
            horizon = self.shape_meta['obs'][key]['horizon']
            shape = [horizon] + shape
            stat = {
                'min': np.zeros(shape, dtype=np.float32),
                'max': np.zeros(shape, dtype=np.float32),
            }
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        # image normalizer
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        def prepare_obs_dict(data):
            for key in self.rgb_keys:
                if not key in data:
                    continue
                data[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
                data[key] = np.rot90(data[key], k=2, axes=(2, 3)).copy()
                data[key] = np.flip(data[key], axis=3).copy()

                # resize image to be 224x224
                resize = 224
                data[key] = F.interpolate(
                    torch.tensor(data[key], dtype=torch.float32),
                    size=(resize, resize),
                    mode="bilinear",
                    align_corners=False,
                ).numpy()
            for key in self.sampler_lowdim_keys:
                data[key] = data[key].astype(np.float32)
        
        if (self.rgb_keys + self.lowdim_keys)[0] in data:
            prepare_obs_dict(data)
        if 'prompt' in data:
            prepare_obs_dict(data['prompt']['obs'])

        task_idx = data['metadata']['task_idx']
        if self.using_language:
            if self.use_finetuned_language_encoder:
                # Tokenize task name and pass token IDs (will be encoded by model)
                task_name = self.replay_buffer.task_names[task_idx]
                # Tokenize the task name
                tokens = self.clip_tokenizer(
                    task_name,
                    padding='max_length',
                    truncation=True,
                    max_length=77,  # CLIP's default max length
                    return_tensors='np'
                )
                # Get input_ids
                # Shape will be (1, max_length) for input_ids
                horizon = self.shape_meta['obs']['task_language']['horizon']
                # For horizon > 1, we repeat the tokens
                input_ids = np.tile(tokens['input_ids'], (horizon, 1))  # (horizon, max_length)
                # Store input_ids in task_language (integers can go through dataloader)
                # The model will create attention mask from token IDs (0 is padding)
                data['task_language'] = input_ids.astype(np.int64)  # (horizon, max_length)
            else:
                # Use precomputed embedding
                language_embedding = self.language_key_to_embedding['task_language'][task_idx]
                language_embedding = np.expand_dims(language_embedding, axis=0) # (D) -> (horizon=1, D)
                data['task_language'] = language_embedding


        # language encoding # OLD, REPLACED 
        # for key in self.language_keys:
        #     if self.shape_meta['obs'][key].get('ignore_by_policy', False):
        #         continue

        #     language_embedding = self.language_key_to_embedding[key]
        #     if key == 'task_language':
        #         task_idx = data['prompt']['metadata']['task_indices'][0] if self.use_prompting else data['metadata']['task_idx']
        #         batch_language_embedding = language_embedding[task_idx]
        #     else:
        #         raise NotImplementedError(f'language key `{key}` not supported')
        #     batch_language_embedding = np.expand_dims(batch_language_embedding, axis=0) # (D) -> (horizon=1, D)
        #     data[key] = batch_language_embedding






        # action and metadata
        action = data.pop('action', None)
        metadata = data.pop('metadata', {})

        # convert to torch
        torch_data = {
            "obs": dict_apply(data, torch.from_numpy),
            "metadata": metadata
        }
        if action is not None:
            torch_data['action'] = torch.from_numpy(action.astype(np.float32))
        torch_data["idx"] = idx # used in some methods 
        
        return torch_data

    def shuffle_data_ordering(self, seed:int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def is_multi_task(self) -> bool:
        return self.sample_type == 'task'

    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        return self.sampler.get_unique_task_name_to_dataset_indices()

    def get_training_split_info(self) -> Dict[Tuple[str, str], bool]:
        """Get the dataset information. Returns a dictionary mapping (demonstration_name, task_name) to a boolean indicating whether the dataset is in the training split."""
        return get_training_split_info_from_train_mask(self.replay_buffer, self.sample_type, self.train_mask)

def _convert_actions(raw_actions, rotation_transformer):
    """Convert the rotation in the actions according to `rotation_transformer`."""
    pos = raw_actions[..., :3]
    rot = raw_actions[..., 3:6]
    gripper = raw_actions[..., 6:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    return raw_actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    include_file_filters=[]
):
    """The obs keys in the demos are: ['agentview_rgb', 'ee_pos', 'ee_ori' (axis angle format), 'ee_states' (ee_pos concat with ee_ori), 'eye_in_hand_rgb', 'gripper_states' , 'joint_states']"""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    def _get_chunks(array: np.ndarray):
        chunks = get_optimal_chunks(
            shape=array.shape, dtype=array.dtype, target_chunk_bytes=2e4
        )
        check_chunks_compatible(chunks=chunks, shape=array.shape)
        return chunks

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0
    count_i_to_relative_i = {}
    dataset_paths = glob.glob(dataset_path + "/*.hdf5")

    if include_file_filters:
        dataset_paths = [x for x in dataset_paths if os.path.basename(x).replace('.hdf5', '') in include_file_filters]

    dataset_paths.sort() # sort the paths for consistent dataset ordering

    for dataset_path_each in dataset_paths:
        cur_bddl_path = hdf5_to_bddl(dataset_path_each)
        problem_info = get_problem_info(cur_bddl_path)
        language_goal = problem_info["language_instruction"]

        print(f"[Dataset] Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]

        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count_i_to_relative_i[count] = i
            count += 1
    print("[Dataset] Total demos:", count)

    demos = demos_all
    episode_ends = list()
    task_lengths = list()
    episode_names = list()
    prev_end = 0
    for i in range(len(demos)):
        demo_i_str = f"demo_{i}"
        relative_demo_i_str = f"demo_{count_i_to_relative_i[i]}"
        demo = demos[demo_i_str]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
        task_lengths.append(episode_length)
        episode_names.append(f"{language_all[demo_i_str]} - {relative_demo_i_str}")
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "episode_names",
        episode_names,
        dtype=str,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_names)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_lengths",
        task_lengths,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(task_lengths)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_data_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_data_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )
    _ = meta_group.array(
        "task_labels_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        chunks=_get_chunks(np.asarray(episode_ends)),
        overwrite=True,
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action", "abs_action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "abs_action":
            data_key = "abs_actions"
        if key == "task_language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            this_data.append(demo[data_key][:].astype(np.float32))

            if key == "action":
                this_language_data.append(language_all[f"demo_{i}"])

        this_data = np.concatenate(this_data, axis=0)

        if key == "ee_ori":
            # convert ee_ori from axis angle to specified rotation representation
            this_data = rotation_transformer.forward(this_data).astype(np.float32)

        if key == "action" or key == "abs_action":
            this_data = _convert_actions(
                raw_actions=this_data,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            if key == "action":
                this_language_data = np.array(this_language_data)
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=_get_chunks(this_data),
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = meta_group.array(
                name="task_names",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=_get_chunks(this_language_data),
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="[Dataset] Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = (3, 128, 128)
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    # Add missing fields in
    _ = root.require_group("labels", overwrite=True)

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def print_warn(msg):
    print(f"\033[1;33mWARNING: {msg}\033[0m")

def print_error(msg):
    print(f"\033[1;31mCRITICAL: {msg}\033[0m")


# # wrapper class to load multitask under current framework 
class LiberoMultiTaskDataset():
    def __init__(self,
        **kwargs
    ):
        # we load the subset data 
        self.dataset_kwargs = kwargs
        dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
        dataset_kwargs["use_cache"] = False # prevents loading a dataset cache which contains the full split 
        print("[MultiDataset]: Loading the subset dataset")
        self.dataset = LiberoReplayImageDataset(**dataset_kwargs)
        print(self.dataset)
        self.is_sequential = False 
        self.current_task = "all"


    def load_full_set(self): 
        # we load the full data to compute the normalizer
        
        # if we are already not filtering, no need to load again 
        if len(self.dataset_kwargs["include_file_filters"]) == 0:
            return self.dataset 
        dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
        print("[MultiDataset] Loading the full multitask mixture set (ignoring include_file_filters)")
        dataset_kwargs["include_file_filters"] = [] # Empty so we always load the full set for normalization purposes 
        dataset_kwargs["use_cache"] = False # prevents loading a dataset cache which contains the full split 
        return LiberoReplayImageDataset(**dataset_kwargs) 
    
    def load_pretrain_subset(self):
        filters = self.dataset_kwargs["include_file_filters"]
        print(f"[MultiDataset] Returning single/subset dataset with demos {filters}")
        return self.dataset 


# wrapper class to load a sequence of tasks 
class LiberoSequentialTaskDataset():
    def __init__(self,
        task_order: list[str],
        **kwargs
    ):
        self.task_order = task_order
        self.task_count = 0 
        self.dataset_kwargs = kwargs
        self.is_sequential = True 
        self.current_task = None
        self.training_split_info = None 
        if len(self.dataset_kwargs["include_file_filters"]) != 0:
            print_error("[LiberoSequentialTaskDataset] Not loading a full dataset here could cause normalization issues / mismatch")
        print("[LiberoSequentialTaskDataset] Loading full multitask mixture")
        self.full_set = LiberoReplayImageDataset(**self.dataset_kwargs) # load to get the validation 
        self.training_split_info = self.full_set.get_training_split_info()
    
    def load_full_set(self):
        print("[LiberoSequentialTaskDataset] Returning the full multitask mixture set")
        self.current_task = "all"
        return self.full_set 
    
    def load_subset_tasks(self, task_list):
        dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
        dataset_kwargs["use_cache"] = False # prevents loading a dataset cache which contains the full split 
        dataset_kwargs["include_file_filters"] = task_list
        dataset_kwargs["training_split_info"] = self.training_split_info 
        self.current_task = self.task_order[self.task_count]
        print(f"[LiberoSequentialTaskDataset] Loading single/subset dataset with demos {task_list}")
        dataset = LiberoReplayImageDataset(**dataset_kwargs)
        print(dataset)
        return dataset
    
    def load_custom_training_split_info(self, training_split_info):
        # not guarnteed that we don't accidentally load an incorrect version
        raw_demos = [x[0] for x in self.training_split_info.keys()]
        raw_provided_demos = [x[0] for x in training_split_info.keys()]
        for demo in raw_provided_demos:
            assert demo in raw_demos, "You are specifying a demo that doesn't exist in this dataset."
        if len(self.training_split_info) != len(training_split_info):
            print("[LiberoSequentialTaskDataset] Combining old split info with new (unseen) data splits. This is perfectly ok.")
        new_training_split_info = {k: training_split_info.get(k, self.training_split_info[k]) for k in self.training_split_info}
   
        # this is a little jank but we just reload it with the existing split info. Hard to do otherwise. 
        self.dataset_kwargs["training_split_info"] = new_training_split_info
        self.full_set = LiberoReplayImageDataset(**self.dataset_kwargs) # load to get the validation 
        self.training_split_info = self.full_set.get_training_split_info()
        assert self.training_split_info == new_training_split_info # should have kept it 
        return new_training_split_info 
        
    def next_task(self): # get new task dataset 
        if self.task_count >= len(self.task_order):
            raise ValueError("No more tasks to load")
        
        dataset_kwargs = copy.deepcopy(self.dataset_kwargs)
        dataset_kwargs["use_cache"] = False # prevents loading a dataset cache which contains the full split 
        dataset_kwargs["include_file_filters"] = [self.task_order[self.task_count]]
        dataset_kwargs["training_split_info"] = self.training_split_info 
        self.current_task = self.task_order[self.task_count]
        print(f"[LiberoSequentialTaskDataset] Loading single task dataset with name {self.task_order[self.task_count]}")
        self.task_count += 1
        dataset = LiberoReplayImageDataset(**dataset_kwargs)
        print(dataset)
        return dataset
