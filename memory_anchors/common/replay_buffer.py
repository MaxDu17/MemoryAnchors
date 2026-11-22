# Modified from UMI codebase to support additional fields (specifically 'episode_names' and for subtask labels).

from typing import Union, Dict, Optional, List
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('labels' in root)
        assert('episode_ends' in root['meta'])
        assert('episode_names' in root['meta'])
        assert('task_names' in root['meta'])
        assert('task_lengths' in root['meta'])
        assert('task_data_ends' in root['meta'])
        assert('task_labels_ends' in root['meta'])
        for key, value in root['data'].items():
            upsample_key = f'upsample_index_{key}'
            downsample_key = f'downsample_index_{key}'
            has_upsample = upsample_key in root['meta']
            has_downsample = downsample_key in root['meta']
            assert not (has_downsample and not has_upsample), (
                f"downsample_index_{key} without upsample_index_{key} is invalid"
            )
            if has_upsample and has_downsample:
                ep_ends = root['meta']['episode_ends']
                if len(ep_ends) > 0:
                    assert root['meta'][upsample_key].shape[0] == ep_ends[-1]
                else:
                    assert root['meta'][upsample_key].shape[0] == 0
                assert root['meta'][downsample_key].shape[0] == value.shape[0]
                eek = f'episode_ends_{key}'
                if eek in root['meta']:
                    eends = root['meta'][eek]
                    if len(eends) > 0:
                        assert eends[-1] == value.shape[0]
                    else:
                        assert value.shape[0] == 0
            elif has_upsample:
                assert root['meta'][upsample_key].shape[0] == root['meta']['episode_ends'][-1]
                assert root['meta'][f'episode_ends_{key}'][-1] == value.shape[0]
            else:
                ee = root['meta']['episode_ends']
                assert (value.shape[0] == 0 and len(ee) == 0) or value.shape[0] == ee[-1]
        for key, value in root['labels'].items():
            tle = root['meta']['task_labels_ends']
            assert (value.shape[0] == 0 and len(tle) == 0) or value.shape[0] == tle[-1]
        self.root = root
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        labels = root.require_group('labels', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        if 'episode_names' not in meta:
            episode_names = meta.zeros('episode_names', shape=(0,), dtype=str,
                compressor=None, overwrite=False)
        if 'task_names' not in meta:
            task_names = meta.zeros('task_names', shape=(0,), dtype=str,
                compressor=None, overwrite=False)
        if 'task_lengths' not in meta:
            task_lengths = meta.zeros('task_lengths', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        if 'task_data_ends' not in meta:
            task_data_ends = meta.zeros('task_data_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        if 'task_labels_ends' not in meta:
            task_labels_ends = meta.zeros('task_labels_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64),
                'episode_names': np.zeros((0,), dtype=object),
                'task_names': np.zeros((0,), dtype=object),
                'task_lengths': np.zeros((0,), dtype=np.int64),
                'task_data_ends': np.zeros((0,), dtype=np.int64),
                'task_labels_ends': np.zeros((0,), dtype=np.int64),
            },
            'labels': dict()
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            labels = dict()
            for key, value in src_root['labels'].items():
                if len(value.shape) == 0:
                    labels[key] = np.array(value)
                else:
                    labels[key] = value[:]

            root = {
                'meta': meta,
                'data': data,
                'labels': labels
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)

            # data
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            
            # labels
            labels_group = root.create_group('labels', overwrite=True)
            keys = src_root['labels'].keys()
            for key in keys:
                value = src_root['labels'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/labels/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=labels_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        
        # save labels, chunk
        labels_group = root.create_group('labels', overwrite=True)
        for key, value in self.root['labels'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/labels/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=labels_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = labels_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    @cached_property
    def labels(self):
        return self.root['labels']
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    @property
    def episode_names(self):
        return self.meta['episode_names']

    @property
    def task_names(self):
        return self.meta['task_names']

    @property
    def task_lengths(self):
        return self.meta['task_lengths']

    @property
    def task_data_ends(self):
        return self.meta['task_data_ends']

    @property
    def task_labels_ends(self):
        return self.meta['task_labels_ends']

    @property
    def is_task_error_correction(self):
        """Per-task bool flags, or None if the buffer has no EC metadata."""
        if 'is_task_error_correction' not in self.meta:
            return None
        return self.meta['is_task_error_correction']

    @property
    def is_episode_error_correction(self):
        """Per-episode bool flags, or None if the buffer has no EC metadata."""
        if 'is_episode_error_correction' not in self.meta:
            return None
        return self.meta['is_episode_error_correction']
    
    def get_episode_idxs(self):
        """Provides a list of length equal to the number of episode steps, where each entry is the index of the episode that the step belongs to. So you can do episode_idx = result[data_idx]"""
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)

    def get_task_labels_idxs(self):
        """Provides a list of length equal to the number of labels steps, where each entry is the index of the task that the label belongs to. So you can do task_idx = result[label_idx]"""
        import numba
        numba.jit(nopython=True)
        def _get_task_labels_idxs(labels_ends, task_lengths):
            result = np.zeros((labels_ends[-1],), dtype=np.int64)
            for i in range(len(labels_ends)):
                end = labels_ends[i]
                start = end - task_lengths[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_task_labels_idxs(self.task_labels_ends, self.task_lengths)

    def get_task_to_episode_idxs(self):
        """Provides a list of length equal to the number of tasks, where each entry is the index of the episode that the task belongs to. So you can do episode_idx = result[task_idx]"""
        result = np.zeros((self.n_tasks,), dtype=np.int64)
        data_idx_to_episode_idx = self.get_episode_idxs()
        for task_i in range(self.n_tasks):
            end = self.task_data_ends[task_i]
            start = end - self.task_lengths[task_i]
            result[task_i] = data_idx_to_episode_idx[start]
        return result

    # def get_episode_idx_from_task_idx(self, task_idx):
    #     return self.get_task_to_episode_idxs()[task_idx]
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_labels_steps(self):
        if len(self.task_labels_ends) == 0:
            return 0
        return self.task_labels_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def n_tasks(self):
        return len(self.task_names)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
    
    def is_key_upsampled(self, key):
        return f'upsample_index_{key}' in self.meta
    
    def map_upsample_index(self, key, data_indices):
        """For keys that are upsampled, map the main data indexing to the lower dimensional indexing that the value is stored in"""
        assert self.is_key_upsampled(key)
        return self.meta[f'upsample_index_{key}'][data_indices]
    
    def create_attribute(self, group, attribute, shape_without_batch, dtype):
        starting_shape = (0,) + shape_without_batch
        is_zarr = (self.backend == 'zarr')
        if is_zarr:
            if attribute not in group:
                arr = group.zeros(name=attribute, 
                    shape=starting_shape, 
                    dtype=dtype,
                    compressor=None, overwrite=False)
        else:
            # copy data to prevent modify
            arr = np.zeros(starting_shape, dtype=dtype)
            group[attribute] = arr
        return arr
    
    def append_attribute(self, group, attribute, value):
        """Appends a value to an attribute in meta, resizing the array if necessary."""
        if not isinstance(value, np.ndarray):
            value = np.array([value])

        is_zarr = (self.backend == 'zarr')
        shape_without_batch = value.shape[1:]
        batch_increase = value.shape[0]

        # create array if it doesn't exist
        if attribute not in group:
            self.create_attribute(group, attribute, shape_without_batch, value.dtype)

        # append to episode names
        array = group[attribute]
        new_size = (array.shape[0] + batch_increase, ) + shape_without_batch
        if is_zarr:
            array.resize(new_size)
        else:
            array.resize(new_size, refcheck=False)
        array[-batch_increase:] = value
        
        # rechunk
        if is_zarr:
            if array.chunks[0] < array.shape[0]:
                rechunk_recompress_array(group, attribute, 
                    chunk_length=int(array.shape[0] * 1.5))

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            tasks: Optional[List[Dict]]=None,
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            episode_name: Optional[str]=None,
            upsample_indexing_values: Optional[Dict[str, np.ndarray]]=None,
            upsample_indexing_lengths: Optional[Dict[str, int]]=None,
            downsample_indexing_values: Optional[Dict[str, np.ndarray]]=None,
            ):
        if tasks is None:
            tasks = []
        if upsample_indexing_values is None:
            upsample_indexing_values = {}
        if upsample_indexing_lengths is None:
            upsample_indexing_lengths = {}
        if downsample_indexing_values is None:
            downsample_indexing_values = {}

        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        has_upsample = len(upsample_indexing_values) > 0
        has_downsample = len(downsample_indexing_values) > 0
        has_lengths = len(upsample_indexing_lengths) > 0

        if has_upsample:
            assert has_lengths, (
                "upsample_indexing_values requires upsample_indexing_lengths with the same keys"
            )
            assert set(upsample_indexing_values.keys()) == set(upsample_indexing_lengths.keys()), (
                f"upsample keys {set(upsample_indexing_values.keys())} != lengths keys {set(upsample_indexing_lengths.keys())}"
            )
        if has_downsample:
            assert has_upsample and has_lengths, (
                "downsample_indexing_values requires upsample_indexing_values and upsample_indexing_lengths"
            )
            assert set(upsample_indexing_values.keys()) == set(downsample_indexing_values.keys()), (
                f"upsample keys {set(upsample_indexing_values.keys())} != downsample keys {set(downsample_indexing_values.keys())}"
            )

        non_upsampled_keys = [key for key in data.keys() if key not in upsample_indexing_values]
        assert len(non_upsampled_keys) > 0, "add_episode needs at least one main-rate data key"
        episode_length = len(data[non_upsampled_keys[0]])

        curr_len = self.n_steps
        for key, value in data.items():
            assert len(value.shape) >= 1
            if key in non_upsampled_keys:
                assert episode_length == len(value)
        new_len = curr_len + episode_length

        meta = self.meta

        for key, value in data.items():
            if key in upsample_indexing_values:
                if key not in self.data:
                    new_shape = (len(value),) + value.shape[1:]
                else:
                    orig_len = self.data[key].shape[0]
                    new_shape = (orig_len + len(value),) + value.shape[1:]
            else:
                new_shape = (new_len,) + value.shape[1:]
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            arr[-value.shape[0]:] = value
        
        self.append_attribute(meta, 'episode_ends', new_len)
        self.append_attribute(meta, 'episode_names', episode_name if episode_name is not None else f'episode_{self.n_episodes-1}')

        for upsample_key_name, upsample_mapping in upsample_indexing_values.items():
            metadata_value_name = f'upsample_index_{upsample_key_name}'
            metadata_ends_name = f'episode_ends_{upsample_key_name}'
            if metadata_value_name in meta and len(meta[metadata_ends_name]) > 0:
                previous_episode_end = meta[metadata_ends_name][-1]
                upsample_mapping = upsample_mapping + previous_episode_end
            else:
                previous_episode_end = 0
            new_episode_end = previous_episode_end + upsample_indexing_lengths[upsample_key_name]
            self.append_attribute(meta, metadata_value_name, upsample_mapping)
            self.append_attribute(meta, metadata_ends_name, new_episode_end)

        for downsample_key_name, downsample_mapping in downsample_indexing_values.items():
            metadata_value_name = f'downsample_index_{downsample_key_name}'
            metadata_ends_name = 'episode_ends'
            if metadata_value_name in meta and len(meta[metadata_ends_name]) > 1:
                previous_episode_end = meta[metadata_ends_name][-2]
                downsample_mapping = downsample_mapping + previous_episode_end
            self.append_attribute(meta, metadata_value_name, downsample_mapping)

        new_episode_label_keys = set()
        for task in tasks:
            if "labels" in task and task["labels"]:
                new_episode_label_keys.update(task["labels"].keys())
        existing_label_keys = set(self.labels.keys())
        if len(existing_label_keys) > 0 and len(new_episode_label_keys) == 0:
            raise ValueError(
                f"Label keys mismatch: Buffer has labels with keys {sorted(existing_label_keys)}, "
                f"but new episode has no labels. All episodes must have the same set of label keys."
            )
        elif len(existing_label_keys) > 0 and len(new_episode_label_keys) > 0:
            if new_episode_label_keys != existing_label_keys:
                missing_keys = existing_label_keys - new_episode_label_keys
                extra_keys = new_episode_label_keys - existing_label_keys
                error_msg = f"Label keys mismatch: Expected {sorted(existing_label_keys)}, "
                error_msg += f"got {sorted(new_episode_label_keys)}."
                if missing_keys:
                    error_msg += f" Missing keys: {sorted(missing_keys)}."
                if extra_keys:
                    error_msg += f" Extra keys: {sorted(extra_keys)}."
                raise ValueError(error_msg)

        for task in tasks:
            task_name = task["name"]
            local_start_idx = task["start_idx"]
            local_end_idx = task["end_idx"]
            labels = task["labels"]

            assert local_end_idx <= episode_length and local_start_idx >= 0, 'task range must fit within episode bounds'

            task_length = local_end_idx - local_start_idx
            new_labels_len = self.n_labels_steps + task_length

            self.append_attribute(meta, 'task_names', task_name)
            self.append_attribute(meta, 'task_lengths', task_length)
            self.append_attribute(meta, 'task_data_ends', curr_len + local_end_idx)
            self.append_attribute(meta, 'task_labels_ends', new_labels_len)

            for lkey, lvalue in labels.items():
                assert len(lvalue) == task_length

                if type(lvalue) == list:
                    lvalue = np.array(lvalue)

                if len(lvalue.shape) == 1:
                    lvalue = np.expand_dims(lvalue, axis=1)

                self.append_attribute(self.labels, lkey, lvalue)

    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert len(episode_ends) > 0
        new_data_end_idx = 0
        if len(episode_ends) > 1:
            new_data_end_idx = episode_ends[-2]

        regular_keys = []
        sampled_keys = []
        for key in self.data.keys():
            if self.is_key_upsampled(key):
                sampled_keys.append(key)
            else:
                regular_keys.append(key)

        for key in regular_keys:
            value = self.data[key]
            new_shape = (new_data_end_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)

        for key in sampled_keys:
            episode_ends_key = f'episode_ends_{key}'
            sampled_episode_ends = self.meta[episode_ends_key][:].copy()
            new_sampled_end_idx = 0
            if len(sampled_episode_ends) > 1:
                new_sampled_end_idx = sampled_episode_ends[-2]
            value = self.data[key]
            new_shape = (new_sampled_end_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)

            upsample_index_key = f'upsample_index_{key}'
            if upsample_index_key in self.meta:
                upsample_index = self.meta[upsample_index_key]
                new_upsample_shape = (new_data_end_idx,) + upsample_index.shape[1:]
                if is_zarr:
                    upsample_index.resize(new_upsample_shape)
                else:
                    upsample_index.resize(new_upsample_shape, refcheck=False)

            downsample_index_key = f'downsample_index_{key}'
            if downsample_index_key in self.meta:
                downsample_index = self.meta[downsample_index_key]
                new_downsample_shape = (new_sampled_end_idx,) + downsample_index.shape[1:]
                if is_zarr:
                    downsample_index.resize(new_downsample_shape)
                else:
                    downsample_index.resize(new_downsample_shape, refcheck=False)

            if is_zarr:
                self.meta[episode_ends_key].resize(len(sampled_episode_ends) - 1)
            else:
                self.meta[episode_ends_key].resize(len(sampled_episode_ends) - 1, refcheck=False)

        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
            self.episode_names.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)
            self.episode_names.resize(len(episode_ends) - 1, refcheck=False)

        tasks_to_drop = []
        for i in range(len(self.task_data_ends) - 1, -1, -1):
            if self.task_data_ends[i] > new_data_end_idx:
                tasks_to_drop.append(i)
            else:
                break

        for _ in tasks_to_drop:
            new_labels_end_idx = 0
            if len(self.task_labels_ends) > 1:
                new_labels_end_idx = self.task_labels_ends[-2]
            for lkey, lvalue in self.labels.items():
                new_shape = (new_labels_end_idx,) + lvalue.shape[1:]
                if is_zarr:
                    lvalue.resize(new_shape)
                else:
                    lvalue.resize(new_shape, refcheck=False)
            if is_zarr:
                self.task_names.resize(len(self.task_names) - 1)
                self.task_lengths.resize(len(self.task_lengths) - 1)
                self.task_data_ends.resize(len(self.task_data_ends) - 1)
                self.task_labels_ends.resize(len(self.task_labels_ends) - 1)
            else:
                self.task_names.resize(len(self.task_names) - 1, refcheck=False)
                self.task_lengths.resize(len(self.task_lengths) - 1, refcheck=False)
                self.task_data_ends.resize(len(self.task_data_ends) - 1, refcheck=False)
                self.task_labels_ends.resize(len(self.task_labels_ends) - 1, refcheck=False)

    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False, data_keys=None, labels_keys=None):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]

        all_data_keys = list(self.data.keys()) if data_keys is None else data_keys
        upsampled_keys = []
        non_upsampled_keys = []
        for key in all_data_keys:
            if self.is_key_upsampled(key):
                upsampled_keys.append(key)
            else:
                non_upsampled_keys.append(key)

        data = self.get_steps_slice(start_idx, end_idx, copy=copy, keep_keys=non_upsampled_keys)

        for key in upsampled_keys:
            episode_ends_key = f'episode_ends_{key}'
            episode_start_upsample = 0
            if idx > 0:
                episode_start_upsample = self.meta[episode_ends_key][idx - 1]
            episode_end_upsample = self.meta[episode_ends_key][idx]
            value = self.data[key]
            x = value[episode_start_upsample:episode_end_upsample]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            data[key] = x

        episode_name = self.episode_names[idx]

        tasks = []
        for task_idx in range(len(self.task_data_ends)):
            task_end_data_idx = self.task_data_ends[task_idx]
            task_start_data_idx = task_end_data_idx - self.task_lengths[task_idx]
            if task_start_data_idx >= start_idx and task_end_data_idx <= end_idx:
                local_start_idx = task_start_data_idx - start_idx
                local_end_idx = task_end_data_idx - start_idx
                task_name = self.task_names[task_idx]
                task_end_label_idx = self.task_labels_ends[task_idx]
                task_start_label_idx = task_end_label_idx - self.task_lengths[task_idx]
                labels = self.get_labels_steps_slice(
                    task_start_label_idx, task_end_label_idx, copy=copy, keep_keys=labels_keys
                )
                tasks.append({
                    "name": task_name,
                    "start_idx": local_start_idx,
                    "end_idx": local_end_idx,
                    "labels": labels,
                })

        upsample_indexing_values = {}
        upsample_indexing_lengths = {}
        for mkey in self.meta.keys():
            if mkey.startswith('upsample_index_'):
                upsample_key_name = mkey[len('upsample_index_'):]
                episode_ends_key = f'episode_ends_{upsample_key_name}'
                if episode_ends_key in self.meta:
                    full_mapping = self.meta[mkey][:]
                    episode_mapping = full_mapping[start_idx:end_idx]
                    episode_start_upsample = 0
                    if idx > 0:
                        episode_start_upsample = self.meta[episode_ends_key][idx - 1]
                    episode_end_upsample = self.meta[episode_ends_key][idx]
                    local_mapping = episode_mapping - episode_start_upsample
                    upsample_length = episode_end_upsample - episode_start_upsample
                    upsample_indexing_values[upsample_key_name] = local_mapping
                    upsample_indexing_lengths[upsample_key_name] = upsample_length

        downsample_indexing_values = {}
        for mkey in self.meta.keys():
            if mkey.startswith('downsample_index_'):
                downsample_key_name = mkey[len('downsample_index_'):]
                full_mapping = self.meta[mkey][:]
                if self.is_key_upsampled(downsample_key_name):
                    episode_ends_key = f'episode_ends_{downsample_key_name}'
                    episode_start_downsample = 0
                    if idx > 0:
                        episode_start_downsample = self.meta[episode_ends_key][idx - 1]
                    episode_end_downsample = self.meta[episode_ends_key][idx]
                    episode_mapping = full_mapping[episode_start_downsample:episode_end_downsample]
                    local_mapping = episode_mapping - start_idx
                else:
                    episode_start_downsample = 0
                    if idx > 0:
                        episode_start_downsample = self.episode_ends[idx - 1]
                    episode_end_downsample = self.episode_ends[idx]
                    episode_mapping = full_mapping[episode_start_downsample:episode_end_downsample]
                    local_mapping = episode_mapping - start_idx
                downsample_indexing_values[downsample_key_name] = local_mapping

        return {
            'data': data,
            'tasks': tasks,
            'episode_name': episode_name,
            'upsample_indexing_values': upsample_indexing_values,
            'upsample_indexing_lengths': upsample_indexing_lengths,
            'downsample_indexing_values': downsample_indexing_values,
        }

    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False, keep_keys=None):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            if keep_keys is not None and key not in keep_keys:
                continue
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    def get_task(self, idx, copy=False):
        """given a task index, return the data and labels associated with that task"""
        # get the data entries
        end_data_idx = self.task_data_ends[idx]
        start_data_idx = end_data_idx - self.task_lengths[idx]
        data_slice = self.get_steps_slice(start_data_idx, end_data_idx, copy=copy)

        # get the label entries
        end_label_idx = self.task_labels_ends[idx]
        start_label_idx = end_label_idx - self.task_lengths[idx]
        label_slice = self.get_labels_steps_slice(start_label_idx, end_label_idx, copy=copy)

        # combine the result
        result = dict()
        result['data'] = data_slice
        result['labels'] = label_slice

        return result

    def get_labels_slice(self, idx):
        end_label_idx = self.task_labels_ends[idx]
        start_label_idx = end_label_idx - self.task_lengths[idx]
        return slice(start_label_idx, end_label_idx)

    def get_labels_steps_slice(self, start, stop, step=None, copy=False, keep_keys=None):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.labels.items():
            if keep_keys is not None and key not in keep_keys:
                continue
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x

        return result

    def get_episode_name(self, episode_idx):
        return self.episode_names[episode_idx]
    
    def get_task_name(self, task_idx):
        return self.task_names[task_idx]
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
