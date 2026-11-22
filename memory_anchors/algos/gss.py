"""Gradient-based Sample Selection (GSS-Greedy) experience replay.

Task-segmented adaptation of Aljundi et al. (NeurIPS 2019) Algorithm 2:
at each task boundary, stream the newly finished task through greedy
gradient-cosine replacement into a fixed replay buffer. Training uses
standard rehearsal (not GEM-style projection).

Reference: https://github.com/rahafaljundi/Gradient-based-Sample-Selection
Paper: https://arxiv.org/abs/1903.08671
"""

from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.algos.base import CL_BasePolicy
from memory_anchors.common.torch_util import move_batch_to_device
from memory_anchors.dataset.truncated_datasets import CustomTruncatedDataset

import collections
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import tqdm


def cycle(dl):
    while True:
        for data in dl:
            yield data


def merge_datas(x, y, device=None):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()
        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k], device)
        return new_x
    elif isinstance(x, torch.Tensor):
        return torch.cat([x, y], 0)
    else:
        raise TypeError(f"[GSS] merge_datas unsupported type: {type(x)}")


def merge_batch_of_dicts(batch_list, device=None):
    """Stack a list of single-example dicts into one batch dict."""
    first = batch_list[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(batch_list, 0)
    elif isinstance(first, (dict, collections.OrderedDict)):
        if isinstance(first, dict):
            merged = dict()
        else:
            merged = collections.OrderedDict()
        for k in first.keys():
            merged[k] = merge_batch_of_dicts([b[k] for b in batch_list], device=device)
        return merged
    else:
        return torch.tensor([b for b in batch_list])


def get_grad_vector(model) -> torch.Tensor:
    """Flatten parameter gradients into one vector (GSS reference get_grad_vector)."""
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, "[GSS] expected grad on a requires_grad parameter"
        grads.append(param.grad.detach().reshape(-1))
    assert len(grads) > 0, "[GSS] no gradients found"
    return torch.cat(grads)


class GSSEpisodicReplay(CL_BasePolicy):
    def __init__(
        self,
        base_learner: BasePolicy,
        filter_size,
        filter_size_mode,
        batch_size,
        num_workers,
        er_weight,
        gss_cfg,
        preload_dataset=None,
        add_new_data=True,
    ):
        super().__init__(base_learner)
        self.full_datasets = list()
        self.last_dataset = None
        self.er_weight = er_weight
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter_size = filter_size
        self.filter_size_mode = filter_size_mode
        self.gss_cfg = gss_cfg

        self.pretrain_buffer = None
        self.buffer = None
        self._buffer_loader = None

        # Persistent GSS buffer state (global indices into ConcatDataset of past tasks).
        self.buffer_indices = None  # np.ndarray[int64] or None
        self.buffer_scores = None  # np.ndarray[float64], +1-shifted cosine scores

        if preload_dataset is not None:
            print("[GSS] Loading the pretrained dataset.")
            self.last_dataset = preload_dataset.load_pretrain_subset()

        self.add_new_data = add_new_data
        assert add_new_data or preload_dataset is not None, \
            "[GSS] you need current-task data if not regularizing to a preloaded dataset"
        if not self.add_new_data:
            print("[GSS] This instantiation keeps only the pretrained set. Are you sure?")

        assert filter_size_mode in ("prop", "examples"), \
            f"[GSS] filter_size_mode must be 'prop' or 'examples', got '{filter_size_mode}'"
        if filter_size_mode == "prop":
            assert 0 < filter_size <= 1, \
                f"[GSS] filter_size must be in (0, 1] when filter_size_mode='prop', got {filter_size}"
        else:
            assert isinstance(filter_size, (int, float)) and filter_size > 0, \
                f"[GSS] filter_size must be positive when filter_size_mode='examples', got {filter_size}"
            assert int(filter_size) == filter_size, \
                f"[GSS] filter_size must be a whole number when filter_size_mode='examples', got {filter_size}"

        n_grad = int(gss_cfg.n_grad_compare)
        assert n_grad >= 1, f"[GSS] n_grad_compare must be >= 1, got {n_grad}"
        max_new = gss_cfg.max_new_examples
        if max_new is not None:
            assert int(max_new) > 0, \
                f"[GSS] max_new_examples must be positive when set, got {max_new}"

    def _get_buffer_size(self, buffer_len: int) -> int:
        if self.filter_size_mode == "prop":
            size = int(self.filter_size * buffer_len)
            size = max(size, self.batch_size)
        else:
            size = min(int(self.filter_size), buffer_len)
        return min(size, buffer_len)

    def _task_nametag(self, current_dataset) -> str:
        if hasattr(current_dataset, "include_file_filters") and current_dataset.include_file_filters is not None:
            nametag = "".join(current_dataset.include_file_filters)
        else:
            nametag = f"task{len(self.full_datasets) + 1}"
        if len(nametag) > 150:
            nametag = nametag[:150]
        return nametag

    def _example_batch(self, dataset, global_idx: int):
        sample = dataset[int(global_idx)]
        batch = merge_batch_of_dicts([sample])
        return move_batch_to_device(batch, self.base_model.device)

    def _compute_example_grad(self, dataset, global_idx: int) -> torch.Tensor:
        """Per-example parameter gradient of the training loss."""
        model = self.accelerator.unwrap_model(self.base_model)
        model.zero_grad(set_to_none=True)
        batch = self._example_batch(dataset, global_idx)
        loss = model(batch)
        assert torch.isfinite(loss).all(), f"[GSS] non-finite loss at index {global_idx}"
        loss.backward()
        grad = get_grad_vector(model)
        assert torch.isfinite(grad).all(), f"[GSS] non-finite grad at index {global_idx}"
        assert grad.norm() > 0, f"[GSS] zero grad at index {global_idx}"
        model.zero_grad(set_to_none=True)
        return grad

    def _max_cosine_score(self, candidate_grad: torch.Tensor, ref_grads: torch.Tensor) -> float:
        """Paper Algo 2: c = max_i cos(g, G_i) + 1 (shift to make positive)."""
        eps = float(self.gss_cfg.score_eps)
        assert ref_grads.ndim == 2 and ref_grads.shape[0] > 0, "[GSS] need at least one reference grad"
        g = candidate_grad / (candidate_grad.norm() + eps)
        G = ref_grads / (ref_grads.norm(dim=1, keepdim=True) + eps)
        cos = torch.mv(G, g)
        c = float(cos.max().item()) + 1.0
        assert c >= 0.0, f"[GSS] score should be non-negative after +1 shift, got {c}"
        return c

    def _sample_replace_index(self, scores: np.ndarray) -> int:
        """Sample replacee i with probability proportional to C_i (min-max normalized)."""
        assert scores.size > 0, "[GSS] empty score vector"
        s = scores.astype(np.float64)
        s_min = s.min()
        s_max = s.max()
        # Match reference GSS_Greedy normalization for multinomial stability.
        probs = (s - s_min) / ((s_max - s_min) + 0.01)
        probs = np.maximum(probs, 1e-8)
        probs = probs / probs.sum()
        return int(np.random.choice(len(scores), p=probs))

    def _greedy_stream_new_task(self, new_task_global_indices: np.ndarray, capacity: int):
        """Stream finished-task indices through GSS-Greedy Algo 2 into the persistent buffer."""
        assert capacity > 0, "[GSS] capacity must be positive"
        dataset = self.pretrain_buffer
        n_compare = int(self.gss_cfg.n_grad_compare)

        if self.buffer_indices is None:
            buf_idx = []
            buf_scores = []
        else:
            buf_idx = list(np.asarray(self.buffer_indices, dtype=np.int64).tolist())
            buf_scores = list(np.asarray(self.buffer_scores, dtype=np.float64).tolist())

        # If prop-mode capacity shrank, drop highest-similarity (least diverse) slots first.
        if len(buf_idx) > capacity:
            order = np.argsort(np.asarray(buf_scores))  # ascending: keep most diverse
            keep = order[:capacity]
            buf_idx = [buf_idx[i] for i in keep]
            buf_scores = [buf_scores[i] for i in keep]

        model = self.accelerator.unwrap_model(self.base_model)
        was_training = model.training
        model.eval()

        order = np.asarray(new_task_global_indices, dtype=np.int64).copy()
        np.random.shuffle(order)

        replacements = 0
        additions = 0

        no_sync_ctx = (
            self.accelerator.no_sync(self.base_model)
            if self.accelerator is not None
            else torch.nullcontext()
        )

        with no_sync_ctx:
            for global_idx in tqdm.tqdm(order, desc="[GSS] greedy selection", leave=True):
                if len(buf_idx) == 0:
                    # First insertion: no references yet (reference code uses default 0.1;
                    # with +1 shift, cos=0 -> score 1.0).
                    buf_idx.append(int(global_idx))
                    buf_scores.append(1.0)
                    additions += 1
                    continue

                n_ref = min(n_compare, len(buf_idx))
                ref_local = np.random.choice(len(buf_idx), size=n_ref, replace=False)
                ref_grads = []
                for loc in ref_local:
                    ref_grads.append(self._compute_example_grad(dataset, buf_idx[int(loc)]))
                ref_grads = torch.stack(ref_grads, dim=0)

                cand_grad = self._compute_example_grad(dataset, int(global_idx))
                c = self._max_cosine_score(cand_grad, ref_grads)

                if len(buf_idx) < capacity:
                    buf_idx.append(int(global_idx))
                    buf_scores.append(c)
                    additions += 1
                else:
                    # Paper: only attempt replacement when max cosine < 0 (c < 1 after +1).
                    if c < 1.0:
                        i = self._sample_replace_index(np.asarray(buf_scores, dtype=np.float64))
                        replace_prob = buf_scores[i] / (buf_scores[i] + c)
                        if np.random.rand() < replace_prob:
                            buf_idx[i] = int(global_idx)
                            buf_scores[i] = c
                            replacements += 1

        if was_training:
            model.train()

        self.buffer_indices = np.asarray(buf_idx, dtype=np.int64)
        self.buffer_scores = np.asarray(buf_scores, dtype=np.float64)
        assert len(self.buffer_indices) == len(self.buffer_scores)
        assert len(self.buffer_indices) <= capacity
        self.accelerator.print(
            f"[GSS] Buffer size={len(self.buffer_indices)}/{capacity} "
            f"(additions={additions}, replacements={replacements}, "
            f"streamed={len(order)})"
        )

    def start_task(self, current_dataset, info_cache_dir):
        self.accelerator.print("[GSS] Starting Task")
        assert self.accelerator is not None, "[GSS] accelerator must be registered"

        if self.last_dataset is not None:
            self.full_datasets.append(self.last_dataset)

        if self.add_new_data:
            self.last_dataset = current_dataset
        else:
            self.last_dataset = None

        nametag = self._task_nametag(current_dataset)

        if len(self.full_datasets) == 0:
            print("[GSS] No past datasets. Proceeding with an empty buffer.")
            self.pretrain_buffer = None
            self.buffer = None
            self._buffer_loader = None
            return

        self.pretrain_buffer = ConcatDataset(self.full_datasets)
        new_task = self.full_datasets[-1]
        new_offset = len(self.pretrain_buffer) - len(new_task)
        new_task_indices = np.arange(new_offset, new_offset + len(new_task), dtype=np.int64)

        max_new = self.gss_cfg.max_new_examples
        if max_new is not None and len(new_task_indices) > int(max_new):
            new_task_indices = np.random.choice(
                new_task_indices, size=int(max_new), replace=False
            )
            self.accelerator.print(
                f"[GSS] Subsampled finished task to {int(max_new)} examples for selection"
            )

        capacity = self._get_buffer_size(len(self.pretrain_buffer))
        self.accelerator.print(
            f"[GSS] Past datasets={len(self.full_datasets)}, "
            f"backing store={len(self.pretrain_buffer)}, "
            f"new-task stream={len(new_task_indices)}, capacity={capacity}"
        )

        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self._greedy_stream_new_task(new_task_indices, capacity=capacity)
                assert self.buffer_indices is not None and len(self.buffer_indices) > 0, \
                    "[GSS] selection produced an empty buffer"
                np.save(
                    os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"),
                    self.buffer_indices,
                )
                np.save(
                    os.path.join(info_cache_dir, f"gss_scores_{nametag}.npy"),
                    self.buffer_scores,
                )
                self.accelerator.print(
                    f"[GSS] Saved selected_indices_{nametag}.npy "
                    f"({len(self.buffer_indices)} indices)"
                )
            else:
                print("[GSS] Alternate process, waiting for buffer selection")

            self.accelerator.wait_for_everyone()
            selected_indices = np.load(
                os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"),
                mmap_mode="r",
            )
            self.buffer_indices = np.asarray(selected_indices, dtype=np.int64).copy()
            scores_path = os.path.join(info_cache_dir, f"gss_scores_{nametag}.npy")
            self.buffer_scores = np.asarray(np.load(scores_path), dtype=np.float64).copy()
            assert len(self.buffer_indices) == len(self.buffer_scores)

            trunc_set = CustomTruncatedDataset(
                self.pretrain_buffer, allowed_indices=self.buffer_indices.tolist()
            )
            assert len(trunc_set) == len(self.buffer_indices), \
                "[GSS] truncated dataset length mismatch"
            self.accelerator.print(
                f"[GSS] Original past size: {len(self.pretrain_buffer)}. "
                f"ER Buffer Size: {len(trunc_set)}"
            )

        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
            self._buffer_loader = None

        self._buffer_loader = DataLoader(
            trunc_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
            shuffle=True,
        )
        self.buffer = cycle(self._buffer_loader)
        self.accelerator.wait_for_everyone()

    def end_task(self):
        pass

    def log_info(self, output_dir):
        pass

    def register_training_progress(self, proportion):
        pass

    def simple_er_call(self, batch):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            buf_data = move_batch_to_device(buf_data, self.base_model.device)
            combined_batch = merge_datas(buf_data, batch, device=self.base_model.device)
            loss_vec = self.base_model(combined_batch, reduce_loss=False)
            er_bs = next(iter(buf_data["obs"].values())).shape[0]
            er_loss = loss_vec[:er_bs].mean()
            task_loss = loss_vec[er_bs:].mean()
        else:
            task_loss = self.base_model(batch)
            er_loss = task_loss * 0.0

        raw_loss = self.er_weight * er_loss + task_loss
        return (
            {"raw_loss": raw_loss, "er_loss": er_loss, "current_task_loss": task_loss},
            {"scalars": None, "distrs": None},
        )

    def __call__(self, batch):
        return self.simple_er_call(batch)
