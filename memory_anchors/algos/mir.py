"""Maximally Interfered Retrieval Experience Replay (MIR-ER).

At task start: build a random ER buffer from past-task data (same as AnchorER random_baseline).
During training: virtual one-step SGD on the current batch, score a buffer subsample by
loss increase, rehearse the top-k most interfered samples mixed with the current batch.

Reference: https://github.com/optimass/Maximally_Interfered_Retrieval
"""

from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.algos.base import CL_BasePolicy
from memory_anchors.common.torch_util import move_batch_to_device
from memory_anchors.dataset.truncated_datasets import CustomTruncatedDataset

import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset


def merge_datas(x, y, device=None):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()
        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k], device)
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) \
            or isinstance(x, torch.cuda.FloatTensor) or isinstance(x, torch.cuda.LongTensor):
        return torch.cat([x, y], 0)
    else:
        raise TypeError(f"[MIRER] merge_datas unsupported type: {type(x)}")


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


def _per_example_loss(loss_tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a possibly multi-dim per-sample loss to shape [B]."""
    assert loss_tensor.ndim >= 1, "[MIRER] expected at least a batch dimension on loss"
    if loss_tensor.ndim == 1:
        return loss_tensor
    return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)


class MIREpisodicReplay(CL_BasePolicy):
    def __init__(
        self,
        base_learner: BasePolicy,
        filter_size,
        filter_size_mode,
        batch_size,
        num_workers,
        er_weight,
        mir_cfg,
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
        self.mir_cfg = mir_cfg

        self.er_dataset = None
        self.er_global_indices = None  # pretrain_buffer-global index per er_dataset row
        self.pretrain_buffer = None

        self._mir_step = 0
        self._selection_counts = collections.Counter()
        self._info_cache_dir = None
        self._current_nametag = None
        self._report_written = False

        if preload_dataset is not None:
            print("[MIRER] Loading the pretrained dataset.")
            self.last_dataset = preload_dataset.load_pretrain_subset()

        self.add_new_data = add_new_data
        assert add_new_data or preload_dataset is not None, \
            "[MIRER] you need current-task data if not regularizing to a preloaded dataset"
        if not self.add_new_data:
            print("[MIRER] This instantiation keeps only the pretrained set. Are you sure?")

        assert filter_size_mode in ("prop", "examples"), \
            f"[MIRER] filter_size_mode must be 'prop' or 'examples', got '{filter_size_mode}'"
        if filter_size_mode == "prop":
            assert 0 < filter_size <= 1, \
                f"[MIRER] filter_size must be in (0, 1] when filter_size_mode='prop', got {filter_size}"
        else:
            assert isinstance(filter_size, (int, float)) and filter_size > 0, \
                f"[MIRER] filter_size must be positive when filter_size_mode='examples', got {filter_size}"
            assert int(filter_size) == filter_size, \
                f"[MIRER] filter_size must be a whole number when filter_size_mode='examples', got {filter_size}"

        assert mir_cfg.subsample >= mir_cfg.retrieve_size, \
            f"[MIRER] subsample ({mir_cfg.subsample}) must be >= retrieve_size ({mir_cfg.retrieve_size})"
        assert mir_cfg.retrieve_size > 0, "[MIRER] retrieve_size must be positive"
        assert mir_cfg.virtual_lr > 0, "[MIRER] virtual_lr must be positive"
        assert mir_cfg.log_every >= 1, "[MIRER] log_every must be >= 1"

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

    def start_task(self, current_dataset, info_cache_dir):
        self.accelerator.print("[MIRER] Starting Task")
        self._info_cache_dir = info_cache_dir
        self._selection_counts = collections.Counter()
        self._mir_step = 0
        self._report_written = False

        if self.last_dataset is not None:
            self.full_datasets.append(self.last_dataset)

        if self.add_new_data:
            self.last_dataset = current_dataset
        else:
            self.last_dataset = None

        nametag = self._task_nametag(current_dataset)
        self._current_nametag = nametag

        if len(self.full_datasets) == 0:
            print("[MIRER] No past datasets. Proceeding with an empty buffer.")
            self.er_dataset = None
            self.er_global_indices = None
            self.pretrain_buffer = None
            return

        self.accelerator.print(f"[MIRER] There are {len(self.full_datasets)} past datasets.")
        self.pretrain_buffer = ConcatDataset(self.full_datasets)

        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                num_indices = self._get_buffer_size(len(self.pretrain_buffer))
                selected_indices = np.random.choice(
                    len(self.pretrain_buffer), size=num_indices, replace=False
                )
                self.accelerator.print(
                    f"\t [MIRER] Random ER buffer of {num_indices} from "
                    f"{len(self.pretrain_buffer)} past examples"
                )
                np.save(
                    os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"),
                    selected_indices,
                )
            else:
                print("[MIRER] Alternate process, waiting for buffer selection")

            self.accelerator.wait_for_everyone()
            selected_indices = np.load(
                os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"),
                mmap_mode="r",
            )
            expected_len = self._get_buffer_size(len(self.pretrain_buffer))
            assert abs(len(selected_indices) - expected_len) < 2, \
                "[MIRER] truncated buffer size does not match requested filter size"

            self.er_global_indices = np.asarray(selected_indices, dtype=np.int64).copy()
            self.er_dataset = CustomTruncatedDataset(
                self.pretrain_buffer, allowed_indices=self.er_global_indices.tolist()
            )
            self.accelerator.print(
                f"[MIRER] Original past size: {len(self.pretrain_buffer)}. "
                f"ER Buffer Size: {len(self.er_dataset)}"
            )

        self.accelerator.wait_for_everyone()

    def end_task(self):
        self._write_selection_report()

    def log_info(self, output_dir):
        if self._info_cache_dir is None:
            self._info_cache_dir = output_dir
        self._write_selection_report()

    def _write_selection_report(self):
        if self._report_written:
            return
        if self._info_cache_dir is None or self._current_nametag is None:
            return
        if not self.accelerator.is_main_process:
            return
        if len(self._selection_counts) == 0:
            if self.er_dataset is None:
                self.accelerator.print(
                    "[MIRER] No ER buffer this task (first task / empty past); skipping selection report."
                )
            else:
                self.accelerator.print(
                    f"[MIRER] ER buffer present but zero logged selections "
                    f"(mir_steps={self._mir_step}, log_every={self.mir_cfg.log_every}); skipping report."
                )
            self._report_written = True
            return

        nametag = self._current_nametag
        counts = self._selection_counts
        # JSON: global_index -> count, sorted by count descending
        sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        report = {str(int(idx)): int(cnt) for idx, cnt in sorted_items}
        json_path = os.path.join(self._info_cache_dir, f"mir_selection_{nametag}.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        arr = np.array([[int(idx), int(cnt)] for idx, cnt in sorted_items], dtype=np.int64)
        np.save(os.path.join(self._info_cache_dir, f"mir_selection_{nametag}.npy"), arr)

        vis_dir = os.path.join(self._info_cache_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.hist([cnt for _, cnt in sorted_items], bins=30, alpha=0.7, color="steelblue")
        plt.xlabel("Selection count")
        plt.ylabel("Number of buffer indices")
        plt.title(f"MIR selection frequency ({nametag})")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{nametag}_mir_selection_hist.png"))
        plt.close()

        self.accelerator.print(
            f"[MIRER] Wrote selection report ({len(report)} unique indices) to {json_path}"
        )
        self._report_written = True

    def register_training_progress(self, proportion):
        pass

    def _sample_candidate_batch(self):
        """Uniformly sample up to mir_cfg.subsample examples from the ER buffer."""
        n = len(self.er_dataset)
        assert n > 0, "[MIRER] ER dataset is empty"
        subsample = min(int(self.mir_cfg.subsample), n)
        retrieve_size = int(self.mir_cfg.retrieve_size)
        assert subsample >= retrieve_size, \
            f"[MIRER] effective subsample ({subsample}) < retrieve_size ({retrieve_size}); " \
            f"increase buffer or lower retrieve_size"

        local_idxs = np.random.choice(n, size=subsample, replace=False)
        global_idxs = self.er_global_indices[local_idxs]
        samples = [self.er_dataset[int(i)] for i in local_idxs]
        cand_batch = merge_batch_of_dicts(samples)
        cand_batch = move_batch_to_device(cand_batch, self.base_model.device)
        return cand_batch, global_idxs, local_idxs

    def _save_params(self):
        return [p.data.detach().clone() for p in self.base_model.parameters() if p.requires_grad]

    def _virtual_sgd_step(self):
        lr = float(self.mir_cfg.virtual_lr)
        stepped = False
        for p in self.base_model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-lr)
            stepped = True
        assert stepped, "[MIRER] virtual SGD found no parameter gradients from the current batch"

    def _restore_params(self, saved):
        params = [p for p in self.base_model.parameters() if p.requires_grad]
        assert len(params) == len(saved), "[MIRER] param count mismatch on restore"
        for p, s in zip(params, saved):
            p.data.copy_(s)

    def _zero_grads(self):
        self.base_model.zero_grad(set_to_none=True)

    def _maybe_log_selection(self, selected_global_idxs):
        """Accumulate selected global indices.

        Counting is cheap; we record every MIR step by default. If mir_cfg.log_every > 1,
        only every Nth MIR step contributes to the histogram (useful for very long runs).
        """
        self._mir_step += 1
        if self._mir_step % int(self.mir_cfg.log_every) != 0:
            return
        for gidx in selected_global_idxs:
            self._selection_counts[int(gidx)] += 1

    def mir_er_call(self, batch):
        """Virtual MIR retrieval, then mixed current+retrieved loss for the real step."""
        if self.er_dataset is None:
            task_losses = self.base_model(batch)
            return (
                {"raw_loss": task_losses, "er_loss": task_losses * 0.0, "current_task_loss": task_losses},
                {"scalars": {"mir_mean_score": 0.0}, "distrs": None},
            )

        cand_batch, cand_global_idxs, _ = self._sample_candidate_batch()
        retrieve_size = int(self.mir_cfg.retrieve_size)

        # Pre-update candidate losses at current θ, with fixed diffusion noise.
        with torch.no_grad():
            pre_loss_raw, pre_intermediates = self.base_model(
                cand_batch, return_intermediates=True, reduce_loss=False
            )
            pre_loss = _per_example_loss(pre_loss_raw)

        # Virtual grads from the current batch only (discarded after scoring).
        assert self.accelerator is not None, "[MIRER] accelerator must be registered"
        with self.accelerator.no_sync(self.base_model):
            curr_loss = self.base_model(batch)
            self.accelerator.backward(curr_loss)

        saved = self._save_params()
        self._virtual_sgd_step()

        with torch.no_grad():
            post_loss_raw, post_intermediates = self.base_model(
                cand_batch,
                return_intermediates=True,
                parallel_intermediates=pre_intermediates,
                reduce_loss=False,
            )
            assert torch.all(pre_intermediates["step"] == post_intermediates["step"]), \
                "[MIRER] diffusion timesteps diverge between pre/post scoring"
            assert torch.all(pre_intermediates["noise"] == post_intermediates["noise"]), \
                "[MIRER] diffusion noise diverges between pre/post scoring"
            post_loss = _per_example_loss(post_loss_raw)
            scores = post_loss - pre_loss

        self._restore_params(saved)
        self._zero_grads()

        topk = torch.topk(scores, k=retrieve_size, largest=True)
        selected_local = topk.indices.detach().cpu().numpy()
        selected_global = cand_global_idxs[selected_local]
        self._maybe_log_selection(selected_global)

        # Rebuild retrieved batch from the already-materialized candidate batch.
        retrieved = self._index_batch(cand_batch, topk.indices)

        combined_batch = merge_datas(retrieved, batch, device=self.base_model.device)
        loss_vec = self.base_model(combined_batch, reduce_loss=False)
        loss_vec = _per_example_loss(loss_vec)
        er_bs = retrieve_size
        er_loss = loss_vec[:er_bs].mean()
        task_loss = loss_vec[er_bs:].mean()
        raw_loss = self.er_weight * er_loss + task_loss

        return (
            {"raw_loss": raw_loss, "er_loss": er_loss, "current_task_loss": task_loss},
            {
                "scalars": {
                    "mir_mean_score": float(scores[topk.indices].mean().item()),
                    "mir_max_score": float(scores[topk.indices].max().item()),
                },
                "distrs": None,
            },
        )

    def _index_batch(self, batch, indices: torch.Tensor):
        """Gather batch elements along dim 0 by tensor indices (on same device as batch tensors)."""
        if isinstance(batch, torch.Tensor):
            return batch.index_select(0, indices.to(batch.device))
        elif isinstance(batch, (dict, collections.OrderedDict)):
            if isinstance(batch, dict):
                out = dict()
            else:
                out = collections.OrderedDict()
            for k, v in batch.items():
                out[k] = self._index_batch(v, indices)
            return out
        else:
            # metadata / non-tensor leaves: convert via CPU list indexing
            idx_list = indices.detach().cpu().tolist()
            if isinstance(batch, (list, tuple)):
                return type(batch)(batch[i] for i in idx_list)
            raise TypeError(f"[MIRER] _index_batch unsupported type: {type(batch)}")

    def __call__(self, batch):
        return self.mir_er_call(batch)
