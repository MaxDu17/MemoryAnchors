from typing import Any
from memory_anchors.model.common.base_policy import BasePolicy
from memory_anchors.algos.base import CL_BasePolicy 

from memory_anchors.common.torch_util import move_batch_to_device
import collections
import torch 

# from memory_anchors.dataset.pushanything_image_dataset import PushAnythingSequenceSampler
# from libero.lifelong.datasets import TruncatedSequenceDataset
from torch.utils.data import ConcatDataset, RandomSampler
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset 
import numpy as np 
from memory_anchors.dataset.truncated_datasets import CustomTruncatedDataset
import json
import os 
import tqdm 
from sklearn.neighbors import NearestNeighbors
import cv2 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import random
import time
import gc

def cycle(dl):
    while True:
        for data in dl:
            yield data


_LIBERO_VIZ_RGB = "agentview_rgb"
_UMI_VIZ_RGB = "camera_right_ultrawide_rgb"


def _viz_rgb_key(obs: dict) -> str:
    """RGB key for Libero vs UMI logging grids (first matching key wins)."""
    if _LIBERO_VIZ_RGB in obs:
        return _LIBERO_VIZ_RGB
    if _UMI_VIZ_RGB in obs:
        return _UMI_VIZ_RGB
    raise KeyError(
        f"AnchorER visualization expects '{_LIBERO_VIZ_RGB}' or '{_UMI_VIZ_RGB}' in obs; "
        f"got keys: {list(obs.keys())}"
    )

# this function recursively merges two batches for ER purposes 
def merge_datas(x, y, device = None):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k], device)
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.FloatTensor) or isinstance(x, torch.cuda.LongTensor):
        return torch.cat([x, y], 0)


def truncate_batch(batch, num_samples):
    """
    Truncate all tensors in the batch along the first dimension to num_samples.
    Args:
        batch: dict or tensor, possibly nested, with leading batch dim.
        num_samples: int, number of samples to keep.
    Returns:
        Truncated batch in same structure.
    """
    if isinstance(batch, torch.FloatTensor) or isinstance(batch, torch.LongTensor) or isinstance(batch, torch.cuda.FloatTensor) or isinstance(batch, torch.cuda.LongTensor):
        return batch[:num_samples]
    elif isinstance(batch, (dict, collections.OrderedDict)):
        if isinstance(batch, dict):
            out = {}
        else:
            out = collections.OrderedDict()
        for k, v in batch.items():
            out[k] = truncate_batch(v, num_samples)
        return out
    else:
        raise Exception("shouldn't get here")
        # for scalars/lists, just return as is
        return batch


def merge_batch_of_dicts(batch_list, device=None):
    '''
    Recursively merges a list of dicts containing torch tensors into one batch dict.
    Each item of batch_list is (possibly nested) dict, with matching structure.
    (Takes inspiration from merge_datas for pairwise merge, but works for N > 2.)

    Args:
        batch_list: list of dict or tensor objects

    Returns:
        merged dict/tensor with tensors concatenated along dim=0
    '''

    # Base case: list of tensors
    first = batch_list[0]
    if isinstance(first, torch.FloatTensor) or isinstance(first, torch.LongTensor) or isinstance(first, torch.cuda.FloatTensor) or isinstance(first, torch.cuda.LongTensor):
        return torch.stack(batch_list, 0)

    # If dict or OrderedDict
    elif isinstance(first, (dict, collections.OrderedDict)):
        if isinstance(first, dict):
            merged = dict()
        else:
            merged = collections.OrderedDict()
        keys = first.keys()
        for k in keys:
            merged[k] = merge_batch_of_dicts([b[k] for b in batch_list], device=device)
        return merged

    else: # for scalars, make them into lists 
        return torch.tensor([b for b in batch_list])

class AnchorER(CL_BasePolicy):
    """Experience replay with a curated buffer (AnchorER and the ER baselines of the paper).

    At the start of every task the replay buffer is re-selected from *all* past-task data
    (``self.full_datasets``) according to ``selection_mode``:

    * ``random_baseline``          RandER: uniform transition-level sample, resampled every task.
    * ``random_baseline_sequential`` FixedER: deterministic prefix of every past task
      (LIBERO's reference ER); see ``selection_cfg.sequential_budget``.
    * ``nearest_neighbor_ood``     Memory Anchors (Sec. 4 / Alg. 1): (1) new data overlapping old
      data in latent space (KMeans on lower-tail kNN distance statistics), (2) the subset with
      high action disagreement (noise-denoise score > mean + std_cutoff*std of held-out old data),
      (3) old data closest to it (median of n_neighbors distances).
        - AnchorER (addition): ``(1 - random_proportion)`` of the buffer are anchors, rest random.
        - Anchor removal: ``invert_selection=true`` excludes the top ``invert_remove_proportion``
          anchors from the past data, then samples the buffer uniformly from the remainder.
    * ``nearest_neighbor``         Ablation without action disagreement (step 2 skipped).

    Buffer size: ``filter_size`` examples (``filter_size_mode=examples``) or a proportion of all
    past data (``filter_size_mode=prop``).
    """

    def __init__(self, base_learner: BasePolicy, filter_size, filter_size_mode, selection_cfg, 
            selection_mode, batch_size, num_workers, er_weight, representation_cfg, 
            preload_dataset = None, add_new_data = True):

        super().__init__(base_learner)
        self.datasets = list() 
        self.full_datasets = list() 

        self.er_weight = er_weight 
        self.selection_mode = selection_mode 
        assert self.selection_mode in (
            "nearest_neighbor_ood", "nearest_neighbor",
            "random_baseline", "random_baseline_sequential",
        ), f"[AnchorER] Invalid selection_mode '{self.selection_mode}'"

        self.last_dataset = None 
        if preload_dataset is not None:
            print("[ER] Loading the pretrained dataset.")
            # self.last_dataset = preload_dataset.load_full_set() # using this to get the validation dataset
            self.last_dataset = preload_dataset.load_pretrain_subset() 
        self.filter_size = filter_size # proportion (prop) or example count (examples)
        self.filter_size_mode = filter_size_mode
        self.selection_cfg = selection_cfg # the settings to the data filter 
        self.representation_cfg = representation_cfg
        assert 0 < self.selection_cfg.invert_remove_proportion < 1, \
            f"[AnchorER] invert_remove_proportion must be in (0,1), got {self.selection_cfg.invert_remove_proportion}"
        assert self.selection_cfg.partition_device in ("cpu", "cuda"), \
            f"[AnchorER] partition_device must be 'cpu' or 'cuda', got '{self.selection_cfg.partition_device}'"
        assert self.selection_cfg.sequential_budget in ("per_task", "total"), \
            f"[AnchorER] sequential_budget must be 'per_task' or 'total', got '{self.selection_cfg.sequential_budget}'"

        self.buffer = None
        self._buffer_loader = None  # owned reference to the DataLoader behind self.buffer
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.task_progress_proportion = 0

        assert add_new_data or preload_dataset is not None,"[ER] you need to use current task data if you're not planning on regularizing to preloaded datatset"
        self.add_new_data = add_new_data 
        if not self.add_new_data:
            print('[ER] This ER instantation keeps only the pretrained set. This is unexpected. Are you sure?')

        assert filter_size_mode in ("prop", "examples"), f"[AnchorER] filter_size_mode must be 'prop' or 'examples', got '{filter_size_mode}'"
        if filter_size_mode == "prop":
            assert 0 < filter_size <= 1, f"[AnchorER] filter_size must be in (0, 1] when filter_size_mode='prop', got {filter_size}"
        else:  # examples
            assert isinstance(filter_size, (int, float)) and filter_size > 0, \
                f"[AnchorER] filter_size must be a positive number when filter_size_mode='examples', got {filter_size}"
            assert int(filter_size) == filter_size, \
                f"[AnchorER] filter_size must be an integer (or whole-number float) when filter_size_mode='examples', got {filter_size}"

    def _get_buffer_size(self, buffer_len: int) -> int:
        """Return the number of examples to keep in the ER buffer."""
        if self.filter_size_mode == "prop":
            size = int(self.filter_size * buffer_len)
            size = max(size, self.batch_size)
        else:  # examples
            size = min(int(self.filter_size), buffer_len)
        return min(size, buffer_len)

    def _sequential_sizes(self):
        """Prefix length kept from each past task for FixedER (``random_baseline_sequential``).

        per_task: every past task keeps its own buffer of ``filter_size`` (LIBERO's reference ER;
                  used for the paper's FixedER-1000 runs, so the total grows with #tasks).
        total:    one buffer of the same size as RandER's, split equally across past tasks.
        In ``prop`` mode the two are (almost) identical.
        """
        lens = [len(d) for d in self.full_datasets]
        if self.selection_cfg.sequential_budget == "per_task":
            return [self._get_buffer_size(n) for n in lens]
        total = self._get_buffer_size(sum(lens))
        base, rem = divmod(total, len(lens))
        sizes = [min(n, base + (1 if i < rem else 0)) for i, n in enumerate(lens)]
        assert sum(sizes) <= total
        return sizes

    def _set_selection_mode(self):
        """Model mode used while computing representations / action disagreement.

        The paper's results were produced with the model left in *train* mode here (so the
        train-time RandomCrop/ColorJitter augmentations are active during selection).
        ``selection_cfg.selection_eval_mode=true`` uses eval mode instead. Returns the previous mode.
        """
        model = self.accelerator.unwrap_model(self.base_model)
        was_training = model.training
        if self.selection_cfg.selection_eval_mode:
            model.eval()
        return was_training

    def _restore_mode(self, was_training):
        self.accelerator.unwrap_model(self.base_model).train(was_training)

    def compute_dataset_representations(self, dataset, normalizer = None):
        # computes the observation representations of the provided dataset 
        model = self.accelerator.unwrap_model(self.base_model)
        representation_list = list()
        loader = DataLoader(
                    dataset,
                    batch_size=8 * self.batch_size, # 8x because we are only using the visual encoder 
                    num_workers=self.num_workers,
                    persistent_workers=False,
                    pin_memory=True,
                    shuffle=False # CRITICAL because we use the indices to make the mask 
                )
        # count = 0 
        repr_keys = self.representation_cfg.keep_modalities 
        with tqdm.tqdm(loader, desc=f"Computing representations",  leave=True) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                with torch.no_grad():
                    select_keys = self.representation_cfg.keep_modalities if self.representation_cfg.use_select_keys else None # filter by representation if needed 
                    representations = model.compute_obs_representations(batch, select_keys = select_keys, return_as_dict = self.representation_cfg.individually_normalized)
                # Store on partition_device so id-partition cdist does not need a later host/device copy of the full matrix.
                store_device = torch.device(self.selection_cfg.partition_device)
                if self.representation_cfg.individually_normalized:
                    representations = {k: v.detach().to(store_device) for k, v in representations.items()}
                else:
                    representations = representations.detach().to(store_device)
                representation_list.append(representations)
                # repr_keys = representations.keys() 
                
                # count += 1 
                # print("DEBUGGING")
                # if count > 20:
                #     break # for debugging only
        # here, we will need to do the reweighting!! 
        norm_dict = None
        if self.representation_cfg.individually_normalized:
            concat_by_modality = {k : torch.concatenate([x[k] for x in representation_list], dim = 0) for k in repr_keys}
            if normalizer is not None:
                assert np.all([k in normalizer for k in repr_keys]), "normalizer doesn't contain the modalities present"
                norm_dict = normalizer 
            else:
                norm_dict = {k : {"mean" : torch.mean(v, dim = 0), "std": torch.std(v, dim = 0)} for k, v in concat_by_modality.items()}        
            normed_by_modality = {k: (v - norm_dict[k]["mean"]) / norm_dict[k]["std"] for k, v in concat_by_modality.items()}

            representation_list = torch.concatenate([normed_by_modality[k] for k in repr_keys], dim = 1)
        else: 
            representation_list = torch.concatenate(representation_list, dim = 0)
        
        torch.cuda.empty_cache()
        return representation_list, norm_dict 
    
    def compute_OOD_baseline(self, valid_dataset):
        # also uses the current policy to compute the normal OOD score 
        model = self.accelerator.unwrap_model(self.base_model)
        baseline_score_list = list() 
        loader = DataLoader(
                    valid_dataset,
                    batch_size=2 * self.batch_size, # 2x because we know the system can handle 2x with batch + ER batch 
                    num_workers=self.num_workers,
                    persistent_workers=False,
                    pin_memory=True,
                    shuffle=True
                )
        with tqdm.tqdm(loader, desc=f"Computing baseline OOD score",  leave=True) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                baseline_score_list.append(
                    model.compute_ood_score(batch, noise_level = self.selection_cfg.noise_level).detach().cpu()
                )
         
        baseline_scores = torch.concatenate(baseline_score_list, dim = 0)
        ood_stat = {"mean" : torch.mean(baseline_scores), "std": torch.std(baseline_scores)} 
        torch.cuda.empty_cache()
        return ood_stat 
    
    
    def partition_new_data(self, last_dataset_reprs, current_dataset_reprs, chunk_size = 8192): 
        '''
        This function takes the new dataset and the old dataset and splits the new dataset into ID and OOD components

        '''
        device = torch.device(self.selection_cfg.partition_device)
        last_dataset_reprs = last_dataset_reprs.detach().to(device)
        current_dataset_reprs = current_dataset_reprs.detach().to(device)

        k = int(0.1 * last_dataset_reprs.shape[0])   # enough to estimate 10% quantile
        assert k > 0, "[AnchorER] partition_new_data requires at least 10 samples in the reference set."

        # For each B, maintain its k smallest distances seen so far
        topk = torch.full((current_dataset_reprs.shape[0], k), float("inf"), device=device)
        print(f"computing the features of B according to A (partition_device={device})")
        for i in tqdm.tqdm(range(0, last_dataset_reprs.shape[0], chunk_size)):
            A_blk = last_dataset_reprs[i : i + chunk_size]                # [chunk, D]
            D_blk = torch.cdist(A_blk, current_dataset_reprs)         # [chunk, NB]

            # merge into running top-k
            merged = torch.cat([topk, D_blk.T], dim=1)   # [NB, k + chunk]
            topk, _ = torch.topk(merged, k, dim=1, largest=False)

        # Now topk[j] contains the k nearest A points to B[j], sorted
        topk = torch.sort(topk, dim=1).values
        labels, in_cluster_idx, cluster_features = self.cluster_and_group_by_distance_features(topk)
        label_idxes = torch.where(torch.tensor(labels == in_cluster_idx))[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return labels, in_cluster_idx, label_idxes 
    
    def visualize_data_partition(self, labels, current_dataset, in_cluster_idx, output_dir, task_name):
        # generate histogram based on these labels of the timestep counts for in distribution vs out of distribution 
        idx_by_labels = {0 : [], 1: []} 
        for i in range(len(current_dataset)):
            current_within_segment_idx, _, _ = current_dataset.sampler.indices[i]
            label = labels[i]
            idx_by_labels[label].append(current_within_segment_idx)

        out_cluster_idx = 0 if in_cluster_idx == 1 else 1 

        # Now plot two histograms, one for each label group
        plt.figure(figsize=(10, 5))
        plt.hist(idx_by_labels[out_cluster_idx], bins=30, alpha=0.7, label='Out of Distribution')
        plt.hist(idx_by_labels[in_cluster_idx], bins=30, alpha=0.7, label='In Distribution')
        plt.xlabel("Step in Trajectory")
        plt.ylabel("Count")
        plt.title("Histogram of Step in Trajectory by cluster label")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task_name}_hist_data_partition.png"))
        plt.close()

        
    def find_OOD_data(self, ood_stat, dataset, task_reprs, id_selection_indices = None):
        '''
        This function takes in a dataset and finds the action-OOD components. 
        Parameters: 
        ood_stat -> the stats we use to threshold the OOD detection 
        dataset -> The target dataset that we want to extract OOD components from
        id_selection_indices (optional) -> The mask to apply on the dataset corresponding to a previous vetting process
            Expect a torch tensor 
        '''
        ood_sample_list = list() # this is a list of images from (s,a) deemed OOD, used for logging 
        ood_step_list = list() # these are the list timesteps of the OOD extraction, used for logging 
        target_representation_list = list() # list of representations from (s,a) deemed OOD, used for the final extraction
        ood_scores_list = list() # list of OOD scores from (s,a) deemed OOD, used for logging 
        ood_indices_list = list() # this is a list of absolute indices that are deemed OOD, which is used for the int diff seleciton process 

        model = self.accelerator.unwrap_model(self.base_model)
        task_reprs = task_reprs.detach().cpu()

        if id_selection_indices is not None:
            if isinstance(id_selection_indices, torch.Tensor):
                id_selection_indices = id_selection_indices.detach().cpu()
            task_dataset = Subset(dataset, id_selection_indices)
        else:
            task_dataset = dataset 

        loader = DataLoader(
                task_dataset,
                batch_size=2 * self.batch_size, # 2x because we know the system can handle 2x with batch + ER batch 
                num_workers=self.num_workers,
                persistent_workers=False,
                pin_memory=True,
                shuffle=False # CRITICAL because we use the indices to make the mask 
            )
        step_count = 0 
        with tqdm.tqdm(loader, desc=f"Selecting OOD examples",  leave=True) as tepoch:

            for batch_idx, batch in enumerate(tepoch):
                scores = model.compute_ood_score(batch, noise_level = self.selection_cfg.noise_level).detach().cpu()
                ood_threshold = ood_stat["mean"] + self.selection_cfg.std_cutoff * ood_stat["std"]
                keep_mask = torch.where(scores > ood_threshold)
                if keep_mask[0].numel() > 0: 
                    ood_scores_list.append(scores[keep_mask])
                    viz_k = _viz_rgb_key(batch["obs"])
                    ood_sample_list.append(batch["obs"][viz_k][keep_mask][:, 0].detach().cpu())
                

                    abs_indices = keep_mask[0] + step_count # makes it absolute
                    abs_indices = abs_indices.detach().cpu().numpy() # past this point we work in numpy space 
                    if id_selection_indices is not None: # if we are using a subset selection, then we're required to convert it back to absolute 
                        abs_indices = id_selection_indices[abs_indices]
                    target_representation_list.append(task_reprs[abs_indices])
                    

                    # within-trajectory step of each OOD sample (column 0), also for a single hit
                    current_within_segment_idx = np.asarray(dataset.sampler.indices)[np.atleast_1d(abs_indices)][:, 0]

                    ood_indices_list.append(abs_indices)

                    ood_step_list.append(current_within_segment_idx) 

                step_count += scores.shape[0] 
        if len(target_representation_list) == 0:
            raise Exception("[AnchorER] Unable to extract any representations from target dataset")

        target_representations = torch.concatenate(target_representation_list, dim = 0)
        ood_scores = torch.concatenate(ood_scores_list, dim = 0).numpy() # numpy because logging 
        ood_samples = torch.concatenate(ood_sample_list, dim = 0).numpy() # numpy because logging 
        ood_step_list = np.concatenate(ood_step_list, axis = 0)
        ood_indices_list = np.concatenate(ood_indices_list, axis = 0).astype(np.int64)

        torch.cuda.empty_cache()
        return target_representations, ood_scores, ood_samples, ood_step_list, ood_indices_list

    
    def cluster_and_group_by_distance_features(self, topk):
        '''
        This function takes in a topk N x k distance matrix where k is the k 10% closest points. It computes
        features from these distances and groups the original N points into two categories. 

        Right now, we rely on the top 1%, top 5%, top 10%, nearest neighbor, spread, and local std as factors in
        the grouping. These are somewhat arbitrary choices about the distributional stats. It's definitely a parameter
        we can tune. 

        Technically a better way of doing this is using a flow model that moves from the latent space back to a continuous space
        but we will use this for now. The heuristic works. 

        Returns: the labels corresponding to the N points, the index of the label that correspond to in-distribution, and the original features

        '''

        k = topk.shape[1]
        q1  = topk[:, int(0.1 * k)]
        q5  = topk[:, int(0.5 * k)]
        q10 = topk[:, -1]
        nn  = topk[:, 0]
        spread = q10 - nn
        local_std = topk.std(dim=1)

        # combininng features and 
        X = torch.stack([nn, q1, q5, q10, spread, local_std], dim=1)
        X = (X - X.mean(0)) / X.std(0)

        labels = KMeans(n_clusters=2, n_init=20).fit_predict(X.cpu().numpy())

        in_cluster_idx = 0 if X[labels==0,0].mean() < X[labels==1,0].mean() else 1 # assigns label identity 

        return labels, in_cluster_idx, X.cpu().numpy()

    def find_nearest_neighbors(self, pretrain_set, target_ood_set, num_nearest_neighbors):
        # returns a list of indices that correspond to the selected ones in the pretrain set 
        assert target_ood_set.shape[0] > 0, "[AnchorER] nearest_neighbor_ood found no OOD target points."
        assert 0 < num_nearest_neighbors <= pretrain_set.shape[0], "[AnchorER] num_nearest_neighbors out of valid range."
        nbrs = NearestNeighbors(n_neighbors=self.selection_cfg.n_neighbors, algorithm='auto')
        print("[AnchorER] Fitting to target representation")
        nbrs.fit(target_ood_set)  # Note: fit on B this time
        print("[AnchorER] Querying existing representation")
        distances, indices_in_target_list = nbrs.kneighbors(pretrain_set)  # Query with A
        # distances is shape (len(A), 1) - min distance from each A point to B
        if self.selection_cfg.consolidation == "median":
            distances = np.median(distances, axis = 1)
        elif self.selection_cfg.consolidation == "mean":
            distances = np.mean(distances, axis = 1)
        else:
            raise Exception("[AnchorER] Invalid consolidation method in find-nearest-neighbors")
        # distances = distances.flatten()
        # Get indices of k points in A with smallest distances to B
        k_nearest_indices = np.argpartition(distances, num_nearest_neighbors)[:num_nearest_neighbors]
        k_nearest_indices = k_nearest_indices[np.argsort(distances[k_nearest_indices])] # sort by distance
        # k_nearest_indices are the indexes in the pretraining set that correspond to the closest in reprsentation space
        # to the OOD samples. So these indices should be used as the mask for learning! 

        indices_in_target_list = indices_in_target_list[k_nearest_indices]
        indices_count = {} # key: anchor point in the new dataset. Value: the number of times it was used 
        for i in range(indices_in_target_list.shape[0]):
            idxes = indices_in_target_list[i]
            for idx in idxes:
                if idx not in indices_count:
                    indices_count[idx] = 0
                indices_count[idx] += 1 

        counterexample_dict = {}
        # this builds up an inverted dictionary of new idx -> old idx close counterexamples 
        assert len(indices_in_target_list) == len(k_nearest_indices) # should always happen 

        # what I want: 
        for i in range(len(indices_in_target_list)):
            old_set_idx = k_nearest_indices[i]
            new_set_idxes = indices_in_target_list[i]
            for new_set_idx in new_set_idxes:
                if new_set_idx not in counterexample_dict:
                    counterexample_dict[new_set_idx] = list()
                counterexample_dict[new_set_idx].append(old_set_idx)
        return k_nearest_indices, indices_count, counterexample_dict

    def log_KNN_results(self, counterexample_dict, indices_count, output_dir, task_name):
        counterexample_counts = [len(v) for k, v in counterexample_dict.items()]
        plt.figure(figsize=(8, 5))
        plt.hist(counterexample_counts, bins=30, alpha=0.7, color='green')
        plt.xlabel("Number of counterexamples per anchor")
        plt.ylabel("Count")
        plt.title("Histogram of Counterexample Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task_name}_hist_counterexample_counts.png"))
        plt.close()

        values = list(indices_count.values())
        # Convert to numpy and sort
        x = np.array(values, dtype=float)
        x = np.sort(x)

        # Cumulative shares
        cum_vals = np.cumsum(x)
        cum_vals = np.insert(cum_vals, 0, 0)          # start at 0
        cum_vals = cum_vals / cum_vals[-1]            # normalize to [0,1]

        # Population shares
        n = len(x)
        pop = np.arange(0, n + 1) / n

        # Plot
        plt.figure(figsize=(6, 6))
        plt.plot(pop, cum_vals, label="Lorenz curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Equality line")
        plt.xlabel("Fraction of items")
        plt.ylabel("Fraction of total value")
        plt.title("Lorenz Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{task_name}_lorentzcurveknn.png"))
        plt.close()

    def log_ood_results(self, scores_list, examples_list, ood_step_list, output_dir, task_name):
        plt.figure(figsize=(8, 5))
        plt.hist(ood_step_list, bins=30, alpha=0.7, color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Count")
        plt.title("Histogram OOD point selection")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task_name}_hist_ood_steps.png"))
        plt.close()

        images_needed = 100
        elements = [] 
        # Get indices of 100 highest OOD scores
        if len(scores_list) >= images_needed:
            topk_indices = np.argpartition(scores_list, -images_needed)[-images_needed:]
            # Sort them in decreasing order
            topk_indices = topk_indices[np.argsort(-scores_list[topk_indices])]
        else:
            topk_indices = np.argsort(-scores_list)
        
        num_imgs = len(topk_indices)
        img_h, img_w = examples_list.shape[2], examples_list.shape[3]

        for i in range(images_needed):
            if i < num_imgs:
                idx = topk_indices[i]
                img = examples_list[idx]
                score = round(scores_list[idx], 3)
                img = np.transpose(img, (1, 2, 0))
                img = np.ascontiguousarray(img) 
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 0, 0)
                thickness = 1
                cv2.putText( img, f"{score:.3f}", (10, 30), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            else:
                # If less than 100 images, fill with black squares
                img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            elements.append(img)
                    
        stacked = np.stack(elements)
        
        grid_rows = []
        for row in range(10):
            row_images = []
            for col in range(10):
                idx = row * 10 + col
                row_images.append(stacked[idx])
            grid_rows.append(np.hstack(row_images))
        
        tiled_image = np.vstack(grid_rows)
        
        # Save using matplotlib
        save_path = os.path.join(output_dir, f"{task_name}_top100_OOD.png")
        plt.imsave(save_path, tiled_image)
        print(f"Saved tiled image to {save_path}")

    def log_selection_results(self, k_nearest_indices, pretrain_dataset, indices, episode_names, output_dir, task_name, task_names=None):
        task_distribution = {} 
        for i in range(k_nearest_indices.shape[0]):
            index = k_nearest_indices[i].item() 
            current_within_segment_idx, episode_idx, task_idx = indices[index]
            episode_name = episode_names[task_idx]
            try:
                name = episode_name.split("-")[0] # this is not generalizable but this is just for visual 
            except:
                name = "" # just for visualization's sake 
            if name not in task_distribution: 
                task_distribution[name] = list()
            task_distribution[name].append(current_within_segment_idx) 

        n_tasks = len(task_distribution)
        n_cols = min(3, n_tasks)
        n_rows = math.ceil(n_tasks / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_tasks == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])

        axes_flat = axes.flatten()

        for idx, (name, idxs) in enumerate(task_distribution.items()):
            ax = axes_flat[idx]
            ax.hist(idxs, bins=20, alpha=0.7)
            ax.set_title(f"Histogram for {name}")
            ax.set_xlabel("Segment Index")
            ax.set_ylabel("Count")

        # Hide any unused axes
        for i in range(idx+1, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        save_hist_path = os.path.join(output_dir, f"{task_name}_selected_data_hist.png")
        plt.savefig(save_hist_path)
        plt.close()
        elements = []
        sel_viz_k = _viz_rgb_key(pretrain_dataset[0]["obs"])
        for i in range(100): 
            if i >= k_nearest_indices.shape[0]:# if we are picking fewer than 100, which can happen 
                elements.append(np.zeros_like(np.transpose(pretrain_dataset[0]["obs"][sel_viz_k][0], (1,2,0)), dtype = np.uint8))
                continue
            index = k_nearest_indices[i].item()
            element = pretrain_dataset[index]["obs"][sel_viz_k][0]  # shape: (3, 224, 224)
            current_within_segment_idx, episode_idx, task_idx = indices[index]
            if task_names is not None:
                prompt_text = str(task_names[task_idx])
            else:
                prompt_text = str(episode_names[task_idx])
            img = element.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = np.ascontiguousarray(img) 
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            thickness = 1
            # Prompt on top; timestep below. Scale font down for long prompts so they fit.
            prompt_font_scale = 0.45 if len(prompt_text) <= 24 else 0.3
            cv2.putText(
                img,
                prompt_text,
                (6, 22),
                font,
                prompt_font_scale,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"t={int(current_within_segment_idx)}",
                (6, 210),
                font,
                0.55,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            elements.append(img)
                    
        # Stack all elements: shape (100, 3, 224, 224)
        stacked = np.stack(elements)
        
        grid_rows = []
        for row in range(10):
            row_images = []
            for col in range(10):
                idx = row * 10 + col
                row_images.append(stacked[idx])
            grid_rows.append(np.hstack(row_images))
        
        tiled_image = np.vstack(grid_rows)
        
        # Save the image
        save_path = os.path.join(output_dir, f"{task_name}_top100_selected.png")
        plt.imsave(save_path, tiled_image)
        print(f"[AnchorER] Saved tiled image to {save_path}")

    def add_random_data(self, last_dataset, pretrain_selected_indices):
        '''
        This function takes the old dataset and the method-selected dataset and adds some more random data 
        '''
        all_indices = set(range(len(last_dataset)))
        selected_set = set(pretrain_selected_indices)
        unselected_indices = list(all_indices - selected_set)
        buffer_size = self._get_buffer_size(len(last_dataset))
        K = int(self.selection_cfg.random_proportion * buffer_size)
        print(f"[AnchorER] Adding {K} random data points!")
        if K > len(unselected_indices):
            raise Exception("[AnchorER] this should not be happening")
            K = len(unselected_indices)
        if K > 0:
            random_extra_indices = np.random.choice(unselected_indices, size=K, replace=False)
            pretrain_selected_indices = np.concatenate([pretrain_selected_indices, random_extra_indices])
        return pretrain_selected_indices 


    def start_task(self, current_dataset, info_cache_dir):
        self.accelerator.print("[AnchorER] Starting Task")
        if self.last_dataset is not None: # this happens at the start if we loaded something, and anytime afterwards if we decide to add data 
            self.full_datasets.append(self.last_dataset)
        
        vis_dir = os.path.join(info_cache_dir, "vis")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        if self.add_new_data: # IN DEVELOPMENT 3/23 
            self.last_dataset = current_dataset # next time, add this buffer to the collection 
        else:
            self.last_dataset = None # if not adding anything new, never add another buffer again 
        
        if len(self.full_datasets) == 0: # logic: when dealing with training from scratch, you provide no last dataset 
            print("[AnchorER] There are no datasets to process. Proceeding with an empty buffer!")
            return 

        # here we recompute the selection (and for fairness, recompute the random assignment baseline too)
        self.accelerator.print(f"[AnchorER] There are {len(self.full_datasets)} datsets to process.")
        self.pretrain_buffer = ConcatDataset(self.full_datasets) # treat all of the last datasets as one 
        
        pretrain_buffer_indices = np.concatenate([dataset.sampler.indices for dataset in self.full_datasets], axis = 0)
        pretrain_buffer_episode_names = np.concatenate([dataset.sampler.replay_buffer.episode_names for dataset in self.full_datasets], axis = 0)
        pretrain_buffer_task_names = np.concatenate(
            [np.asarray(dataset.sampler.replay_buffer.task_names[:]) for dataset in self.full_datasets],
            axis=0,
        ) 
        # Gracefully handle case where 'include_file_filters' does not exist on current_dataset (e.g., umi style)
        if hasattr(current_dataset, "include_file_filters") and current_dataset.include_file_filters is not None:
            current_task_name = "".join(current_dataset.include_file_filters)
        else:
            # fallback: just use a number to describe the task (e.g., use current number of full_datasets + 1)
            current_task_name = f"task{len(self.full_datasets) + 1}"
        nametag = current_task_name #this is how we reference the splits and everything 
        if len(nametag) > 150:
            nametag = nametag[:150] # truncates for sake of file naming convention 
        current_dataset_reprs = None 
        was_training = self._set_selection_mode()
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process and self.selection_mode in (
                "nearest_neighbor_ood", "nearest_neighbor"
            ):
                # last dataset at the start would be the preload dataset 
                # 1) run through last_dataset sequentially to find representations 
                # normalizer will be none if we aren't individually normalizing 
                last_dataset_reprs, normalizer = self.compute_dataset_representations(self.pretrain_buffer)
                current_dataset_reprs, _ = self.compute_dataset_representations(current_dataset, normalizer = normalizer)
                self.accelerator.print(
                    f"\t [AnchorER] Representation tensors on {self.selection_cfg.partition_device} "
                    f"(pretrain={tuple(last_dataset_reprs.shape)}, current={tuple(current_dataset_reprs.shape)})"
                )

                if self.selection_mode == "nearest_neighbor_ood":
                    # 2) Find the new data that is considered ID in state space 
                    id_selection_idxes = None 
                    if self.selection_cfg.id_partition: # enables / disables this step 
                        labels, in_cluster_idx, id_selection_idxes = self.partition_new_data(last_dataset_reprs, current_dataset_reprs)
                        self.visualize_data_partition(labels, current_dataset, in_cluster_idx, info_cache_dir, nametag)
                    
                    # 3) Pick new data from the ID set that is OOD in action space 
                    last_dataset_val = ConcatDataset([dataset.get_validation_dataset() for dataset in self.full_datasets])
                    ood_stats = self.compute_OOD_baseline(last_dataset_val)
                    target_reprs, ood_scores, ood_examples, ood_steps, ood_indices_list = self.find_OOD_data(
                        ood_stats, current_dataset, current_dataset_reprs, id_selection_indices=id_selection_idxes
                    )
                else:
                    # nearest_neighbor baseline: top-K against the whole new dataset (no ID/OOD filtering)
                    self.accelerator.print("\t [AnchorER] nearest_neighbor baseline: KNN against full new-task set")
                    
                    target_reprs = current_dataset_reprs

                # 4) Pick the old data based on similarity to the target representation set
                buffer_sz = self._get_buffer_size(len(self.pretrain_buffer))
                target_reprs_np = target_reprs.detach().cpu().numpy()
                if self.selection_cfg.invert_selection:
                    num_to_exclude = int(self.selection_cfg.invert_remove_proportion * len(self.pretrain_buffer))
                    assert num_to_exclude > 0, "[AnchorER] invert_remove_proportion removes zero points. Increase it."
                    nn_excluded, indices_count, counterexample_dict = self.find_nearest_neighbors(
                        last_dataset_reprs.detach().cpu().numpy(),
                        target_reprs_np,
                        num_to_exclude,
                    )
                    pool = np.setdiff1d(np.arange(len(self.pretrain_buffer)), nn_excluded, assume_unique=False)
                    assert len(pool) >= buffer_sz, "[AnchorER] exclusion leaves fewer samples than requested ER buffer size."
                    last_dataset_selected_indices = np.random.choice(pool, size=buffer_sz, replace=False)
                    self.accelerator.print(
                        f"\t [AnchorER] INVERT_SELECTION: random ER of {buffer_sz} from "
                        f"{len(pool)} indices after excluding {num_to_exclude} nearest neighbors "
                        f"({self.selection_cfg.invert_remove_proportion * 100:.2f}% of pretrain buffer)"
                    )
                else:
                    num_nearest_neighbors = int((1 - self.selection_cfg.random_proportion) * buffer_sz)
                    last_dataset_selected_indices, indices_count, counterexample_dict = self.find_nearest_neighbors(
                        last_dataset_reprs.detach().cpu().numpy(),
                        target_reprs_np,
                        num_nearest_neighbors,
                    )
                    last_dataset_selected_indices = self.add_random_data(self.pretrain_buffer, last_dataset_selected_indices)
                
                self.accelerator.print("\t [AnchorER] Done computing, saving the results for the other processes")
                np.save(os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"), last_dataset_selected_indices)
                np.save(os.path.join(info_cache_dir, f"retrieval_dict_{nametag}.npy"), counterexample_dict)
                if self.selection_mode == "nearest_neighbor_ood":
                    np.save(os.path.join(info_cache_dir, f"ood_indices_{nametag}.npy"), ood_indices_list)
                    self.log_ood_results(ood_scores, ood_examples, ood_steps, vis_dir, nametag)
                
                self.log_selection_results(k_nearest_indices = last_dataset_selected_indices, pretrain_dataset = self.pretrain_buffer, 
                                            indices = pretrain_buffer_indices, episode_names = pretrain_buffer_episode_names,
                                            task_names = pretrain_buffer_task_names,
                                            output_dir = vis_dir, task_name = nametag)
                self.log_KNN_results(counterexample_dict, indices_count, vis_dir, nametag)
                last_dataset_reprs = last_dataset_reprs.detach().cpu() # free the memory 
                current_dataset_reprs = current_dataset_reprs.detach().cpu()
                target_reprs = target_reprs.detach().cpu()
                if self.selection_mode == "nearest_neighbor_ood":
                    del ood_examples
                    del ood_scores
                    del ood_steps
                    del ood_indices_list
                torch.cuda.empty_cache()

            elif self.accelerator.is_main_process and self.selection_mode == "random_baseline":
                # if self.selection_cfg.random_selection_mode == "transition":
                num_indices = self._get_buffer_size(len(self.pretrain_buffer))
                pretrain_selected_indices = np.random.choice(len(self.pretrain_buffer), size=num_indices, replace=False)
                self.accelerator.print("\t [AnchorER: Random Baseline] Done computing, saving the results for the other processes")

                np.save(os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"), pretrain_selected_indices)
            elif self.accelerator.is_main_process and self.selection_mode == "random_baseline_sequential":
                per_dataset_selected_indices = []
                dataset_offset = 0
                for dataset, dataset_num_indices in zip(self.full_datasets, self._sequential_sizes()):
                    per_dataset_selected_indices.append(np.arange(dataset_num_indices) + dataset_offset)
                    dataset_offset += len(dataset)
                pretrain_selected_indices = np.concatenate(per_dataset_selected_indices, axis=0)
                self.accelerator.print("\t [AnchorER: Random Baseline Sequential] Done computing, saving the results for the other processes")

                np.save(os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"), pretrain_selected_indices)
            elif self.accelerator.is_main_process:
                raise Exception("Selection Mode not valid!")
            else:
                print("[AnchorER] Alternate process, waiting for completion")
        
        

            self.accelerator.wait_for_everyone()
            self.accelerator.print("[AnchorER] Main process completed, loading result")
            selected_indices = np.load(os.path.join(info_cache_dir, f"selected_indices_{nametag}.npy"), mmap_mode="r")
            if self.selection_mode in ("nearest_neighbor_ood", "nearest_neighbor"):
                self.retrieval_dict = np.load(os.path.join(info_cache_dir, f"retrieval_dict_{nametag}.npy"), allow_pickle=True).item()
            if self.selection_mode == "nearest_neighbor_ood":
                self.ood_indices = np.load(os.path.join(info_cache_dir, f"ood_indices_{nametag}.npy"), mmap_mode="r")
            # 5) Generate the truncated dataset based on those indices 
            self.accelerator.print("[AnchorER] Extracting most similar data in the existing dataset")
            trunc_set = CustomTruncatedDataset(self.pretrain_buffer, allowed_indices = selected_indices.tolist())
            if self.selection_mode == "random_baseline_sequential":
                expected_trunc_len = sum(self._sequential_sizes())
            else:
                expected_trunc_len = self._get_buffer_size(len(self.pretrain_buffer))
            assert abs(len(trunc_set) - expected_trunc_len) < 2, "the size of the truncated set should be the number of indices specified"
            
            self.accelerator.print(f"[AnchorER] Original data size: {len(self.pretrain_buffer)}. ER Buffer Size: {len(trunc_set)}")
        
        self._restore_mode(was_training)
        if current_dataset_reprs is not None: 
            current_dataset_reprs = current_dataset_reprs.detach().cpu() # free the vram memory 

        # Tear down any prior task's buffer before swapping in the new one, so
        # persistent workers + pin_memory thread from the previous task don't
        # leak (refcount alone is not enough; the iterator has cycles).
        self._teardown_buffer()

        self._buffer_loader = DataLoader(
            trunc_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            shuffle=True,
        )
        self.buffer = cycle(self._buffer_loader)


        self.accelerator.wait_for_everyone()

    def _teardown_buffer(self):
        """
        Shut down the ER buffer's DataLoader and its persistent workers.
        Safe to call when no buffer has been built yet (first task).
        """
        # Close the cycle() generator: raises GeneratorExit at the yield,
        # exits the inner `for data in dl` loop, and drops the local `dl` ref.
        if self.buffer is not None: 
            self.buffer.close()
            self.buffer = None

            # With persistent_workers=True the DataLoader caches its iterator,
            # which owns worker processes and the pin_memory thread. Reference
            # cycles inside the iterator prevent prompt cleanup by refcounting,
            # so we shut workers down explicitly here.
            iterator = self._buffer_loader._iterator
            if iterator is not None:
                iterator._shutdown_workers()
                self._buffer_loader._iterator = None
            self._buffer_loader = None

            gc.collect()

    def end_task(self):
        # because we have set self.last_dataset to current_dataset in the start_task, we have nothing to do here 
        pass 

    
    def log_info(self, output_dir):
        pass 

        # with open(os.path.join(output_dir, "er_splits.json"), "w") as f:
        #     splits = {k : v.tolist() for k, v in self.current_splits.items()}
        #     json.dump(splits, f, indent=2)

    def register_training_progress(self, proportion):
        # this function allows us to ramp the ER contribution based on how far through we are, with each epoch getting a boost 
        self.task_progress_proportion = proportion 

    # compute ER and task separately 
    def simple_er_call(self, batch):
        """
        Compute ER loss and current-task loss in a single model forward by concatenating
        the ER buffer batch (if present) and the current batch along the batch dimension,
        then splitting the per-sample losses.
        """
        if self.buffer is not None:
            buf_data = next(self.buffer)
            buf_data = move_batch_to_device(buf_data, self.base_model.device)
            # ER samples first, then current-task samples
            combined_batch = merge_datas(buf_data, batch, device=self.base_model.device)

            # Get per-sample losses and split into ER vs task components
            loss_vec = self.base_model(combined_batch, reduce_loss=False)

            # Infer ER batch size from obs tensors
            er_bs = next(iter(buf_data["obs"].values())).shape[0]
            er_loss_vec = loss_vec[:er_bs]
            task_loss_vec = loss_vec[er_bs:]

            ER_losses = er_loss_vec.mean()
            task_losses = task_loss_vec.mean()
        else:
            # No ER buffer yet; only task loss
            task_losses = self.base_model(batch)
            ER_losses = task_losses * 0.0
        er_weight = 1
        task_weight = 1 
        if self.selection_cfg.ramp_task_contribution: 
            task_weight = min(1, self.task_progress_proportion) 
 
        raw_losses = er_weight * self.er_weight * ER_losses + task_weight * task_losses # reduces task contribution 

        return {"raw_loss": raw_losses, "er_loss" : ER_losses, "current_task_loss" : task_losses}, {"scalars": {"er_weight": er_weight, "task_weight": task_weight}, "distrs": None} 

    def _compute_orthogonal_component(self, base, target):
        # computes the ortogonal component that goes from base to target, everything normalized
        # Normalize base and target vectors
        base_norm = base / (base.norm(dim=-1, keepdim=True) + 1e-8)
        target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
        proj = (target_norm * base_norm).sum(dim=-1, keepdim=True) * base_norm
        # Orthogonal component
        orthogonal = target_norm - proj
        return orthogonal

    def er_inverted_counterexample_call(self, batch):
        # idea: use the ER buffer and look up the examples in the task dataset 
        pass 
    
    def er_with_counterexample_call(self, batch):
        ER_losses = 0
        counterexample_proportion = 0 
        if self.buffer is not None: 
            buf_data = next(self.buffer)
            buf_data = move_batch_to_device(buf_data, self.base_model.device)

            mask = np.isin(batch["idx"].cpu(), list(self.retrieval_dict.keys()))
            num_hits = mask.sum()
            counterexample_proportion = num_hits / self.batch_size 
            # if we have any counterexamples, extract them from the pretraining buffer and turn them into a batch 
            if np.any(mask):
                x_hit = batch["idx"][mask].cpu()
                # for the ones present in the lookup table, it samples from the selected closest 
                counterexample_sample_idx = np.array([np.random.choice(self.retrieval_dict[v.item()]) for v in x_hit])
     
                counterexample_batch = [self.pretrain_buffer[idx] for idx in counterexample_sample_idx]
                counterexample_batch = merge_batch_of_dicts(counterexample_batch)
                counterexample_batch = move_batch_to_device(counterexample_batch, self.base_model.device)
                
                # if self.selection_cfg.counterexample_cfg.orthogonal_task_shift:
                #     orthogonal_component = self._compute_orthogonal_component(counterexample_batch["obs"]["task_language"], batch["obs"]["task_language"][mask])
                #     orthogonal_component *= self.selection_cfg.counterexample_cfg.orthogonal_travel_scale
                #     task_weight = min(1, self.selection_cfg.counterexample_cfg.ramp_rate * self.task_progress_proportion) 
                #     batch["obs"]["task_language"][mask] += (1 - task_weight) * orthogonal_component 

            # if we have counterexamples, trim the random buffer (shuffled so ok) and combine with counterexamples 
            if num_hits > 0:
                buf_data = truncate_batch(buf_data, self.batch_size - num_hits)
                buf_data = merge_datas(buf_data, counterexample_batch)

            # ER samples (including any counterexamples) first, then current-task samples
            combined_batch = merge_datas(buf_data, batch, device=self.base_model.device)
            loss_vec = self.base_model(combined_batch, reduce_loss=False)

            er_bs = next(iter(buf_data["obs"].values())).shape[0]
            er_loss_vec = loss_vec[:er_bs]
            task_loss_vec = loss_vec[er_bs:]

            ER_losses = er_loss_vec.mean()
            task_losses = task_loss_vec.mean()
        else:
            task_losses = self.base_model(batch)
            ER_losses = task_losses * 0.0
        er_weight = 1
        task_weight = 1 
        if self.selection_cfg.ramp_task_contribution: 
            task_weight = min(1, self.task_progress_proportion / self.selection_cfg.ramp_prop) 
 
        raw_losses = er_weight * self.er_weight * ER_losses + task_weight * task_losses # reduces task contribution 

        return {"raw_loss": raw_losses, "er_loss" : ER_losses, "current_task_loss" : task_losses}, {"scalars": {"er_weight": er_weight, "task_weight": task_weight, "counterexample_prop": counterexample_proportion}, "distrs": None}
    
    def __call__(self, batch): # batch come        # double forward pass 
        if not self.selection_cfg.counterexample:
            return self.simple_er_call(batch)
        else:
            assert self.selection_mode == "nearest_neighbor_ood", "only nearest neighbor ood can support counterexample"
            return self.er_with_counterexample_call(batch)

        
        # ER_losses = 0
        # if self.buffer is not None: 
        #     buf_data = next(self.buffer)
           
        #     buf_data = move_batch_to_device(buf_data, self.base_model.device)
        #     # batch = self.trim_batch(batch, 2) # 2-16 split 
        #     ER_losses = self.base_model(buf_data)
        
        # # noising data 
        # if self.selection_cfg.embedding_noising:
        #     mask = np.isin(batch["idx"].cpu(), list(self.ood_indices))
        #     # used to be 0.02 
        #     # noise = 0.16 * torch.normal(torch.zeros_like(batch["obs"]["task_language"]))
        #     noise = 0.08 * torch.normal(torch.zeros_like(batch["obs"]["task_language"]))
        #     task_weight = min(1, self.task_progress_proportion) 
        #     if self.selection_cfg.noise_ood_only:
        #         batch["obs"]["task_language"][mask] += (1 - task_weight) * noise[mask]
        #     else:
        #         batch["obs"]["task_language"] += (1 - task_weight) * noise 
        #     # noise = 0.02 * np.random.normal(np.zeros_like(name_to_embedding[target_embedding]))


        # task_losses = self.base_model(batch)
        # er_weight = 1
        # task_weight = 1 
        # if self.selection_cfg.ramp_task_contribution: 
        #     # ramp up from 0 -> 1 as self.task_progress_proportion goes from 0->0.5 
        #     # next: stronger ramp-up 
        #     # task_weight = min(1, self.task_progress_proportion * 2) 
        #     task_weight = min(1, self.task_progress_proportion) 
        #     # task_weight = min(1, self.task_progress_proportion * 4) 
 
        # raw_losses = er_weight * self.er_weight * ER_losses + task_weight * task_losses # reduces task contribution 

        # return {"raw_loss": raw_losses, "er_loss" : ER_losses, "current_task_loss" : task_losses}, {"scalars": {"er_weight": er_weight, "task_weight": task_weight}, "distrs": None}