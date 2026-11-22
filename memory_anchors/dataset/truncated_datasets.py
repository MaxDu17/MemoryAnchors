import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset 

# This class hijacks the train mask of a dataset 
# accomplish the limiting of the ER buffer on the 
# trajectory level. This should work on both single and multi-task datasets
# unlike the naive approach, which blindly cuts the number of indices 
class ProportionTruncatedEpisodeDataset(Dataset):
    def __init__(self, sequence_dataset, proportion, use_mask = None, sample_type = "episode"):
        self.sequence_dataset = sequence_dataset

        if use_mask is not None: 
            print("[TruncatedDataset] Using provided mask")
            self.sequence_dataset.set_new_training_mask(use_mask) # you can load from a file 
            new_train_mask = use_mask
            # should index by task name 
        else: 
            if sample_type == "episode": # you just care about sampling based on episode 
                existing_train_mask = self.sequence_dataset.train_mask 
                true_indices = torch.where(torch.as_tensor(existing_train_mask))[0]
                num_true = len(true_indices)
                keep_n = int(round(num_true * proportion))
                if keep_n == 0 and num_true > 0:
                    keep_n = 1  # Ensure at least 1 is kept if there are available trues
                keep_indices = torch.randperm(num_true)[:keep_n]
                selected_true_indices = true_indices[keep_indices]
                # Create new train mask
                new_train_mask = torch.zeros_like(torch.as_tensor(existing_train_mask), dtype=torch.bool)
                new_train_mask[selected_true_indices] = True
                self.sequence_dataset.set_new_training_mask(new_train_mask.numpy())
            elif sample_type == "task": # you care about episode proportion per task 
                existing_train_mask = self.sequence_dataset.train_mask 
                task_names_list = self.sequence_dataset.replay_buffer.task_names[()]
                task_names = np.unique(task_names_list)

                # this does the calculation of how many true we want to keep 
                true_indices = torch.where(torch.as_tensor(existing_train_mask))[0]
                num_true = len(true_indices)
                assert num_true > 0 
                keep_n = int(round(num_true * proportion))
                keep_n_per_task = keep_n // len(task_names)
                if keep_n_per_task == 0:
                    keep_n_per_task = 1  # Ensure at least 1 is kept if there are available trues
                new_train_mask = np.zeros_like(existing_train_mask)

                for task_name in task_names:
                    task_train_indices = np.where(np.logical_and(task_names_list == task_name, existing_train_mask))[0]
                    if len(task_train_indices) > keep_n_per_task:
                        task_train_indices = np.random.choice(task_train_indices, keep_n_per_task, replace = False)
                    new_train_mask[task_train_indices] = True 
                self.sequence_dataset.set_new_training_mask(new_train_mask)
            else:
                raise Exception("[ProportionTruncatedEpisodeDataset] Invalid sample type specified")
            # assert new_train_mask is strictly a subset of existing_train_mask
            assert np.all(existing_train_mask[new_train_mask]) # make sure we're not including anything we shouldn't 
        self.train_mask = new_train_mask 
        # self.sequence_dataset.train_mask = new_train_mask.numpy() 
        # self.sequence_dataset.sampler = PushAnythingSequenceSampler(
        #     mask=new_train_mask,
        #     **self.sequence_dataset.sampler_kwargs
        # )

        # self.buffer_size = keep_n
        # self.buffer_size = buffer_size

    def __len__(self):
        return len(self.sequence_dataset)
        # return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)
        # critical bug if this is a multitask dataset 
        # instead, we actually want to modify the train mask 
        # return self.sequence_dataset.__getitem__(idx)


# this class narrows down the dataset exposure to a set of indices so we can select them deliberately 
# we don't need to modify the train mask because we assume that it's already there 
class CustomTruncatedDataset(Dataset):
    def __init__(self, sequence_dataset, allowed_indices: list):
        self.sequence_dataset = sequence_dataset
        self.allowed_indices = allowed_indices 

    def __len__(self):
        return len(self.allowed_indices)
        # return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(self.allowed_indices[idx])