from typing import Dict, Tuple
from diffusion_policy.dataset.base_dataset import BaseDataset as DiffusionPolicyBaseDataset

class BaseDataset(DiffusionPolicyBaseDataset):

    # the constructor should have an argument `training_split_info` which is a dictionary mapping (demonstration_name, task_name) to a boolean indicating whether the dataset is in the training split and, if provided, should be used to create the training and validation masks.

    def shuffle_data_ordering(self, seed:int) -> None:
        """For prompt based policies, we need to shuffle the data ordering since we sample large regions of the data and thus the ordering of tasks in the dataset matters. Typically this is called once per epoch when using prompting"""
        raise NotImplementedError

    def requires_epoch_shuffle(self) -> bool:
        """Whether the dataset requires shuffling. For prompt based policies, we need to shuffle the data ordering since we sample large regions of the data and thus the ordering of tasks in the dataset matters."""
        raise NotImplementedError

    def is_multi_task(self) -> bool:
        raise NotImplementedError
    
    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        """Returns a dictionary mapping task names to dataset indices. This is used for multi-task training and evaluation."""
        raise NotImplementedError

    def get_training_split_info(self) -> Dict[Tuple[str, str], bool]:
        """Get the dataset information. Returns a dictionary mapping (demonstration_name, task_name) to a boolean indicating whether the dataset is in the training split."""
        return {}
