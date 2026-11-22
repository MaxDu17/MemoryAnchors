from typing import Dict, List, Optional, Tuple
import torch

from diffusion_policy.policy.base_image_policy import BaseImagePolicy



class BasePolicy(BaseImagePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_split_info = None

    def predict_action_training(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Some policies have different rollout and policy training action prediction methods. For example, prompting methods will predict actions in parallel for every observation in the training batch which consists of a long prompt+rollout, while for standard receeding horizon diffusion policy there is only a single action prediction for each observation."""

        # default behavior is with just observation as input. Past actions are not normally input into model except for prompting based policies
        return self.predict_action(obs_dict)

    def prompt(self, prompt_dict: Dict):
        raise NotImplementedError
    
    def num_available_actions(self) -> Optional[int]:
        """Returns the number of actions that are available to execute. Returns None if no limit is imposed."""
        return None
    
    # reset state for stateful policies
    def reset(self, action_exec_horizon=None):
        """The actual horizon of the action sequence to execute. This should be <= the horizon of the policy."""
        pass

    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError

    def register_training_split_info(self, training_split_info: Dict[Tuple[str, str], bool]) -> None:
        """For prompting models it's important that we know which demonstrations were in the training split because at test time we want to prompt the model with a demonstration that we trained on to help the model stay more in distribution"""
        self._training_split_info = training_split_info
    
    def get_training_split_info(self) -> Dict[Tuple[str, str], bool]:
        return self._training_split_info
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state['_extra_training_split_info'] = self._training_split_info
        return state

    def load_state_dict(self, state_dict, strict=True):
        self._training_split_info = state_dict.pop('_extra_training_split_info', None)

        # CLIP text `position_ids` is a constant `torch.arange(...)` buffer whose
        # persistence flipped across `transformers` versions:
        #   - older: persistent → present in checkpoints / expected by the model
        #   - newer: non-persistent → absent from checkpoints / not expected
        # Align the checkpoint with whatever the live model currently expects.
        # Dropping or re-injecting this buffer is safe and behavior-preserving.
        pos_suffix = 'text_model.embeddings.position_ids'
        live_sd = super().state_dict()
        live_pos_keys = {k for k in live_sd if k.endswith(pos_suffix)}
        for k in [k for k in state_dict if k.endswith(pos_suffix) and k not in live_pos_keys]:
            state_dict.pop(k)
        for k in live_pos_keys:
            if k not in state_dict:
                state_dict[k] = live_sd[k]

        return super().load_state_dict(state_dict, strict)

class BaseContinualLearningPolicy(BasePolicy):
    # this is the continual learning policy class that also allows preparation between tasks 
    
    def start_task(self, task):
        raise NotImplementedError
    
    def end_task(self, task):
        raise NotImplementedError
    

