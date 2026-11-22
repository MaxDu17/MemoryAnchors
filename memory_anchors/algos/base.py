from memory_anchors.model.common.base_policy import BasePolicy
import torch.nn as nn 
import copy 
# this is an encapsulating class for the learning diffusion policy 
# I'm encapsulating because the general approach is surrounding a policy and it might 
# need multiple policies, etc. Trying to make an encapsulating relationship early 

# move the ema model inside 
class CL_BasePolicy(nn.Module):
    # TODO: FIGURE OUT HOW TO LOAD!!

    def __init__(self, base_learner: BasePolicy):
        super().__init__()
        self.base_model = base_learner
        self.accelerator = None 
    
    # def forward(self, batch):
    #     return self.base_model.forward(batch)
    
    def get_optimizer(self, lr, **kwargs):
        return self.base_model.get_optimizer(lr, **kwargs)
    
    # def get_lr_scheduler(self, **kwargs):
    #     return self.base_model.get_lr_scheduler(**kwargs)
    
    def register_accelerator(self, accelerator):
        self.accelerator = accelerator  # useful for accumulation things 

    def start_task(self, current_dataset, info_cache_dir):
        # info_cache_dir is used for computation done using accelerator 
        pass 

    def end_task(self):
        pass 

    def post_backward(self):
        pass
    
    def log_info(self, output_dir):
        pass 

    def register_training_split_info(self, info):
        return self.base_model.register_training_split_info(info)
    
    def register_training_progress(self, proportion):
        pass 
    
    def set_normalizer(self, normalizer):
        return self.base_model.set_normalizer(normalizer)

    def reset(self):
        return self.base_model.reset()

    @property
    def device(self):
        return self.base_model.device 
    
    def __call__(self, batch):
        return {"raw_loss" : self.base_model(batch)}, {} 

    def state_dict(self, *args, **kwargs):
        # this prevents adding "base_model" to the model name 
        return self.base_model.state_dict() 

    def load_state_dict(self, state_dict, strict=True):
        # I need to manually define this or else we have an issue with the added parameter 
        return self.base_model.load_state_dict(state_dict, strict)
        # self._training_split_info = state_dict.pop('_extra_training_split_info', None)
        # return super().load_state_dict(state_dict, strict)

    def register_accelerator(self, accelerator):
        self.accelerator = accelerator # this is for unwrapping purposes 

    # def copy_model(self): # copies all the weights and stuff 
    #     return copy.deepcopy(self)

    # def compute_obs_representations(self, batch):
    #     return self.base_model.compute_obs_representations(batch) 

    # def compute_intermediate_obs(self, batch):
    #     return self.base_model.compute_intermediate_obs(batch)


    # def train(self):
    #     self.base_model.train() 
    #     # you can add stuff here 
    
    # def eval(self):
    #     self.base_model.eval()
    
    # def parameters(self):
    #     return self.base_model.parameters() 
    #     # consider either exposing it in the workspace or here
        
    # def named_parameters(self):
    #     return self.base_model.named_parameters()

    def modules(self):
        return self.base_model.modules() 