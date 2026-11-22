"""Simple wrapper about DiffusionUnetTimmPolicy to make it compatible with memory_anchors BasePolicy."""

import torch
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy as DiffusionUnetTimmPolicyBase
from memory_anchors.model.common.base_policy import BasePolicy


class DiffusionUnetTimmPolicy(DiffusionUnetTimmPolicyBase, BasePolicy):
    def __init__(self, name, **kwargs):
        self.name = name
        # When obs_encoder.pretrained is True, vision params use lr * this scale (default 0.1).
        self.pretrained_vision_lr_scale = float(kwargs.pop("pretrained_vision_lr_scale", 0.1))
        super().__init__(**kwargs)

    def get_optimizer(self, lr, **kwargs):
        obs_encorder_lr = lr
        if self.obs_encoder.pretrained:
            obs_encorder_lr *= self.pretrained_vision_lr_scale
        obs_encorder_params = list()
        for param in self.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        param_groups = [
            {'params': self.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=lr, **kwargs)
        return optimizer
