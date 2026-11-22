"""
Robomimic-style observation encoder for diffusion policy (ResNet18 + SpatialSoftmax, optional
proprio MLP, language MLP). Compatible with diffusion_policy DiffusionUnetTimmPolicy (output_shape + forward).
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from packaging.version import parse as parse_version

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from memory_anchors.common.augmentation import ImageAugmentation


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("replace_submodules requires pytorch >= 1.9.0")
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features,
        ),
    )
    return root_module


class SpatialSoftmax(nn.Module):
    """Spatial softmax; ported from huihan_libero rgb_modules."""

    def __init__(self, in_c: int, in_h: int, in_w: int, num_kp: Optional[int] = None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
            indexing="ij",
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class DiffusionImageEncoder(nn.Module):
    """ResNet{18,34,50,101} backbone + SpatialSoftmax + linear to embed_size."""

    def __init__(
        self,
        input_shape,
        output_size: int = 64,
        num_kp: int = 32,
        resnet_type: int = 18,
        resnet_pretrained: bool = False,
    ):
        super().__init__()
        C, H, W = input_shape

        resnet_type = int(resnet_type)
        if resnet_type not in (18, 34, 50, 101):
            raise ValueError(f"Unsupported resnet_type={resnet_type}; expected 18/34/50/101")

        def _make_resnet(depth: int, pretrained: bool) -> nn.Module:
            # torchvision>=0.13 uses `weights=` API.
            weights_enum = getattr(torchvision.models, f"ResNet{depth}_Weights")
            if pretrained:
                return getattr(torchvision.models, f"resnet{depth}")(
                    weights=weights_enum.DEFAULT
                )
            return getattr(torchvision.models, f"resnet{depth}")(weights=None)

        resnet = _make_resnet(resnet_type, pretrained=bool(resnet_pretrained))
        layers = list(resnet.children())[:-2]  # keep conv stem + residual stages
        if C != 3:
            layers[0] = nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.backbone(dummy)
            _, out_c, out_h, out_w = out.shape
        self.spatial_softmax = SpatialSoftmax(out_c, out_h, out_w, num_kp=num_kp)
        self.output_dim = num_kp * 2
        self.fc = nn.Linear(self.output_dim, output_size)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.spatial_softmax(h)
        h = self.fc(h)
        return h


# TODO: This doesn't have some of the features we need for SELECTIVE ER.
class MLPEncoder(nn.Module):
    """ReLU MLP for language / task embedding projection."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        assert num_layers >= 1
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers: List[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class RobomimicDiffusionObsEncoder(ModuleAttrMixin):
    """
    Multi-camera ResNet{18,34,50} + SpatialSoftmax over time, optional proprio per timestep, language MLP.
    Layout matches huihan BCDiffusionPolicy.encode_obs.
    """

    def __init__(
        self,
        shape_meta: dict,
        embed_size: int = 64,
        num_kp: int = 32,
        observation_horizon: int = 2,
        resnet_type: int = 18,
        resnet_pretrained: bool = False,
        rgb_keys: Optional[List[str]] = None,
        proprio_keys: Optional[List[str]] = None,
        language_key: str = "task_language",
        language_mlp: Optional[dict] = None,
        train_image_transforms: Optional[ImageAugmentation] = None,
        eval_image_transforms: Optional[ImageAugmentation] = None,
        gn_features_per_group: int = 16,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.embed_size = embed_size
        self.observation_horizon = observation_horizon
        self.language_key = language_key
        self.use_language = self.language_key in shape_meta["obs"]
        # Used by policy optimizer to optionally adjust vision LR.
        self.pretrained = bool(resnet_pretrained)

        obs_meta = shape_meta["obs"]

        if rgb_keys is None:
            rgb_keys = sorted(
                k for k, a in obs_meta.items() if a.get("type", "low_dim") == "rgb"
            )
        self.rgb_keys = rgb_keys

        if proprio_keys is None:
            proprio_keys = []
            for k, a in obs_meta.items():
                if a.get("type", "low_dim") != "low_dim":
                    continue
                if k == language_key:
                    continue
                if a.get("ignore_by_policy", False):
                    continue
                if a.get("is_language", False):
                    continue
                proprio_keys.append(k)
            proprio_keys = sorted(proprio_keys)
        self.proprio_keys = proprio_keys

        self.proprio_dim = 0
        for k in self.proprio_keys:
            sh = tuple(obs_meta[k]["shape"])
            self.proprio_dim += int(math.prod(sh))

        self.image_encoders = nn.ModuleDict()
        for name in self.rgb_keys:
            input_shape = tuple(obs_meta[name]["shape"])
            enc = DiffusionImageEncoder(
                input_shape=input_shape,
                output_size=embed_size,
                num_kp=num_kp,
                resnet_type=resnet_type,
                resnet_pretrained=resnet_pretrained,
            )
            enc = replace_bn_with_gn(enc, features_per_group=gn_features_per_group)
            self.image_encoders[name] = enc

        if self.proprio_dim > 0:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
            )
        else:
            self.proprio_encoder = None

        if self.use_language:
            lm = language_mlp or {}
            lang_in = int(lm.get("input_size", int(math.prod(obs_meta[language_key]["shape"]))))
            lang_hidden = int(lm.get("hidden_size", 128))
            lang_layers = int(lm.get("num_layers", 1))
            self.language_encoder = MLPEncoder(
                input_size=lang_in,
                hidden_size=lang_hidden,
                output_size=embed_size,
                num_layers=lang_layers,
            )
        else:
            self.language_encoder = None

        self.key_train_transform_map = nn.ModuleDict()
        self.key_eval_transform_map = nn.ModuleDict()
        for name in self.rgb_keys:
            self.key_train_transform_map[name] = (
                train_image_transforms.get_transform(name)
                if train_image_transforms is not None
                else nn.Identity()
            )
            self.key_eval_transform_map[name] = (
                eval_image_transforms.get_transform(name)
                if eval_image_transforms is not None
                else nn.Identity()
            )

    def feature_by_key(self, obs_dict: dict, keys: List[str]) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).shape[0]
        T = self.observation_horizon
        requested_keys = set(keys)

        features: List[torch.Tensor] = []
        for img_name in self.rgb_keys:
            if img_name not in requested_keys:
                continue
            img = obs_dict[img_name][:, :T]
            B, t, c, h, w = img.shape
            assert B == batch_size
            assert t == T, f"{img_name} horizon {t} != observation_horizon {T}"
            img_flat = img.reshape(B * self.observation_horizon, c, h, w)
            if self.training:
                img_flat = self.key_train_transform_map[img_name](img_flat)
            else:
                img_flat = self.key_eval_transform_map[img_name](img_flat)
            feat = self.image_encoders[img_name](img_flat)
            feat = feat.reshape(B, self.observation_horizon, -1)
            features.append(feat)

        proprio_keys = [key for key in self.proprio_keys if key in requested_keys]
        if proprio_keys:
            assert self.proprio_encoder is not None
            proprio_list = []
            for key in proprio_keys:
                data = obs_dict[key][:, :T]
                assert data.shape[0] == batch_size
                assert data.shape[1] == T
                proprio_list.append(data)
            proprio = torch.cat(proprio_list, dim=-1)
            proprio_feat = self.proprio_encoder(proprio)
            features.append(proprio_feat)

        assert len(features) > 0, "feature_by_key received no selectable rgb/proprio keys"
        obs_features = torch.cat(features, dim=-1)
        obs_features = obs_features.flatten(start_dim=1)

        if self.language_encoder is not None and self.language_key in requested_keys:
            assert self.language_key in obs_dict, f"missing {self.language_key}"
            lang = obs_dict[self.language_key]
            assert lang.shape[0] == batch_size
            lang = lang.reshape(batch_size, -1)
            lang_feat = self.language_encoder(lang)
            return torch.cat([obs_features, lang_feat], dim=-1)

        return obs_features

    def forward(self, obs_dict: dict) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).shape[0]
        T = self.observation_horizon

        features: List[torch.Tensor] = []
        for img_name in self.rgb_keys:
            img = obs_dict[img_name][:, :T]
            B, t, c, h, w = img.shape
            assert t == T, f"{img_name} horizon {t} != observation_horizon {T}"
            img_flat = img.reshape(B * self.observation_horizon, c, h, w)
            if self.training:
                img_flat = self.key_train_transform_map[img_name](img_flat)
            else:
                img_flat = self.key_eval_transform_map[img_name](img_flat)
            feat = self.image_encoders[img_name](img_flat)
            feat = feat.reshape(B, self.observation_horizon, -1)
            features.append(feat)

        if self.proprio_encoder is not None:
            proprio_list = []
            for key in self.proprio_keys:
                proprio_list.append(obs_dict[key][:, :T])
            proprio = torch.cat(proprio_list, dim=-1)
            proprio_feat = self.proprio_encoder(proprio)
            features.append(proprio_feat)

        obs_features = torch.cat(features, dim=-1)
        obs_features = obs_features.flatten(start_dim=1)

        if self.language_encoder is not None:
            assert self.language_key in obs_dict, f"missing {self.language_key}"
            lang = obs_dict[self.language_key]
            lang = lang.reshape(batch_size, -1)
            lang_feat = self.language_encoder(lang)
            return torch.cat([obs_features, lang_feat], dim=-1)

        return obs_features

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            th = int(attr.get("horizon", 1))
            if key == self.language_key:
                tdim = th
            elif attr.get("type", "low_dim") == "rgb" or key in self.proprio_keys:
                tdim = self.observation_horizon
            else:
                tdim = min(th, self.observation_horizon)
            example_obs_dict[key] = torch.zeros(
                (1, tdim) + shape,
                dtype=self.dtype,
                device=self.device,
            )
        out = self.forward(example_obs_dict)
        return out.shape
