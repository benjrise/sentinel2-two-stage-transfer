import torch.nn as nn
import torch
import torchvision.transforms as T
from .models import ResNetSentinel2, DenseNetSentinel2, VIT16Sentinel2, DinoV2Sentinel2

import logging

logger = logging.getLogger(__name__)


def get_model(model, modality, nb_class, weights):
    if model.lower().startswith("resnet"):
        return ResNetSentinel2(
            nb_class, model_variant=model, modality=modality, weights=weights
        )
    elif model.lower().startswith("densenet"):
        return DenseNetSentinel2(nb_class, model, modality, weights)
    elif model.lower().startswith("vit"):
        return VIT16Sentinel2(nb_class, "vit_small_patch16_224", modality, weights)
    elif model.startswith("dino"):
        assert modality == "rgb"
        return DinoV2Sentinel2(nb_class, model)
        raise NotImplementedError


class FusionModel(nn.Module):
    def __init__(
        self,
        nb_class,
        s2_model,
        rgb_model,
        s2_weights,
        rgb_weight,
        dinov2_strategy=False,
        end_to_end=False,
    ):
        super().__init__()
        logger.info("========= Loading s2 =========")
        self.s2_model = get_model(s2_model, "s2", nb_class, s2_weights)
        logger.info("========= Loading rgb =========")
        if "dino" in rgb_model:
            self.preprocess = T.Resize((112, 112))
        else:
            self.preprocess = nn.Identity()
        self.rgb_model = get_model(rgb_model, "rgb", nb_class, rgb_weight)
        in_features = self.s2_model.fc.in_features + self.rgb_model.fc.in_features
        for model in [self.s2_model, self.rgb_model]:
            model.fc = nn.Identity()
        self.fc = nn.Linear(in_features, nb_class)
        self.end_to_end = end_to_end

    @property
    def backbone(self):
        return (self.s2_model.backbone, self.rgb_model.backbone)

    @backbone.setter
    def backbone(self, value):
        if value is None:
            del self.s2_model.backbone
            del self.rgb_model.backbone

    @classmethod
    def from_config(cls, cfg, nb_class):
        # fmt: off
        s2_model         = cfg.FUSION.S2_MODEL
        rgb_model        = cfg.FUSION.RGB_MODEL
        s2_weights_path  = cfg.FUSION.S2_WEIGHTS
        rgb_weights_path = cfg.FUSION.RGB_WEIGHTS
        end_to_end       = cfg.FUSION.END_TO_END
        # fmt: on
        return cls(
            nb_class, s2_model, rgb_model, s2_weights_path, rgb_weights_path, end_to_end
        )

    def forward(
        self,
        inputs,
    ):
        if not self.end_to_end:
            self.rgb_model.eval()
            self.s2_model.eval()

        s2_inputs = inputs
        rgb_inputs = inputs[:, [2, 1, 0], ...]
        # For dinov2 we need a different sized input (multiple of 14)
        # For other models this is the identity
        rgb_inputs = self.preprocess(rgb_inputs)
        x1 = self.rgb_model(rgb_inputs)
        x2 = self.s2_model(s2_inputs)
        # This is speial case for dinov2
        if isinstance(x1, tuple):
            return (x1, x2)
        outputs = [x1, x2]
        outputs = torch.cat(outputs, dim=1)
        return self.fc(outputs)

    def forward_backbone(
        self,
        inputs,
    ):
        self.rgb_model.eval()
        self.s2_model.eval()
        s2_inputs = inputs
        rgb_inputs = inputs[:, [2, 1, 0], ...]
        rgb_inputs = self.preprocess(rgb_inputs)
        x1 = self.rgb_model(rgb_inputs)
        x2 = self.s2_model(s2_inputs)
        # This is speial case for dinov2
        if isinstance(x1, tuple):
            return (x1, x2)
        outputs = [x1, x2]
        outputs = torch.cat(outputs, dim=1)
        return outputs
