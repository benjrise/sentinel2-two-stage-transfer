import logging
import torch
import os

import timm
import torch.nn as nn
import torch.hub as hub
import torchvision.models as models
from torchgeo.models import ResNet50_Weights, ViTSmall16_Weights
from sentinel2.utils.logger import setup_logger


logger = logging.getLogger(__name__)


class SentinelBase(nn.Module):
    def __init__(
        self, nb_class, model_variant, modality, weights, random_init_first_layer=False
    ):
        super().__init__()
        self.nb_class = nb_class
        self.input_channels = 3 if modality.lower() == "rgb" else 10
        self.model_variant = model_variant.lower()
        self.modality = modality
        self.weights = weights
        self.random_init_first_layer = random_init_first_layer
        logger.info(f"Using {model_variant} as backbone")
        logger.info(f"Using {modality} as modality")

        pretrained = False
        load_weights_from_file = False
        if self.weights.lower() in ["imagenet", "moco", "dino"]:
            if self.weights.lower() == "imagenet":
                logger.info("Loading imagenet pretrained weights")
                pretrained = True
            else:
                pretrained_type = "MoCo" if self.weights.lower() == "moco" else "DINO"
                self.input_channels = 13
                logger.info(f"Loading {pretrained_type} pretrained weights")
        elif os.path.isfile(self.weights):
            load_weights_from_file = True
            logger.info(f"Loading weights from file: {self.weights}")
        elif self.weights == "":
            logger.info("Using random weights")
        else:
            raise ValueError(f"Unknown weights: {self.weights}")

        self.initialize_backbone(pretrained)
        if weights.lower() in ["moco", "dino"]:
            assert modality.lower() != "rgb"
            self.handle_ssl_weights(pretrained_type)

        if self.random_init_first_layer and pretrained:
            logger.info("Using randomly initialised first layer.")
            self.init_first_layer()
        elif not self.random_init_first_layer and pretrained:
            logger.info("Using repeated first layer.")

        if load_weights_from_file:
            self.load_weights_except_last_layer(self.weights)

    def initialize_backbone(self, pretrained):
        raise NotImplementedError

    def handle_ssl_weights(self, pretrained_type):
        raise NotImplementedError

    def init_first_layer(self):
        raise NotImplementedError

    def load_weights_except_last_layer(self, weights_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(weights_path)
        exclude_keys = ["fc.weight", "fc.bias"]
        missing_keys = [
            k for k in pretrained_dict if k not in exclude_keys and k not in model_dict
        ]
        if missing_keys:
            raise ValueError(
                f"The following keys from the pretrained model are missing in the current model: {missing_keys}"
            )

        filtered_pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k not in exclude_keys
        }
        model_dict.update(filtered_pretrained_dict)
        self.load_state_dict(model_dict)

        logger.info(
            f"Loaded pretrained weights from {weights_path}, except for the last layer."
        )

    @classmethod
    def from_config(cls, cfg, num_class):
        return cls(
            num_class,
            cfg.MODEL.NAME,
            cfg.MODEL.MODALITY,
            cfg.MODEL.WEIGHTS,
            cfg.MODEL.RANDOM_INIT_FIRST_LAYER,
        )


class ResNetSentinel2(SentinelBase):
    def initialize_backbone(self, pretrained):
        self.backbone = timm.create_model(
            self.model_variant,
            pretrained=pretrained,
            num_classes=0,  # Exclude the classifier
            in_chans=self.input_channels,
        )

        num_features = self.backbone.num_features
        self.fc = nn.Linear(num_features, self.nb_class)

    def handle_ssl_weights(self, pretrained_type):
        assert self.modality.lower() != "rgb"
        weights = getattr(ResNet50_Weights, f"SENTINEL2_ALL_{pretrained_type.upper()}")
        self.backbone.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        channels = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        weights = self.backbone.conv1.weight.data[:, channels, :, :]
        layer = nn.Conv2d(
            self.input_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        layer.weight.data = weights
        self.backbone.conv1 = layer

    def init_first_layer(self):
        logger.info(
            f"Randomly initializing first layer with {self.input_channels} channels and random weights"
        )
        original_first_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            self.input_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False,
        )

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        return x

    def forward_backbone(self, inputs):
        x = self.backbone(inputs)
        return x


class VIT16Sentinel2(SentinelBase):
    @classmethod
    def from_config(cls, cfg, num_class):
        return cls(
            num_class,
            "vit_small_patch16_224",
            cfg.MODEL.MODALITY,
            cfg.MODEL.WEIGHTS,
            cfg.MODEL.RANDOM_INIT_FIRST_LAYER,
        )

    def initialize_backbone(self, pretrained):
        self.backbone = timm.create_model(
            self.model_variant,
            pretrained=pretrained,
            num_classes=0,
            in_chans=self.input_channels,
        )
        num_features = self.backbone.num_features
        self.fc = nn.Linear(num_features, self.nb_class)

    def handle_ssl_weights(self, pretrained_type):
        assert self.modality.lower() != "rgb"
        weights = getattr(
            ViTSmall16_Weights, f"SENTINEL2_ALL_{pretrained_type.upper()}"
        )
        self.input_channels = 13
        self.backbone.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )

        original_patch_embed = self.backbone.patch_embed
        selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        new_conv_weights = original_patch_embed.proj.weight.data[
            :, selected_channels, :, :
        ]
        new_conv_layer = nn.Conv2d(
            in_channels=len(selected_channels),
            out_channels=original_patch_embed.proj.out_channels,
            kernel_size=original_patch_embed.proj.kernel_size,
            stride=original_patch_embed.proj.stride,
            padding=original_patch_embed.proj.padding,
            bias=(original_patch_embed.proj.bias is not None),
        )
        new_conv_layer.weight.data = new_conv_weights
        if original_patch_embed.proj.bias is not None:
            new_conv_layer.bias.data = original_patch_embed.proj.bias.data

        self.backbone.patch_embed.proj = new_conv_layer

    def init_first_layer(self):
        if self.input_channels == 3:
            return

        logger.info(
            f"Randomly initializing patch embedding layer with {self.input_channels} channels"
        )
        original_patch_embed = self.backbone.patch_embed
        new_conv_layer = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=original_patch_embed.proj.out_channels,
            kernel_size=original_patch_embed.proj.kernel_size,
            stride=original_patch_embed.proj.stride,
            padding=original_patch_embed.proj.padding,
            bias=(original_patch_embed.proj.bias is not None),
        )
        self.backbone.patch_embed.proj = new_conv_layer

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        return x


class DenseNetSentinel2(SentinelBase):
    def initialize_backbone(self, pretrained):
        self.backbone = timm.create_model(
            self.model_variant,
            pretrained=pretrained,
            num_classes=0,
            in_chans=self.input_channels,
        )
        num_features = self.backbone.feature_info[-1]["num_chs"]
        self.fc = nn.Linear(num_features, self.nb_class)

    def handle_ssl_weights(self, pretrained_type):
        raise ValueError("DenseNet121 does not support SSL weights")

    def init_first_layer(self):
        if self.input_channels == 3:
            return

        logger.info(
            f"Randomly initializing first convolutional layer with {self.input_channels} channels"
        )
        original_first_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            self.input_channels,
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False,
        )

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.fc(x)
        return x


class DinoV2Sentinel2(nn.Module):
    dinov2_versions = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]

    def __init__(self, nb_class, dino_version="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", dino_version)
        self.n_last_blocks = 4
        self.fc = nn.Linear(384, nb_class)

    def forward(self, x):
        outputs = self.backbone.get_intermediate_layers(
            x, self.n_last_blocks, return_class_token=True
        )
        return outputs

    def forward_backbone(self, x):
        outputs = self.backbone.get_intermediate_layers(
            x, self.n_last_blocks, return_class_token=True
        )
        print(outputs.shape)
        return outputs

    @classmethod
    def from_config(cls, cfg, num_class):
        if cfg.MODEL.MODALITY.lower() != "rgb":
            raise ValueError("DinoV2 only supports RGB modality")

        return cls(num_class)
