from .models import ResNetSentinel2, DinoV2Sentinel2, DenseNetSentinel2, VIT16Sentinel2
from .fusion_models import FusionModel


def get_model(cfg, nb_class):
    if cfg.FUSION.ENABLED:
        print("Fusion model enabled")
        return FusionModel.from_config(cfg, nb_class)

    if cfg.MODEL.NAME.lower() == "resnet50":
        return ResNetSentinel2.from_config(cfg, nb_class)
    if cfg.MODEL.NAME.lower() == "densenet121":
        return DenseNetSentinel2.from_config(cfg, nb_class)
    if cfg.MODEL.NAME.lower() == "vit":
        return VIT16Sentinel2.from_config(cfg, nb_class)
    if cfg.MODEL.NAME.lower() == "dinov2":
        return DinoV2BigEarthNet.from_config(cfg, nb_class)
    else:
        raise NotImplementedError(f"Model {cfg.MODEL.NAME} not implemented")
