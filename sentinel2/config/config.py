import os

from .yacs import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.MODE = "train"

_C.MODEL = CN()
_C.MODEL.NAME = "ResNet50"
_C.MODEL.LABEL_TYPE = "BigEarthNet-19"
_C.MODEL.MODALITY = "MM"
_C.MODEL.PRETRAINED = False
_C.MODEL.FINE_TUNE = False
_C.MODEL.WEIGHTS = ""
_C.MODEL.RANDOM_INIT_FIRST_LAYER = False
_C.MODEL.DROPOUT_RATE = None

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WEIGHT_DECAY = None
_C.TRAIN.BS = 128
_C.TRAIN.OPTIM = "SGD"
_C.TRAIN.LR = 0.001
_C.TRAIN.CHECKPOINT_INTERVAL = 1
_C.TRAIN.LR_STEP = [27]
_C.TRAIN.LR_DECAY = 0.1
_C.TRAIN.SHUFFLE_BUFFER_SIZE = 13000
_C.TRAIN.LABEL_SMOOTHING = 0.0
_C.TRAIN.BACKWARD_PASSES_PER_STEP = 4
_C.TRAIN.COSINE = False
_C.TRAIN.LINEAR_PROBE = False

_C.VALIDATION = CN()
_C.VALIDATION.BS = 1000
_C.VALIDATION.VALIDATION_INTERVAL = 5
_C.VALIDATION.SAVE_AFTER_VALIDATION = True

_C.DATASET = CN()
_C.DATASET.NAME = "BigEarthNet"
_C.DATASET.EUROSAT_ROOT = "/users/benjrise/sharedscratch2/data"
_C.DATASET.BIGEARTHNET_ROOT = "/users/benjrise/sharedscratch2/data"
_C.DATASET.PERCENTAGE = 1.0
_C.DATASET.SPLIT = 50

_C.TEST = CN()
_C.TEST.EVAL_INTERVAL = 1
_C.TEST.BS = 1000

_C.FUSION = CN()
_C.FUSION.ENABLED = False
_C.FUSION.RGB_MODEL = "ResNet50"
_C.FUSION.S2_MODEL = "ResNet50"
_C.FUSION.MODALITIES = ["RGB", "S2"]
_C.FUSION.FREEZE_TO = False
_C.FUSION.FREEZE_BACKBONES = True
_C.FUSION.LOAD_MODELS = True
_C.FUSION.S2_WEIGHTS = ""
_C.FUSION.RGB_WEIGHTS = ""
_C.FUSION.S1_WEIGHTS_PATH = ""
_C.FUSION.RGB_IMAGENET = False
_C.FUSION.USE_ATTENTION = False
_C.FUSION.ATTENTION_TYPE = "cbam"
_C.FUSION.FUSE_AT_FOUR = False
_C.FUSION.END_TO_END = False
_C.FUSION.DROPOUT = CN()
_C.FUSION.DROPOUT.ENABLED = False
_C.FUSION.DROPOUT.RATE = 0.2


def get_cfg_defaults():
    # Avoid accidently changing config leading to difficult to detect error
    return _C.clone()


def get_cfg(config_yaml):
    cfg = get_cfg_defaults()
    if config_yaml:
        cfg.merge_from_file(config_yaml)
    cfg.freeze()
    return cfg


def save_cfg(cfg, log_dir):
    out_str = cfg.dump()
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        f.write(out_str)


if __name__ == "__main__":
    # Needs to be run as module: python -m config.config
    cfg = get_cfg(None)
    out_str = cfg.dump()
    with open("configs/defaults.yaml", "w") as f:
        f.write(out_str)
