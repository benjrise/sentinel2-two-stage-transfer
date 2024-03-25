import argparse
import os

import yaml

#


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update YAML configuration file with provided arguments"
    )
    parser.add_argument(
        "--output_yaml",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fusion", action="store_true", help="Update fusion model configuration file"
    )
    parser.add_argument(
        "--split",
        type=int,
    )

    parser.add_argument(
        "--repeat_first_layer",
        action="store_true",
    )
    parser.add_argument(
        "--lr_step", type=int, nargs="+", help="Learning rate decay steps"
    )
    parser.add_argument(
        "--rgb_weights",
        type=str,
    )
    parser.add_argument(
        "--s2_weights",
        type=str,
    )

    parser.add_argument(
        "--input_yaml",
        required=True,
        type=str,
        help="Path to the input YAML configuration file",
    )
    parser.add_argument(
        "--linear_probe",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["MM", "RGB", "S2", "S1"],
        help="Modality to train",
    )
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["resnet50", "densenet121", "vit", "dinov2"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["eurosat", "bigearthnet"],
    )

    parser.add_argument(
        "--weights",
        type=str,
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
    )

    parser.add_argument(
        "--freeze_backbones", action="store_true", help="Freeze backbones of the models"
    )
    parser.add_argument("--freeze_to", action="store_true", help="Freeze  to conv5")
    parser.add_argument(
        "--load_models", action="store_true", help="Load models from the provided paths"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument(
        "--blocks", required=False, type=int, nargs="*", help="an integer input"
    )
    parser.add_argument(
        "--down_sample", default="false", type=str.lower, choices=["true", "false"]
    )
    parser.add_argument(
        "--relu", default="false", type=str.lower, choices=["true", "false"]
    )
    parser.add_argument(
        "--bn", default="false", type=str.lower, choices=["true", "false"]
    )
    parser.add_argument(
        "--constrain", default="false", type=str.lower, choices=["true", "false"]
    )
    parser.add_argument(
        "--percentage", type=float, help="Percentage of the dataset to use for training"
    )
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--eval_interval", type=int, help="Evaluation interval")
    parser.add_argument("--eval_period", type=int, help="Evaluation period")
    parser.add_argument(
        "--use_attention", action="store_true", help="Use attention in the fusion model"
    )
    parser.add_argument(
        "--fuse_at_four",
        action="store_true",
        help="Fuse at the fourth resnet block of the fusion model",
    )
    parser.add_argument(
        "--cosine", action="store_true", help="Use cosine learning rate decay"
    )
    parser.add_argument(
        "--undersample", action="store_true", help="Undersample the dataset"
    )

    parser.add_argument("--beta", type=float, help="Beta for undersampling")
    parser.add_argument("--alpha", type=float, help="Alpha for undersampling")
    parser.add_argument(
        "--custom_weights",
        action="store_true",
        help="Use custom weights for the loss function",
    )

    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--max_lr", type=float)
    parser.add_argument("--cut_frac", type=float)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--random_init_first_layer", action="store_true")
    parser.add_argument("--end_to_end", action="store_true")
    return parser.parse_args()


def update_yaml(args, input_yaml):
    with open(input_yaml, "r") as file:
        config = yaml.safe_load(file)

    if args.end_to_end:
        config["FUSION"]["END_TO_END"] = True
    if args.repeat_first_layer:
        config["MODEL"]["RANDOM_INIT_FIRST_LAYER"] = False
    if args.random_init_first_layer:
        config["MODEL"]["RANDOM_INIT_FIRST_LAYER"] = True

    if args.model:
        config["MODEL"]["NAME"] = args.model
    if args.split:
        config["DATASET"]["SPLIT"] = args.split

    if args.architecture:
        config["MODEL"]["NAME"] = args.architecture
    if args.dataset:
        config["DATASET"]["NAME"] = args.dataset
    if args.weights:
        config["MODEL"]["WEIGHTS"] = args.weights

    if args.linear_probe:
        config["TRAIN"]["LINEAR_PROBE"] = True
    # if args.num_classes:
    #     config["MODEL"]["NUM_CLASSES"] = args.num_classes
    if args.rgb_weights is not None:
        config["FUSION"]["RGB_WEIGHTS"] = args.rgb_weights
    if args.s2_weights is not None:
        config["FUSION"]["S2_WEIGHTS"] = args.s2_weights

    if args.modality:
        config["MODEL"]["MODALITY"] = args.modality

    if args.pretrained:
        config["MODEL"]["PRETRAINED"] = True

    if args.lr:
        config["TRAIN"]["LR"] = args.lr

    if args.lr_step:
        config["TRAIN"]["LR_STEP"] = args.lr_step

    if args.epochs:
        config["TRAIN"]["EPOCHS"] = args.epochs

    if args.bs:
        config["TRAIN"]["BS"] = args.bs

    if args.percentage:
        config["DATASET"]["PERCENTAGE"] = args.percentage

    if args.fusion:
        if "FUSION" not in config:
            config["FUSION"] = {}

        config["FUSION"]["ENABLED"] = True
        if args.freeze_backbones:
            config["FUSION"]["FREEZE_BACKBONES"] = True
        else:
            config["FUSION"]["FREEZE_BACKBONES"] = False

        if args.freeze_to:
            config["FUSION"]["FREEZE_TO"] = True

        if args.load_models:
            config["FUSION"]["LOAD_MODELS"] = True
        else:
            config["FUSION"]["LOAD_MODELS"] = False
        if args.modality:
            config["MODEL"]["MODALITY"] = args.modality
        if args.dropout:
            config["FUSION"]["DROPOUT"]["ENABLED"] = True
            config["FUSION"]["DROPOUT"]["RATE"] = args.dropout
        if args.use_attention:
            config["FUSION"]["USE_ATTENTION"] = True
        if args.fuse_at_four:
            config["FUSION"]["FUSE_AT_FOUR"] = True
        else:
            config["FUSION"]["FUSE_AT_FOUR"] = False

    if args.eval_period:
        config["VALIDATION"]["VALIDATION_INTERVAL"] = args.eval_period
    if args.eval_interval:
        config["VALIDATION"]["VALIDATION_INTERVAL"] = args.eval_interval

    return config


def save_yaml(config, config_path):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def main():
    args = parse_args()
    if args.output_yaml:
        output_yaml = args.output_yaml
    else:
        output_yaml = "config_gen.yaml"
    config = update_yaml(args, args.input_yaml)
    folder_path = os.path.dirname(args.input_yaml)
    config_path = os.path.join(folder_path, output_yaml)
    save_yaml(config, config_path)
    print(config_path)


if __name__ == "__main__":
    main()
