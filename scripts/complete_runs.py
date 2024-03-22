import os
import argparse
import time
import hashlib
import copy
import subprocess

ARCHS = ["resnet50", "densenet121", "vit", "dinov2"]
DATASETS = ["eurosat", "bigearthnet"]
BIGEARTH_WEIGHTS = {
    "resnet50": "runs/bigearthnet/runs_bigearthnet_resnet50/S2_none_resnet50/checkpoints/best_model.pth",
    "densenet121": "runs/bigearthnet/runs_bigearthnet_densenet121/S2_none_densenet121/checkpoints/best_model.pth",
    "vit": "runs/bigearthnet/runs_bigearthnet_vit_learning_rate0.0001/S2_none_vit/checkpoints/best_model.pth",
}
dataset_params = {
    "eurosat": {
        "num_classes": 10,
    },
    "bigearthnet": {
        "num_classes": 19,
    },
}

hyper_map_all = {
    "num_classes": "--num_classes",
    "learning_rate": "--lr",
    "split": "--split",
}
hyper_map_run = {
    "epochs": "--epochs",
    "lr_step": "--lr_step",
    "repeat_first_layer": "--repeat_first_layer",
    "random_init_first_layer": "--random_init_first_layer",
    "linear_probe": "--linear_probe",
    "end_to_end": "--end_to_end",
}


def get_output_base(dataset, architecture, hyperparameters):
    output_dir = os.path.join("runs", f"{dataset}", f"runs_{dataset}_{architecture}")
    if hyperparameters:
        for key, val in hyperparameters.items():
            if key not in hyper_map_all:
                continue
            if isinstance(val, bool):
                output_dir += "_" + key
            else:
                output_dir += "_" + key + str(val)

    return output_dir


def get_output_1stage(output_dir, modality, weights, architecture, hyperparameters):
    output_dir = os.path.join(output_dir, f"{modality}_{weights}_{architecture}")
    if hyperparameters:
        for key, val in hyperparameters.items():
            if key not in hyper_map_run:
                continue
            if isinstance(val, bool):
                output_dir += "_" + key
            elif isinstance(val, list):
                output_dir += "_" + key + str(val)
            else:
                output_dir += "_" + key + str(val)
    return output_dir


def get_output_2stage(output_dir, exp_name, architecture, hyperparameters):
    output_dir = os.path.join(output_dir, f"{exp_name}_{architecture}")
    if hyperparameters:
        for key, val in hyperparameters.items():
            if key not in hyper_map_run:
                continue
            if key == "linear_probe":
                continue
            if isinstance(val, bool):
                output_dir += "_" + key
            elif isinstance(val, list):
                output_dir += "_" + key + str(val)
            else:
                output_dir += "_" + key + str(val)
    return output_dir


def train(
    modality,
    weights,
    dataset,
    architecture,
    output_dir,
    hyperparameters,
    debug=False,
    resume=False,
):
    os.makedirs(output_dir, exist_ok=True)
    base_config = "config_yaml/s2/base_resnet50.yaml"
    combined_hyperparameters = {
        k: v for d in [dataset_params[dataset], hyperparameters] for k, v in d.items()
    }
    config_path = create_config_subprocess(
        base_config,
        modality,
        weights,
        dataset,
        architecture,
        hyperparameters=combined_hyperparameters,
    )
    if debug:
        command = [
            "python",
            "-m",
            "debugpy",
            "--wait-for-client",
            "--listen",
            "0.0.0.0:5678",
            "train.py",
            "--out_dir",
            output_dir,
            "--configs",
            config_path,
        ]
    else:
        command = [
            "python",
            "train.py",
            "--out_dir",
            output_dir,
            "--configs",
            config_path,
        ]
    print("Running command: {}".format(command))

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        exit(1)

    try:
        os.remove(config_path)
    except:
        pass


def create_config_subprocess(
    config_path,
    modality,
    weights,
    dataset,
    architecture,
    hyperparameters,
    rgb_weights=None,
    s2_weights=None,
    linear_probe=False,
    print_config=False,
):
    unique_string = f"{weights}_{rgb_weights}_{s2_weights}_{linear_probe}_{sorted(hyperparameters.items())}"
    unique_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    base_name = f"{modality}_{dataset}_{architecture}"
    output_yaml_filename = f"{base_name}_{unique_id}.yaml"

    create_config_args = [
        "python",
        "tools/create_config.py",
        "--input_yaml",
        config_path,
        "--modality",
        modality.upper(),
        "--dataset",
        dataset,
        "--architecture",
        architecture,
    ]
    if weights != "none":
        create_config_args.extend(["--weights", weights])
    if rgb_weights is not None:
        create_config_args.extend(["--rgb_weights", rgb_weights])
    if s2_weights is not None:
        create_config_args.extend(["--s2_weights", s2_weights])
    if linear_probe:
        create_config_args.append("--linear_probe")

    for key, value in hyperparameters.items():
        arg = hyper_map_all.get(key) or hyper_map_run.get(key) or None
        if arg:
            if isinstance(value, list):
                for val in value:
                    create_config_args.extend([arg, str(val)])
                value = " ".join(map(str, sorted(value)))
            elif isinstance(value, bool):
                if value:
                    create_config_args.append(arg)
                continue
            else:
                value = str(value)
            create_config_args.extend([arg, value])

    create_config_args.extend(["--output_yaml", output_yaml_filename])
    config_out = subprocess.check_output(create_config_args).decode().strip()
    if print_config:
        print("\nGenerated config file: {}".format(config_out))
        with open(config_out, "r") as f:
            print(f.read())
        print("\n" + "=" * 80 + "\n")
    return config_out


def train_two_stage(
    dataset,
    architecture,
    rgb_weights,
    s2_weights,
    output_dir,
    hyperparameters,
    debug=False,
    resume=False,
):
    os.makedirs(output_dir, exist_ok=True)
    base_config = f"config_yaml/fusion/rgbs2/base_{architecture}.yaml"
    hyperparameters = copy.deepcopy(hyperparameters)
    hyperparameters["learning_rate"] = 0.001

    # config_path, modality, weights, dataset, archictecture, hyperparameters={}, print_config=False
    combined_hyperparameters = {
        k: v for d in [dataset_params[dataset], hyperparameters] for k, v in d.items()
    }
    config_path = create_config_subprocess(
        base_config,
        weights="none",
        modality="S2",
        dataset=dataset,
        architecture=architecture,
        rgb_weights=rgb_weights,
        s2_weights=s2_weights,
        linear_probe=hyperparameters.get("linear_probe", False),
        hyperparameters=combined_hyperparameters,
    )
    if debug:
        command = [
            "python",
            "-m",
            "debugpy",
            "--wait-for-client",
            "--listen",
            "0.0.0.0:5678",
            "train.py",
            "--out_dir",
            output_dir,
            "--configs",
            config_path,
        ]
    else:
        command = [
            "python",
            "train.py",
            "--out_dir",
            output_dir,
            "--configs",
            config_path,
        ]
    print("Running command: {}".format(command))

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        exit(1)

    try:
        os.remove(config_path)
    except:
        pass


def check_experiment_done(output_dir):
    return os.path.exists(os.path.join(output_dir, "best_test_metrics.csv"))


def train_model(
    modality, weights, dataset, architecture, hyperparameters, resume=False
):
    output_dir = get_output_base(dataset, architecture, hyperparameters)
    output_dir = get_output_1stage(
        output_dir, modality, weights, architecture, hyperparameters
    )
    if check_experiment_done(output_dir):
        print(
            f"Skipping {modality} model with {weights} on {dataset} using {architecture}, already done."
        )
        return output_dir

    print(
        f"Training {modality} model with {weights} weights on {dataset} using {architecture}"
    )
    if weights == "bigearth":
        weights = BIGEARTH_WEIGHTS[architecture]

    train(
        modality,
        weights,
        dataset,
        architecture,
        output_dir,
        hyperparameters=hyperparameters,
    )
    return output_dir


def run_two_stage_experiments(
    hypers_two_stage, weights_directories, dataset, arch, resume, checkpoint_paths
):
    for exp_name, config in hypers_two_stage.items():
        if dataset not in config["datasets"] or arch not in config["architectures"]:
            print(
                f"Skipping two-stage experiment {exp_name} due to dataset or architecture mismatch."
            )
            continue

        rgb_weights = config["rgb_weights"]
        s2_weights = config["s2_weights"]
        rgb_weights_path = checkpoint_paths.get(rgb_weights, rgb_weights)
        s2_weights_path = checkpoint_paths.get(s2_weights, s2_weights)

        if rgb_weights_path != rgb_weights and not os.path.exists(rgb_weights_path):
            print(
                f"Cannot run two-stage experiment {exp_name} because RGB weights path does not exist."
            )
            continue

        if s2_weights_path != s2_weights and not os.path.exists(s2_weights_path):
            print(
                f"Cannot run two-stage experiment {exp_name} because S2 weights path does not exist."
            )
            continue

        output_dir = get_output_base(dataset, arch, config["hyperparameters"])
        output_dir = get_output_2stage(
            output_dir, exp_name, arch, config["hyperparameters"]
        )

        if check_experiment_done(output_dir):
            print(
                f"Skipping two-stage experiment {exp_name} on {dataset} using {arch}, already done."
            )
            continue

        print(
            f"Running two-stage experiment {exp_name} with RGB weights from {rgb_weights_path} and S2 weights from {s2_weights_path}"
        )
        train_two_stage(
            dataset,
            arch,
            rgb_weights_path,
            s2_weights_path,
            output_dir,
            resume=resume,
            hyperparameters=config["hyperparameters"],
        )


def run_experiments(dataset, arch, hypers_one_stage, hypers_two_stage, resume=False):
    weights_directories = {}
    checkpoint_paths = {}
    for exp_name, config in hypers_one_stage.items():
        if dataset in config["datasets"] and arch in config["architectures"]:
            weights = config["weights"]
            weights_dir = train_model(
                config["modality"],
                weights,
                dataset,
                arch,
                config["hyperparameters"],
                resume,
            )
            weights_directories[(exp_name, dataset, arch)] = weights_dir
            checkpoint_paths[exp_name] = os.path.join(
                weights_dir, "checkpoints", "best_model.pth"
            )
        else:
            print(f"Skipping {exp_name} due to dataset or architecture mismatch.")
            continue

    run_two_stage_experiments(
        hypers_two_stage, weights_directories, dataset, arch, resume, checkpoint_paths
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        help="Dataset to use",
        default="eurosat",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=ARCHS,
        help="Architecture to use",
        default="resnet50",
    )
    parser.add_argument(
        "--lr",
        type=float,
    )
    parser.add_argument("--split", type=int, help="Split to use")
    return parser.parse_args()


def get_hyperparameters(args):
    hypers = {}
    if args.lr:
        hypers["learning_rate"] = args.lr
    if args.split:
        hypers["split"] = args.split
    return hypers


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    arch = args.arch
    hypers = get_hyperparameters(args)
    repeat_hypers = copy.deepcopy(hypers)
    repeat_hypers["repeat_first_layer"] = True

    rgb_randinit_hypers = copy.deepcopy(hypers)
    rgb_randinit_hypers["random_init_first_layer"] = True
    hypers_one_stage = {
        "RGB": {
            "modality": "RGB",
            "weights": "none",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "RGB_PRETRAIN": {
            "modality": "RGB",
            "weights": "imagenet",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "S2": {
            "modality": "S2",
            "weights": "none",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "S2_PRETRAIN_IMAGENET": {
            "modality": "S2",
            "weights": "imagenet",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "S2_PRETRAIN_IMAGENET": {
            "modality": "S2",
            "weights": "imagenet",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "S2_PRETRAIN_IMAGENET_REPEAT": {
            "modality": "S2",
            "weights": "imagenet",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": repeat_hypers,
        },
        "S2_PRETRAIN_BIGEARTH_EUROSAT": {
            "modality": "S2",
            "weights": "bigearth",
            "datasets": ["eurosat"],
            "architectures": ARCHS,
            "hyperparameters": hypers,
        },
        "S2_PRETRAIN_MOCO": {
            "modality": "S2",
            "weights": "moco",
            "datasets": DATASETS,
            "architectures": ["resnet50", "vit"],
            "hyperparameters": hypers,
        },
    }

    linear_hypers = copy.deepcopy(hypers)
    linear_hypers["linear_probe"] = True
    get_hyperparameters(args)
    hypers_two_stage = {
        "RGB_PRETRAIN_S2": {
            "rgb_weights": "RGB_PRETRAIN",
            "s2_weights": "S2",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": linear_hypers,
        },
        "RGB_S2": {
            "rgb_weights": "RGB",
            "s2_weights": "S2",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": linear_hypers,
        },
        "RGB_PRETRAIN_S2_PRETRAIN": {
            "rgb_weights": "RGB_PRETRAIN",
            "s2_weights": "S2_PRETRAIN_IMAGENET",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": linear_hypers,
        },
        "RGB_PRETRAIN_S2_PRETRAIN_REPEAT": {
            "rgb_weights": "RGB_PRETRAIN",
            "s2_weights": "S2_PRETRAIN_IMAGENET_REPEAT",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": linear_hypers,
        },
        "RGB_S2_PRETRAIN_REPEAT": {
            "rgb_weights": "RGB",
            "s2_weights": "S2_PRETRAIN_IMAGENET_REPEAT",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": linear_hypers,
        },
        "SCRATCH_RGB_PRETRAIN_S2": {
            "rgb_weights": "imagenet",
            "s2_weights": "",
            "datasets": DATASETS,
            "architectures": ARCHS,
            "hyperparameters": {
                "linear_probe": False,
                "epochs": 30,
                "lr_step": [27],
                "end_to_end": True,
            },
        },
    }

    run_experiments(dataset, arch, hypers_one_stage, hypers_two_stage, resume=False)
