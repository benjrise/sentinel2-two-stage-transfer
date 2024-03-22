import time
import argparse
import os
import random as rn
import subprocess
import gc
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tabulate import tabulate
from torchgeo.datasets import BigEarthNet
from torchvision import transforms
from tqdm import tqdm


from sentinel2.utils import (
    create_small_table,
    log_every_n,
    log_every_n_seconds,
    log_first_n,
    setup_logger,
    write_performance_metrics,
    write_class_metrics,
)


from sentinel2.config import get_cfg, save_cfg
from sentinel2.metrics import CustomMetrics
from sentinel2.models import get_model
from sentinel2.transforms import Normalize, GetModality, Resize
from sentinel2.dataset import get_datasets

SEED = 42

import torch
import os


def save_checkpoint(epoch, model, optimizer, scheduler, best_fscore, file_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_fscore": best_fscore,
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(file_path, model, optimizer, scheduler):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # start from epoch after checkpoint saved
    return checkpoint["epoch"] + 1, checkpoint["best_fscore"]


def determine_nb_class(cfg):
    dataset_name = cfg.DATASET.NAME.lower()
    if dataset_name == "bigearthnet":
        return 19 if cfg.MODEL.LABEL_TYPE == "BigEarthNet-19" else 43
    elif dataset_name == "eurosat":
        return 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def determine_output_type(cfg):
    return "softmax" if cfg.DATASET.NAME.lower() == "eurosat" else "sigmoid"


def determine_loss_criterion(output_type):
    return (
        torch.nn.BCEWithLogitsLoss()
        if output_type == "sigmoid"
        else nn.CrossEntropyLoss()
    )


def determine_out_func(output_type):
    return torch.nn.Softmax(dim=1) if output_type == "softmax" else torch.sigmoid


def linear_probe_forward(batch, model, device):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model.fc(inputs)
    return outputs, targets


def fine_tune_forward(batch, model, device):
    inputs, targets = batch["image"], batch["label"]
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    return outputs, targets


def extract_features(model, data_loader, device, gather_to_gpu=True):
    model.to(device)
    model.eval()
    features = []
    labels = []

    for batch in tqdm(data_loader, desc="Extracting features"):
        inputs = batch["image"].to(device)
        targets = batch["label"].to(device)
        with torch.no_grad():
            outputs = model.forward_backbone(inputs)
        if gather_to_gpu:
            features.append(outputs)
            labels.append(targets)
        else:
            features.append(outputs.cpu())
            labels.append(targets.float().cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def run_model(cfg, out_dir):
    OUT_DIR = out_dir
    NUM_WORKERS = 4
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    tensorboard_logs = os.path.join(out_dir, "scalars")
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_cfg(cfg, out_dir)
    logging.getLogger("sentinel2logger")

    logger.info("Running model with config:")
    logger.info(str(cfg))
    logger.info(f"Running using random seed: {SEED}")
    logger.info(f"Batch size: {cfg.TRAIN.BS}")
    logger.info(f"Epochs: {cfg.TRAIN.EPOCHS}")
    logger.info(f"Learning rate: {cfg.TRAIN.LR}")
    logger.info("Using dataset: {}".format(cfg.DATASET.NAME))
    logger.info("Using model: {}".format(cfg.MODEL.NAME))

    transformations = [
        GetModality(cfg.MODEL.MODALITY, cfg.DATASET.NAME),
        Normalize(cfg.MODEL.MODALITY, cfg.DATASET.NAME),
    ]

    if "dino" in cfg.MODEL.NAME:
        transformations.append(Resize(112))
    elif "vit" in cfg.MODEL.NAME:
        transformations.append(Resize(224))

    transform = transforms.Compose(transformations)

    train_dataset, val_dataset, test_dataset = get_datasets(cfg, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_class = determine_nb_class(cfg)
    logger.info(f"Using {nb_class} classes")
    output_type = determine_output_type(cfg)
    logger.info(f"Using output type: {output_type}")
    out_func = determine_out_func(output_type)
    logger.info(f"Using output function: {out_func}")
    loss_criterion = determine_loss_criterion(output_type)
    logger.info(f"Using loss criterion: {loss_criterion}")
    custom_metrics = CustomMetrics(
        nb_class=nb_class, multilabel=output_type == "sigmoid", dataset=cfg.DATASET.NAME
    )
    custom_metrics.to(device)

    model = get_model(cfg, nb_class)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_DECAY
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BS, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.VALIDATION.BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.VALIDATION.BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        logger.info("Loading checkpoint...")
        start_epoch, best_fscore = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
        )
        logger.info(f"Starting training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info("No checkpoint found, starting from scratch...")
        best_fscore = -1.0

    writer = SummaryWriter(tensorboard_logs)
    if cfg.TRAIN.LINEAR_PROBE:
        logger.info("Linear probe mode enabled")
        logger.info("Extracting features from train set...")
        train_features, train_labels = extract_features(model, train_loader, device)
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=cfg.TRAIN.BS,
            drop_last=False,
            num_workers=0,
            shuffle=True,
        )
        val_features, val_labels = extract_features(model, val_loader, device)
        logger.info("Extracting features from val set...")
        val_loader = torch.utils.data.DataLoader(
            TensorDataset(val_features, val_labels),
            batch_size=1000,
            drop_last=False,
            num_workers=0,
            shuffle=False,
        )
        logger.info("Extracting features from test set...")
        test_features, test_labels = extract_features(model, test_loader, device)
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=1000,
            drop_last=False,
            num_workers=0,
            shuffle=False,
        )
        model.backbone = None
        torch.cuda.empty_cache()
        gc.collect()

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        model.train()
        for i, (batch) in enumerate(train_loader):
            if cfg.TRAIN.LINEAR_PROBE:
                outputs, targets = linear_probe_forward(batch, model, device)
            else:
                outputs, targets = fine_tune_forward(batch, model, device)

            if output_type == "sigmoid":
                targets = targets.float()

            optimizer.zero_grad()
            loss = loss_criterion(outputs, targets)
            outputs = out_func(outputs)
            custom_metrics.update(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
                logger.info(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss}")

        scheduler.step()
        train_metrics, class_train_metrics = custom_metrics.compute()
        metrics_table = create_small_table(train_metrics)
        logger.info(f"Training metrics:\n{metrics_table}")
        for k, v in train_metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)

        if epoch % cfg.TRAIN.CHECKPOINT_INTERVAL == 0:
            logger.info("Saving checkpoint...")
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                best_fscore,
                os.path.join(checkpoint_dir, "latest_checkpoint.pth"),
            )

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            logger.info("Evaluating on validation set...")
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                    if cfg.TRAIN.LINEAR_PROBE:
                        outputs, targets = linear_probe_forward(batch, model, device)
                    else:
                        outputs, targets = fine_tune_forward(batch, model, device)

                    if output_type == "sigmoid":
                        targets = targets.float()
                    loss = loss_criterion(outputs, targets)
                    outputs = out_func(outputs)
                    custom_metrics.update(outputs, targets)

                val_metrics, class_val_metrics = custom_metrics.compute()
                metrics_table = create_small_table(val_metrics)
                logger.info(f"Validation metrics:\n{metrics_table}")
                if val_metrics["micro_f1"] > best_fscore:
                    best_fscore = val_metrics["micro_f1"]
                    logger.info(f"New best f1 score {best_fscore}.")
                    logger.info("Saving model...")
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_dir, f"best_model.pth"),
                    )
                    write_performance_metrics(
                        out_dir, "best_val_metrics.csv", epoch, val_metrics
                    )

                for k, v in val_metrics.items():
                    writer.add_scalar(f"Val/{k}", v, epoch)

    logger.info("Training complete!")
    logger.info(f"Best validation F1 score: {best_fscore}")
    logger.info("Loading best model...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best_model.pth")))
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logger.info("Evaluating on test set...")
        for batch in tqdm(test_loader):
            if cfg.TRAIN.LINEAR_PROBE:
                outputs, targets = linear_probe_forward(batch, model, device)
            else:
                outputs, targets = fine_tune_forward(batch, model, device)

            if output_type == "sigmoid":
                targets = targets.float()
            loss = loss_criterion(outputs, targets)
            outputs = out_func(outputs)
            custom_metrics.update(outputs, targets)

        total_images = len(test_dataset)
        elapsed_time = time.time() - start_time
        images_per_second = total_images / elapsed_time

        logger.info(f"Processed {total_images} images in {elapsed_time:.2f} seconds.")
        logger.info(f"Images processed per second: {images_per_second:.2f}")

        test_metrics, class_test_metrics = custom_metrics.compute()
        metrics_table = create_small_table(test_metrics)
        logger.info(f"Best test metrics:\n{metrics_table}")
        write_performance_metrics(out_dir, "best_test_metrics.csv", epoch, test_metrics)
        write_class_metrics(
            out_dir, "best_test_class_metrics.csv", epoch, class_test_metrics
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigEarthNet Training")
    parser.add_argument(
        "--configs",
        required=False,
        default="configs/base.json",
        help="JSON config file",
    )
    parser.add_argument(
        "--out_dir", required=True, help="name of dir to save logs and checkpoints to"
    )
    parser_args = parser.parse_args()
    rn.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    logger = setup_logger(
        name="sentinel2logger", abbrev_name="sen2log", output=parser_args.out_dir
    )
    setup_logger(parser_args.out_dir, name="sentinel2", abbrev_name="sentinel2src")
    cfg = get_cfg(parser_args.configs)
    run_model(cfg, parser_args.out_dir)
