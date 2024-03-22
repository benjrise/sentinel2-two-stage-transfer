from .datasets import get_bigearthnet_datasets, get_eurosat_datasets


def get_datasets(cfg, transform):
    if cfg.DATASET.NAME.lower() == "bigearthnet":
        train_dataset, val_dataset, test_dataset = get_bigearthnet_datasets(
            transform, cfg
        )
    elif cfg.DATASET.NAME.lower() == "eurosat":
        train_dataset, val_dataset, test_dataset = get_eurosat_datasets(
            transform,
            cfg,
            cfg.DATASET.SPLIT,
        )
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset
