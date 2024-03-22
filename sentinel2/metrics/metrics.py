import torch.nn as nn
import torchmetrics
import logging


logger = logging.getLogger(__name__)
CLASS_NAMES = {
    "bigearthnet": [
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    ],
    "eurosat": [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ],
}


class CustomMetrics(nn.Module):
    def __init__(
        self, nb_class, multilabel=True, dataset="bigearthnet", print_args=True
    ):
        super().__init__()
        self.nb_class = nb_class

        assert dataset.lower() in CLASS_NAMES.keys()
        self.class_names = CLASS_NAMES[dataset.lower()]

        micro_kwargs = {
            "num_labels" if multilabel else "num_classes": nb_class,
            "task": "multilabel" if multilabel else "multiclass",
            "average": "micro",
        }
        macro_kwargs = {
            "num_labels" if multilabel else "num_classes": nb_class,
            "task": "multilabel" if multilabel else "multiclass",
            "average": "macro",
        }
        class_kwargs = {
            "num_labels" if multilabel else "num_classes": nb_class,
            "task": "multilabel" if multilabel else "multiclass",
            "average": "none",
        }

        if print_args:
            logger.info(f"Micro kwargs: {micro_kwargs}")
            logger.info(f"Macro kwargs: {macro_kwargs}")

        # Micro metrics
        self.micro_precision = torchmetrics.Precision(**micro_kwargs)
        self.micro_recall = torchmetrics.Recall(**micro_kwargs)
        self.micro_f1 = torchmetrics.F1Score(**micro_kwargs)
        self.micro_accuracy = torchmetrics.Accuracy(**micro_kwargs)

        # Macro metrics
        self.macro_precision = torchmetrics.Precision(**macro_kwargs)
        self.macro_recall = torchmetrics.Recall(**macro_kwargs)
        self.macro_f1 = torchmetrics.F1Score(**macro_kwargs)
        self.macro_accuracy = torchmetrics.Accuracy(**macro_kwargs)

        self.class_precision = torchmetrics.Precision(**class_kwargs)
        self.class_recall = torchmetrics.Recall(**class_kwargs)
        self.class_f1 = torchmetrics.F1Score(**class_kwargs)
        self.class_accuracy = torchmetrics.Accuracy(**class_kwargs)

    def update(self, y_pred, y_true):
        # Update the metrics
        self.micro_precision.update(y_pred, y_true)
        self.micro_recall.update(y_pred, y_true)
        self.micro_f1.update(y_pred, y_true)
        self.micro_accuracy.update(y_pred, y_true)

        self.macro_precision.update(y_pred, y_true)
        self.macro_recall.update(y_pred, y_true)
        self.macro_f1.update(y_pred, y_true)
        self.macro_accuracy.update(y_pred, y_true)

        self.class_precision.update(y_pred, y_true)
        self.class_recall.update(y_pred, y_true)
        self.class_f1.update(y_pred, y_true)
        self.class_accuracy.update(y_pred, y_true)

    def compute(self):
        # Compute and return the metrics
        results = {
            "micro_accuracy": self.micro_accuracy.compute(),
            "micro_recall": self.micro_recall.compute(),
            "micro_precision": self.micro_precision.compute(),
            "micro_f1": self.micro_f1.compute(),
            "macro_accuracy": self.macro_accuracy.compute(),
            "macro_recall": self.macro_recall.compute(),
            "macro_precision": self.macro_precision.compute(),
            "macro_f1": self.macro_f1.compute(),
        }

        class_metrics = {
            "precision": self.class_precision.compute(),
            "recall": self.class_recall.compute(),
            "f1": self.class_f1.compute(),
            "accuracy": self.class_accuracy.compute(),
        }

        # Reset metrics
        self.micro_precision.reset()
        self.micro_recall.reset()
        self.micro_f1.reset()
        self.micro_accuracy.reset()

        self.macro_precision.reset()
        self.macro_recall.reset()
        self.macro_f1.reset()
        self.macro_accuracy.reset()

        # Reset class metrics
        self.class_precision.reset()
        self.class_recall.reset()
        self.class_f1.reset()
        self.class_accuracy.reset()

        # Map class-based metrics to class names
        class_based_results = {}
        for idx, class_name in enumerate(self.class_names):
            class_based_results[class_name] = {
                "precision": class_metrics["precision"][idx].item(),
                "recall": class_metrics["recall"][idx].item(),
                "f1": class_metrics["f1"][idx].item(),
                "accuracy": class_metrics["accuracy"][idx].item(),
            }

        return results, class_based_results
