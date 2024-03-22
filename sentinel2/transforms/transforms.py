import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import logging

logger = logging.getLogger(__name__)

EUROSAT_BAND_STATS = {
    "mean": [
        1353.72692573,
        1117.20229235,
        1041.88472484,
        946.55425487,
        1199.1886645,
        2003.00679978,
        2374.00844879,
        2301.22043931,
        732.18195008,
        12.09952762,
        1820.69637795,
        1118.20272293,
        2599.78293565,
    ],
    "std": [
        245.30185222,
        333.44698356,
        395.22451403,
        594.48167363,
        567.0180941,
        861.02498734,
        1086.97728026,
        1118.32462899,
        403.84709819,
        4.72937175,
        1002.59843175,
        760.62769155,
        1231.68461936,
    ],
}


def get_eurosat_stats(modality):
    if modality.lower() == "rgb":
        # B04, B03, B02 stats
        indexes = [3, 2, 1]
        return (
            torch.tensor(EUROSAT_BAND_STATS["mean"])[indexes],
            torch.tensor(EUROSAT_BAND_STATS["std"])[indexes],
        )
    else:
        # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 stats
        indexes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        return (
            torch.tensor(EUROSAT_BAND_STATS["mean"])[indexes],
            torch.tensor(EUROSAT_BAND_STATS["std"])[indexes],
        )


def get_bigearthnet_stats(modality):
    if modality.lower() == "rgb":
        # B04, B03, B02 stats
        mean = [
            590.23569706,
            614.21682446,
            429.9430203,
        ]
        std = [
            675.88746967,
            582.87945694,
            572.41639287,
        ]
    else:
        mean = [
            429.9430203,
            614.21682446,
            590.23569706,
            950.68368468,
            1792.46290469,
            2075.46795189,
            2218.94553375,
            2266.46036911,
            1594.42694882,
            1009.32729131,
        ]
        std = [
            572.41639287,
            582.87945694,
            675.88746967,
            729.89827633,
            1096.01480586,
            1273.45393088,
            1365.45589904,
            1356.13789355,
            1079.19066363,
            818.86747235,
        ]
    return torch.tensor(mean), torch.tensor(std)


class Normalize(nn.Module):
    def __init__(self, modality, dataset):
        super().__init__()
        if dataset.lower() == "bigearthnet":
            self.mean, self.std = get_bigearthnet_stats(modality)
        elif dataset.lower() == "eurosat":
            self.mean, self.std = get_eurosat_stats(modality)

    def forward(self, x):
        x["image"] = (x["image"] - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        return x


class GetModality(nn.Module):
    def __init__(self, modality, dataset):
        super().__init__()
        self.modality = modality
        if dataset.lower() == "bigearthnet":
            # Note bigearth torchgeo is missing band 10, but we want to exclude anyway
            # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
            self.s2_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
            self.rgb_indexes = [3, 2, 1]
        elif dataset.lower() == "eurosat":
            # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
            self.s2_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
            self.rgb_indexes = [3, 2, 1]
        else:
            raise NotImplementedError

    def forward(self, inputs):
        # B04 = 3, B03 = 2, B02 = 1 (RGB)
        if self.modality.lower() == "rgb":
            inputs["image"] = inputs["image"][self.rgb_indexes, :, :]
        # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        elif self.modality.lower() == "s2":
            inputs["image"] = inputs["image"][self.s2_indexes, :, :]
        return inputs


class Resize(nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, inputs):
        inputs["image"] = F.resize(inputs["image"], self.size, self.interpolation)
        return inputs
