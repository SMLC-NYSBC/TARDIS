#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import os
from os import listdir
from os.path import join, splitext
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tardis_em.cnn.datasets.augmentation import preprocess
from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import load_image
from tardis_em.utils.normalization import MinMaxNormalize


class CNNDataset(Dataset):
    """
    DATASET BUILDER FOR IMAGES AND SEMANTIC LABEL MASKS FOR TRAINING

    Args:
        img_dir (str): Source of the 2D/3D .tif file.
        mask_dir (str): Source of the 2D/3D .tif  images masks.
        size (int): Output patch size for image and mask.
        mask_suffix (str): Suffix name for mask images.
        transform (bool): Call for random transformation on img and mask.
        out_channels (int): Number of output channels.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        size=64,
        mask_suffix="_mask",
        transform=True,
        out_channels=1,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.out_channels = out_channels
        self.minmax = MinMaxNormalize()

        self.ids = [
            splitext(file)[0] for file in listdir(img_dir) if not file.startswith(".")
        ]
        self.format = [
            splitext(file)[1]
            for file in listdir(img_dir)[:5]
            if not file.startswith(".")
        ][1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select and process dataset for CNN.

        Args:
            i (int): Image ID number.

        Returns:
            torch.Tensor, torch.Tensor: Tensor of processed image and mask.
        """
        # Find next image and corresponding label mask image
        idx = self.ids[i]
        mask_file = os.path.join(self.mask_dir, str(idx) + "_mask" + self.format)
        img_file = os.path.join(self.img_dir, str(idx) + self.format)

        # Load image and corresponding label mask
        img, _ = load_image(img_file)
        mask, _ = load_image(mask_file)

        if mask.dtype != np.uint8:
            TardisError(
                "147",
                "tardis_em/cnn/dataset/dataloader.py",
                f"Mask should be of np.uint8 dtype but is {mask.dtype}!",
            )

        # Process image and mask
        img, mask = preprocess(
            image=img,
            mask=mask,
            size=self.size,
            transformation=self.transform,
            output_dim_mask=self.out_channels,
        )

        if img.dtype != np.float32 and mask.dtype != np.uint8:
            TardisError(
                "147",
                "tardis_em/cnn/dataset/dataloader.py",
                f"Mask {mask.dtype} and image  {img.dtype} has wrong dtype!",
            )
        if not img.min() >= -1 and not img.max() <= 1:
            TardisError(
                "147",
                "tardis_em/cnn/dataset/dataloader.py",
                "Image file is not binary!",
            )

        return torch.from_numpy(img.copy()).type(torch.float32), torch.from_numpy(
            mask.copy()
        ).type(torch.float32)


class PredictionDataset(Dataset):
    """
    DATASET BUILDER FOR IMAGES AND SEMANTIC LABEL MASKS FOR PREDICTION

    Module has turn off all transformations.

    Args:
        img_dir (str): Source of the 2D/3D .tif file.
        out_channels (int): Number of output channels.
    """

    def __init__(self, img_dir: str, out_channels=1):
        self.img_dir = img_dir
        self.out_channels = out_channels

        self.ids = [
            splitext(file)[0] for file in listdir(img_dir) if not file.startswith(".")
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        """
        Select and process dataset for CNN.

        Args:
            i (int): Image ID number.

        Returns:
            torch.Tensor, str: Tensor of processed image and image file name.
        """
        idx = self.ids[i]
        img_file = join(self.img_dir, str(idx) + ".tif")

        # Load image
        img, _ = load_image(img_file)
        img = img.astype(np.float32)

        # Process image and mask
        img = preprocess(
            image=img,
            size=img.shape,
            transformation=False,
            output_dim_mask=self.out_channels,
        )

        return torch.from_numpy(img).type(torch.float32), idx
