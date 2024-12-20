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
    Handles dataset creation and processing for Convolutional Neural Network (CNN) training and inference.

    This class manages the loading, processing, and formatting of image and mask data
    needed for training CNN models. It includes normalization, size adjustments,
    and optional transformations to prepare data for further model usage.
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
        """
        Represents a dataset class designed to accommodate image and mask directories for image
        processing tasks. This class initializes various properties including image and
        mask directories, normalization strategies, and options for transformation. It manages
        the internal identifiers for files in the directories and detects the file format
        based on sample files from the image directory.

        :param img_dir: Path to the directory containing the input images.
        :param mask_dir: Path to the directory containing the masks for the input images.
        :param size: Desired size of the images and masks after resizing. Defaults to 64.
        :param mask_suffix: Suffix added to the filenames of masks to associate them
            with corresponding input images. Defaults to "_mask".
        :param transform: Boolean value indicating whether to apply transformations
            to the images and masks. Defaults to True.
        :param out_channels: Number of output channels for processed data.
            Typically used in segmentation tasks. Defaults to 1.
        """
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
        Retrieves the image and corresponding label mask located at the specified index.
        The function performs the following steps:
        1. Identifies image and mask filenames using the given index.
        2. Loads the image and its associated mask.
        3. Validates their data types and ensures a proper binary range for the image.
        4. Preprocesses the image and mask into the desired format and size.
        5. Returns both as PyTorch tensors.

        :param i: Index of the desired image-mask pair within the dataset.
        :type i: int

        :return: Tuple containing a preprocessed image tensor and the corresponding mask tensor.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
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
    Manages loading, processing, and serving of datasets for image prediction tasks.

    This class provides methods to load images from a specified directory, preprocess
    and format them into tensors compatible with convolutional neural networks. It also
    facilitates retrieval by a specific index. Used for predictive model input preparation.
    """

    def __init__(self, img_dir: str, out_channels=1):
        """
        This class initializes with the specified image directory path and output channel
        configuration. It lists all the files in the directory, excluding hidden files,
        and extracts their base names (without extension).

        :param img_dir: Path to the directory containing images. Represents the input
            directory from which image files' names will be retrieved and stored.
        :param out_channels: An integer representing the number of output channels to
            be used or processed. Default value is set to 1.
        """
        self.img_dir = img_dir
        self.out_channels = out_channels

        self.ids = [
            splitext(file)[0] for file in listdir(img_dir) if not file.startswith(".")
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        """
        Access an item from the dataset by its index. This method retrieves an image
        given its index from the dataset, processes it, and returns the preprocessed
        image along with the corresponding index.

        :param i: An integer index of the image to retrieve from the dataset.
        :type i: int

        :return: A tuple containing the preprocessed image as a tensor and the
                 corresponding index of the image in the dataset.
        :rtype: Tuple[torch.Tensor, int]
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
