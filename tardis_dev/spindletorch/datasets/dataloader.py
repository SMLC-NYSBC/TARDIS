import os
from os import listdir
from os.path import join, splitext

import numpy as np
import torch
from tardis_dev.spindletorch.datasets.augment import MinMaxNormalize, preprocess
from tardis_dev.utils.load_data import load_image
from torch.utils.data import Dataset


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

    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 size=64,
                 mask_suffix='_mask',
                 transform=True,
                 out_channels=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.out_channels = out_channels
        self.minmax = MinMaxNormalize()

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,
                    i: int) -> torch.Tensor:
        """
        Select and process dataset for CNN.

        Args:
            i (int): Image ID number.

        Returns:
            torch.Tensor: Tensor of processed image and mask.
        """
        # Find next image and corresponding label mask image
        idx = self.ids[i]
        mask_file = os.path.join(self.mask_dir, str(idx) + '_mask' + '.tif')
        img_file = os.path.join(self.img_dir, str(idx) + '.tif')

        # Load image and corresponding label mask
        img, _ = load_image(img_file)
        img = self.minmax(img.astype(np.float32))

        mask, _ = load_image(mask_file)
        assert mask.dtype == np.uint8

        # Process image and mask
        img, mask = preprocess(image=img,
                               mask=mask,
                               size=self.size,
                               transformation=self.transform,
                               output_dim_mask=self.out_channels)

        assert img.dtype == np.float32 and mask.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 1
        return torch.from_numpy(img.copy()).type(torch.float32), \
            torch.from_numpy(mask.copy()).type(torch.float32)


class PredictionDataset(Dataset):
    """
    DATASET BUILDER FOR IMAGES AND SEMANTIC LABEL MASKS FOR PREDICTION

    Module has turn off all transformations.

    Args:
        img_dir (str): Source of the 2D/3D .tif file.
        out_channels (int): Number of output channels.
    """

    def __init__(self,
                 img_dir: str,
                 out_channels=1):
        self.img_dir = img_dir
        self.out_channels = out_channels

        self.minmax = MinMaxNormalize()

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,
                    i: int):
        """
        Select and process dataset for CNN.

        Args:
            i (int): Image ID number.

        Returns:
            torch.Tensor, str: Tensor of processed image and image file name.
        """
        idx = self.ids[i]
        img_file = join(self.img_dir, str(idx) + '.tif')

        # Load image
        img, _ = load_image(img_file)
        img = self.minmax(img.astype(np.float32))

        # Process image and mask
        img = preprocess(image=img,
                         mask=None,
                         size=img.shape,
                         transformation=False,
                         output_dim_mask=self.out_channels)

        return torch.from_numpy(img).type(torch.float32), idx
