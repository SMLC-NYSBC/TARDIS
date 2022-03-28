import os
from os import listdir
from os.path import join, splitext

import numpy as np
import torch
from spindletorch.utils.augment import preprocess
from tifffile import tifffile
from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    """
        Class module to load image and semantic label masks

    Args:
        img_dir: source of the 3D .tif file
        mask_dir: source of the 3D .tif  images masks
        size: numeric value between 0 and 1 for scaling px
        mask_suffix: numeric value of pixel size
        normalize: type of normalization for img data ["simple", "minmax"]
        transform: call for random transformation on img and mask data

        :author Robert Kiewisz
    """

    def __init__(self,
                 img_dir, mask_dir,
                 size=(64, 512, 512),
                 mask_suffix='_mask',
                 normalize="simple",
                 transform=True,
                 out_channels=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.mask_suffix = mask_suffix
        self.normalize = normalize
        self.transform = transform
        self.out_channels = out_channels

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """
        Get list of all images and masks, load and prepare for packaging
        """
        idx = self.ids[i]
        mask_file = os.path.join(self.mask_dir, str(idx) + '_mask' + '.tif')
        img_file = os.path.join(self.img_dir, str(idx) + '.tif')

        img, mask = tifffile.imread(img_file), tifffile.imread(mask_file)
        img, mask = np.array(img, dtype='uint8'), np.array(mask, dtype='uint8')

        img, mask = preprocess(image=img,
                               mask=mask,
                               size=self.size,
                               normalization=self.normalize,
                               transformation=self.transform,
                               output_dim_mask=self.out_channels)

        img = torch.from_numpy(img).type(torch.float32)
        mask = torch.from_numpy(mask.copy()).type(torch.float32)

        return img, mask


class PredictionDataSet(Dataset):
    def __init__(self,
                 img_dir,
                 out_channels=1):
        self.img_dir = img_dir
        self.out_channels = out_channels

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """
        Get list of all images and masks, load and prepare for packaging
        """
        idx = self.ids[i]
        img_file = join(self.img_dir, str(idx) + '.*')

        img = tifffile.imread(img_file)
        img = np.array(img, dtype='uint8')
        img, _ = preprocess(image=img,
                            mask=img,
                            size=None,
                            normalization="minmax",
                            transformation=False,
                            output_dim_mask=self.out_channels)
        img = torch.from_numpy(img).type(torch.float32)

        return img, idx
