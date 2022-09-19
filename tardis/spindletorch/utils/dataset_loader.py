import os
from os import listdir
from os.path import join, splitext

import numpy as np
import torch
from tardis.spindletorch.utils.augment import preprocess
from tifffile import tifffile
from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    """
    DATASET BUILDER FOR IMAGES AND SEMANTIC LABEL MASKS FOR TRAINING

    Args:
        img_dir: source of the 2D/3D .tif file
        mask_dir: source of the 2D/3D .tif  images masks
        size: Output patch size for image and mask
        mask_suffix: numeric value of pixel size
        normalize: type of normalization for img data ["simple", "minmax"]
        transform: call for random transformation on img and mask data
        out_channels: Number of output channels
    """

    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 size=64,
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

    def __getitem__(self,
                    i):
        """Find next image and corresponding label mask image"""
        idx = self.ids[i]
        mask_file = os.path.join(self.mask_dir, str(idx) + '_mask' + '.tif')
        img_file = os.path.join(self.img_dir, str(idx) + '.tif')

        img, mask = tifffile.imread(img_file), tifffile.imread(mask_file)
        img, mask = np.array(img, dtype='uint8'), np.array(mask, dtype='uint8')

        """Pre-process image and mask"""
        img, mask = preprocess(image=img,
                               mask=mask,
                               size=self.size,
                               normalization=self.normalize,
                               transformation=self.transform,
                               output_dim_mask=self.out_channels)

        return torch.from_numpy(img).type(torch.float32), \
            torch.from_numpy(mask.copy()).type(torch.float32)


class PredictionDataSet(Dataset):
    """
    DATASET BUILDER FOR IMAGES AND SEMANTIC LABEL MASKS FOR PREDICTION

    Module has turn off all transformations

    Args:
        img_dir: source of the 2D/3D .tif file
        size: Output patch size for image and mask
        out_channels: Number of output channels
    """

    def __init__(self,
                 img_dir: str,
                 size: tuple,
                 out_channels=1):
        self.img_dir = img_dir
        self.size = size
        self.out_channels = out_channels

        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """Find next image and corresponding label mask image"""
        idx = self.ids[i]
        img_file = join(self.img_dir, str(idx) + '.*')

        img = tifffile.imread(img_file)
        img = np.array(img, dtype='uint8')

        """Pre-process image and mask"""
        img, _ = preprocess(image=img,
                            mask=img,
                            size=self.size,
                            normalization=self.normalize,
                            transformation=False,
                            output_dim_mask=self.out_channels)

        return torch.from_numpy(img).type(torch.float32), idx


class GPU_DataLoader(object):
    """
    SIMPLE DATA PROVIDER FOR DATASETS THAT FIT IN GPU MEMORY COMPLETELY

    Every epoch a new shuffle of the data is generated.

    Args:
        x: Image data
        y: Mask data
        batch_size: batch size

    Author:
        Paul Kim
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size=128):
        self._n_examples = len(x)

        self.x = torch.tensor(x).float().cuda()
        self.y = torch.tensor(y).float().cuda()

        self.batch_index = 0
        self.batch_size = batch_size
        self._update_permutation()

    def __iter__(self):
        return self

    def _update_permutation(self):
        """
        Update the list of indices defining the order in which examples are provided.
        Meant to be called once an epoch.
        """
        self.permutation = torch.randperm(self._n_examples).cuda()

    def _get_batch(self):
        batch_indices = self.permutation.narrow(
            dim=0, start=self.batch_index, length=self.batch_size)
        x_batch = torch.index_select(self.x, dim=0, index=batch_indices)
        y_batch = torch.index_select(self.y, dim=0, index=batch_indices)

        return x_batch, y_batch

    def __next__(self):
        if self.batch_index + self.batch_size > self._n_examples:
            self.batch_index = 0
            self._update_permutation()
            raise StopIteration
        else:
            batch = self._get_batch()
            self.batch_index += self.batch_size
            return batch
