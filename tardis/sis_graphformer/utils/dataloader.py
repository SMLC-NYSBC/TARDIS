from os import listdir
from os.path import join, splitext
from typing import Optional

import numpy as np
import torch
from tardis.sis_graphformer.utils.augmentation import preprocess_data
from torch.utils.data import Dataset

from tardis.sis_graphformer.utils.voxal import VoxalizeDataSetV2


class GraphDataset(Dataset):
    """
    MODULE TO LOAD 2D/3D COORDINATES AND IMAGE PATCHES

     This module accepts point cloud in shape [X x Y]/[X x Y x Z]
     and output dataset that are expected by graphformer (coord, graph
     and image patches for each coordinate).

    Args:
        coord_dir: source of the 3D .tif images masks.
        coord_format: call for random transformation on img and mask data.
        coord_downsample: Define downsampling method.
        downsample_setting: Define setting for given downsampling method.
        img_dir: source of the 3D .tif file.
        prefix: Prefix name of coordinate file.
        size: numeric value between 0 and 1 for scaling px.
        normalize: type of normalization for img data ["simple", "minmax"]
        memory_save: If True data are loaded with memory save mode on
            (~10x faster computation).
    """

    def __init__(self,
                 coord_dir: str,
                 coord_format="csv",
                 coord_downsample: Optional[str] = None,
                 img_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 size=(12, 12),
                 voxal_size=500,
                 downsampling_if=500,
                 drop_rate=1,
                 downsampling_rate=2.1,
                 normalize="simple",
                 memory_save=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format
        self.coord_downsample = coord_downsample

        # Image setting
        self.img_dir = img_dir
        self.prefix = prefix
        self.size = size
        self.normalize = normalize
        self.memory_save = memory_save

        # Voxal setting
        self.drop_rate = drop_rate
        self.downsampling = downsampling_if
        self.downsampling_rate = downsampling_rate
        self.voxal_size = voxal_size

        self.ids = [file for file in listdir(coord_dir) if file.endswith(f'.{coord_format}')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.coord_format == "csv":
            coord_file = join(self.coord_dir, str(idx))
        elif self.coord_format == "npy":
            coord_file = join(self.coord_dir, str(idx))
        elif self.coord_format == "am":
            coord_file = join(self.coord_dir, str(idx))

        if self.prefix is not None:
            img_idx = idx[:-len(self.prefix)]
        else:
            img_idx = idx

        if self.img_dir is not None:
            img_file = join(self.img_dir, str(img_idx))
        else:
            img_file = None

        coord, img = preprocess_data(coord=coord_file,
                                     image=img_file,
                                     include_label=True,
                                     size=self.size,
                                     normalization=self.normalize,
                                     memory_save=self.memory_save)

        if self.img_dir is None:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=None,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=self.downsampling_rate,
                                   graph=True)
        else:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=img,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=self.downsampling_rate,
                                   graph=True)

        coords_v, imgs_v, graph_v, output_idx = VD.voxalize_dataset(
            out_idx=True)

        return coords_v, imgs_v, graph_v, output_idx


class PredictDataset(Dataset):
    def __init__(self,
                 coord_dir: str,
                 coord_format="csv",
                 coord_downsample: Optional[str] = None,
                 img_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 size=(12, 12),
                 voxal_size=500,
                 downsampling=500000,
                 drop_rate=1,
                 downsampling_rate=2,
                 normalize="simple",
                 memory_save=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format
        self.coord_downsample = coord_downsample

        # Image setting
        self.img_dir = img_dir
        self.prefix = prefix
        self.size = size
        self.normalize = normalize
        self.memory_save = memory_save

        # Voxal setting
        self.drop_rate = drop_rate
        self.downsampling = downsampling
        self.downsampling_rate = downsampling_rate
        self.voxal_size = voxal_size

        self.ids = [splitext(file)[0] for file in listdir(coord_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.coord_format == "csv":
            coord_file = join(self.coord_dir, str(idx) + '.csv')
        elif self.coord_format == "npy":
            coord_file = join(self.coord_dir, str(idx) + '.npy')
        elif self.coord_format == "am":
            coord_file = join(self.coord_dir, str(idx) + '.am')

        if self.prefix is not None:
            img_idx = idx[:-len(self.prefix)]
        else:
            img_idx = idx

        if self.img_dir is not None:
            img_file = join(self.img_dir, str(img_idx) + '.*')
        else:
            img_file = None

        coord, img = preprocess_data(coord=coord_file,
                                     image=img_file,
                                     include_label=True,
                                     size=self.size,
                                     normalization=self.normalize,
                                     memory_save=self.memory_save)

        if self.img_dir is None:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=None,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=self.downsampling_rate,
                                   graph=False)
        else:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=img,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=self.downsampling_rate,
                                   graph=False)

        coords_v, imgs_v, output_idx = VD.voxalize_dataset(out_idx=True)

        return coords_v, imgs_v, output_idx


def filter_collate_fn(batch, dataset):
    """
    MODULE TO REMOVE NONE FROM DATASET AND PICKING NEW SAMPLE

    Args:
        batch: Return batch from DataLoader
        dataset:
    Modified from:
          https://stackoverflow.com/a/57882783
    """
    if batch is not None:
        original_len_batch = len(batch)
        # filter out Nones
        batch = list(filter(lambda x: x is not None, batch))
        filtered_batch_len = len(batch)
        diff = original_len_batch - filtered_batch_len

        # If all are None
        if filtered_batch_len == 0:
            diff = original_len_batch
    else:
        diff = 1
        batch = []

    """
    If Nones detected pick new dataset on their place and check recursively if
    newly picked datasets are not corrupted as well.
    """
    if diff > 0:
        batch.extend([dataset[np.random.randint(0, len(dataset))]]
                     for _ in range(diff))
        return filter_collate_fn(batch, dataset)

    if len(torch.utils.data.dataloader.default_collate(batch)) == 3:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return torch.utils.data.dataloader.default_collate(batch)[0]
