from os import listdir
from os.path import join, splitext
from typing import Optional

import numpy as np
import torch
from tardis.dist_pytorch.utils.augmentation import preprocess_data
from tardis.dist_pytorch.utils.voxal import VoxalizeDataSetV2
from tardis.utils.utils import pc_median_dist
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """
    MODULE TO LOAD 2D/3D COORDINATES AND IMAGE PATCHES FOR TRAINING

    This module accepts point cloud in shape [X x Y]/[X x Y x Z]
    and output dataset that are expected by graphformer (coord, graph
    and image patches for each coordinate).

    Args:
        coord_dir: source of the 3D .tif images masks.
        coord_format: call for random transformation on img and mask data.
        img_dir: source of the 3D .tif file.
        prefix: Prefix name of coordinate file.
        voxal_size: Initial voxal size
        downsampling_if: Number of points in a cloud after which downsamling is run
        drop_rate: Drop rate for voxal size during optimization of voxal size
        downsampling_rate: Value used for downsamling with open3D
        size: numeric value between 0 and 1 for scaling px.
        normalize: type of normalization for img data ["simple", "minmax", "rescale"]
        memory_save: If True data are loaded with memory save mode on
            (~10x faster computation).
    """
    def __init__(self,
                 coord_dir: str,
                 coord_format=[".csv"],
                 img_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 size: Optional[int] = 12,
                 voxal_size=500,
                 downsampling_if=500,
                 drop_rate=1,
                 downsampling_rate: Optional[float] = None,
                 normalize="simple",
                 memory_save=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format[0]

        # Image setting
        self.img_dir = img_dir
        if self.img_dir is not None:
            self.img_format = coord_format[1]

        self.prefix = prefix
        self.size = size
        self.normalize = normalize
        self.memory_save = memory_save

        # Voxal setting
        self.drop_rate = drop_rate
        self.downsampling = downsampling_if
        self.downsampling_rate = downsampling_rate
        self.voxal_size = voxal_size

        self.ids = [f for f in listdir(
            coord_dir) if f.endswith(f'{self.coord_format}')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.coord_format == ".csv":
            coord_file = join(self.coord_dir, str(idx))
        elif self.coord_format == ".npy":
            coord_file = join(self.coord_dir, str(idx))
        elif self.coord_format == ".CorrelationLines.am":
            coord_file = join(self.coord_dir, str(idx))

        if self.img_dir is not None and self.prefix is not None:
            img_idx = idx[:-len(self.prefix + self.coord_format)]
            img_idx = f'{img_idx}{self.img_format}'
        elif self.img_dir is not None and self.prefix is None:
            img_idx = idx[:-len(self.coord_format)]
            img_idx = f'{img_idx}{self.img_format}'
        else:
            img_idx = None

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
        dist = pc_median_dist(pc=coord[:, 1:], avg_over=2)

        if self.img_dir is None:
            coord[:, 1:] = coord[:, 1:] / dist

            VD = VoxalizeDataSetV2(coord=coord,
                                   image=None,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=None,
                                   graph=True)
        else:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=img,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=None,
                                   graph=True)

        coords_v, imgs_v, graph_v, output_idx = VD.voxalize_dataset(out_idx=True,
                                                                    prune=True)
        if self.img_dir is not None:
            for id, c in enumerate(coords_v):
                coords_v[id] = c / dist

        return [c / pc_median_dist(c) for c in coords_v], imgs_v, graph_v, output_idx


class PredictDataset(Dataset):
    def __init__(self,
                 coord_dir: str,
                 coord_format="csv",
                 img_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 size=(12, 12),
                 voxal_size=500,
                 downsampling=500000,
                 drop_rate=1,
                 downsampling_rate=2,
                 normalize="simple",
                 memory_save=True):
        """
        MODULE TO LOAD 2D/3D COORDINATES AND IMAGE PATCHES FOR PREDICTIONS

        This module accepts point cloud in shape [X x Y]/[X x Y x Z]
        and output dataset that are expected by graphformer (coord, graph
        and image patches for each coordinate).

        Build dataset without graph

        Args:
            coord_dir: source of the 3D .tif images masks.
            coord_format: call for random transformation on img and mask data.
            img_dir: source of the 3D .tif file.
            prefix: Prefix name of coordinate file.
            voxal_size: Initial voxal size
            downsampling_if: Number of points in a cloud after which downsamling is run
            drop_rate: Drop rate for voxal size during optimization of voxal size
            downsampling_rate: Value used for downsamling with open3D
            size: numeric value between 0 and 1 for scaling px.
            normalize: type of normalization for img data ["simple", "minmax"]
            memory_save: If True data are loaded with memory save mode on
                (~10x faster computation).
        """
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format

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

        if self.coord_format == ".csv":
            coord_file = join(self.coord_dir, str(idx) + '.csv')
        elif self.coord_format == ".npy":
            coord_file = join(self.coord_dir, str(idx) + '.npy')
        elif self.coord_format == ".am":
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
                                   downsampling_rate=None,
                                   graph=False)
        else:
            VD = VoxalizeDataSetV2(coord=coord,
                                   image=img,
                                   init_voxal_size=self.voxal_size,
                                   drop_rate=self.drop_rate,
                                   downsampling_threshold=self.downsampling,
                                   downsampling_rate=None,
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
