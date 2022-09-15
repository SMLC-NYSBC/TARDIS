from os import listdir, mkdir, getcwd
from os.path import isdir, join, splitext
from shutil import rmtree
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
        downsampling_if: Number of points in a cloud after which downsampling
        drop_rate: Drop rate for voxal size during optimization of voxal size
        downsampling_rate: Value used for downsampling with open3D
        size: numeric value between 0 and 1 for scaling px.
        normalize: type of normalization for img data
            ["simple", "minmax", "rescale"]
        memory_save: If True data are loaded with memory save mode on
            (~10x faster computation).
    """

    def __init__(self,
                 coord_dir: str,
                 coord_format=(".csv"),
                 img_format='.tif',
                 img_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 size: Optional[int] = 12,
                 downsampling_if=500,
                 downsampling_rate: Optional[float] = None,
                 normalize="simple",
                 mesh=False,
                 datatype=True,
                 train=True,
                 memory_save=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format

        self.train = train
        self.cwd = getcwd()
        if self.train:
            if isdir(join(self.cwd, 'temp_train')):
                rmtree(join(self.cwd, 'temp_train'))
            mkdir(join(self.cwd, 'temp_train'))
        else:
            if isdir(join(self.cwd, 'temp_test')):
                rmtree(join(self.cwd, 'temp_test'))
            mkdir(join(self.cwd, 'temp_test'))

        # Image setting
        self.img_dir = img_dir
        if self.img_dir is not None:
            self.img_format = img_format

        self.prefix = prefix
        self.size = size
        self.normalize = normalize
        self.memory_save = memory_save

        self.ids = [f for f in listdir(coord_dir) if f.endswith(self.coord_format)]

        # Voxal setting
        self.downsampling = downsampling_if
        self.downsampling_rate = downsampling_rate
        self.voxal_size = np.zeros((len(self.ids), 1))  # Save voxal size value for speed-up

        # Graph setting
        self.mesh = mesh
        self.datatype = datatype

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            temp = 'temp_train'
        else:
            temp = 'temp_test'
        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.img_dir is not None:
            filetype = [ft for ft in self.coord_format if idx.endswith(ft)][0]

        # Define what image format are available if non set to None
        if self.img_dir is not None and self.prefix is not None:
            img_idx = idx[:-len(self.prefix + filetype)]
            img_idx = f'{img_idx}{self.img_format}'
        elif self.img_dir is not None and self.prefix is None:
            img_idx = idx[:-len(filetype)]
            img_idx = f'{img_idx}{self.img_format}'
        else:
            img_idx = None

        if self.img_dir is not None:
            img_file = join(self.img_dir, str(img_idx))
        else:
            img_file = None

        if self.voxal_size[i, 0] == 0:
            # Pre process coord and image data also, if exist remove duplicates
            coord, img = preprocess_data(coord=coord_file,
                                         datatype=self.datatype,
                                         image=img_file,
                                         include_label=True,
                                         size=self.size,
                                         normalization=self.normalize,
                                         memory_save=self.memory_save)

            # TODO replace normalization with something more robust...
            # TODO Introduce normalization by pixel size resolution for MTs
            # TODO Introduce fix normalization for ScanNet and PartNet datasets
            # Normalize point cloud
            if coord_file.endswith('.ply'):
                dist = pc_median_dist(pc=coord[:, 1:], avg_over=False)
            else:
                dist = pc_median_dist(pc=coord[:, 1:], avg_over=True)

        if self.img_dir is None:
            if self.voxal_size[i, 0] == 0:
                coord[:, 1:] = coord[:, 1:] / dist  # Normalize point cloud

                if self.mesh and self.datatype:
                    classes = coord[:, 0]
                else:
                    classes = None

                VD = VoxalizeDataSetV2(coord=coord,
                                       image=None,
                                       init_voxal_size=0,
                                       drop_rate=1,
                                       downsampling_threshold=self.downsampling,
                                       downsampling_rate=None,
                                       label_cls=classes,
                                       graph=True,
                                       tensor=False)
        else:
            classes = None

            if self.voxal_size[i, 0] == 0:
                VD = VoxalizeDataSetV2(coord=coord,
                                       image=img,
                                       init_voxal_size=0,
                                       drop_rate=1,
                                       downsampling_threshold=self.downsampling,
                                       downsampling_rate=None,
                                       label_cls=classes,
                                       graph=True,
                                       tensor=False)

        if self.voxal_size[i, 0] == 0:
            coords_v, imgs_v, graph_v, output_idx, cls_idx = VD.voxalize_dataset(mesh=self.mesh)

            # save data for faster access later
            np.save(join(self.cwd, temp, f'coord_{i}.npy'), np.asarray(coords_v, dtype=object))
            np.save(join(self.cwd, temp, f'img_{i}.npy'), np.asarray(imgs_v, dtype=object))
            np.save(join(self.cwd, temp, f'graph_{i}.npy'), np.asarray(graph_v, dtype=object))
            np.save(join(self.cwd, temp, f'out_{i}.npy'), np.asarray(output_idx, dtype=object))
            np.save(join(self.cwd, temp, f'cls_{i}.npy'), np.asarray(cls_idx, dtype=object))
        else:
            # Load pre-process data
            coords_v = np.load(join(self.cwd, temp, f'coord_{i}.npy'), allow_pickle=True)
            imgs_v = np.load(join(self.cwd, temp, f'img_{i}.npy'), allow_pickle=True)
            graph_v = np.load(join(self.cwd, temp, f'graph_{i}.npy'), allow_pickle=True)
            output_idx = np.load(join(self.cwd, temp, f'out_{i}.npy'), allow_pickle=True)
            cls_idx = np.load(join(self.cwd, temp, f'cls_{i}.npy'), allow_pickle=True)

        coords_v = [torch.Tensor(co.astype(np.float32)).type(torch.float32) for co in coords_v]
        imgs_v = [torch.Tensor(im.astype(np.float32)).type(torch.float32) for im in imgs_v]
        graph_v = [torch.Tensor(gr.astype(np.float32)).type(torch.float32) for gr in graph_v]
        output_idx = [torch.Tensor(ou.astype(np.float32)).type(torch.int16) for ou in output_idx]
        cls_idx = [torch.Tensor(cx.astype(np.float32)).type(torch.float32) for cx in cls_idx]

        # Store initial patch size for each data to speed up computation
        if self.voxal_size[i, 0] == 0:
            self.voxal_size[i, 0] = VD.voxal_patch_size + 1

        if self.img_dir is not None:
            for id, c in enumerate(coords_v):
                coords_v[id] = c / dist

        return coords_v, imgs_v, graph_v, output_idx, cls_idx


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
            downsampling_if: Number of points in a cloud after which
                downsampling is run
            drop_rate: Drop rate for voxal size during optimization of
                voxal size
            downsampling_rate: Value used for downsampling with open3D
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
