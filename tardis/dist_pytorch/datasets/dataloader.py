#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tardis.dist_pytorch.datasets.augmentation import preprocess_data
from tardis.dist_pytorch.datasets.patches import PatchDataSet
from tardis.utils.errors import TardisError
from tardis.utils.load_data import (load_ply_partnet, load_ply_scannet, load_s3dis_scene)


class BasicDataset(Dataset):
    """
    BASIC CLASS FOR STANDARD DATASET CONSTRUCTION

    Args:
        coord_dir (str): Dataset directory.
        coord_format (tuple, str): A tuple of allowed coord formats.
        patch_if (int): Max number of points per patch.
        train (bool): If True, compute as for training dataset, else test.
    """

    def __init__(self,
                 coord_dir: str,
                 coord_format=".csv",
                 patch_if=500,
                 benchmark=False,
                 train=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format

        self.train = train
        self.benchmark = benchmark

        # Setup environment
        self.cwd = getcwd()
        self.temp = ''

        if self.train:
            if isdir(join(self.cwd, 'temp_train')):
                rmtree(join(self.cwd, 'temp_train'))
            mkdir(join(self.cwd, 'temp_train'))
        else:
            if isdir(join(self.cwd, 'temp_test')):
                rmtree(join(self.cwd, 'temp_test'))
            mkdir(join(self.cwd, 'temp_test'))

        # List of detected coordinates files
        self.ids = [f for f in listdir(coord_dir) if f.endswith(self.coord_format)]

        # Patch build setting
        self.max_point_in_patch = patch_if

        # Save patch size value for speed-up
        self.patch_size = np.zeros((len(self.ids), 1))

    def __len__(self):
        return len(self.ids)

    def save_temp(self,
                  i: int,
                  **kwargs):
        """
        General class function to save temp data.

        Args:
            i (int): Temp index value.
            kwargs (np.ndarray): Dictionary of all arrays to save.
        """

        for key, values in kwargs.items():
            np.save(join(self.cwd, self.temp, f'{key}_{i}.npy'),
                    np.asarray(values, dtype=object))

    def load_temp(self,
                  i: int,
                  **kwargs) -> List[np.ndarray]:
        """
        General class function to load temp data

        Args:
            i (int): Temp index value.
            kwargs (bool): Dictionary of all arrays to load.

        Returns:
            List (list[np.ndarray]): List of kwargs arrays as a tensor arrays.
        """

        return [np.load(join(self.cwd, self.temp, f'{key}_{i}.npy'), allow_pickle=True)
                for key, _ in kwargs.items()]

    @staticmethod
    def list_to_tensor(**kwargs) -> List[List[np.ndarray]]:
        """
        Static class function to transform a list of numpy arrays to a list of
        tensor arrays.

        Args:
            kwargs (np.ndarray): Dictionary of all files to transform into a tensor.

        Returns:
            List (list[torch.Tensor]): List of kwargs arrays as a tensor array.
        """
        return [[torch.Tensor(df.astype(np.float32)).type(torch.float32) for df in value]
                for _, value in kwargs.items()]

    def __getitem__(self,
                    i: int):
        pass


class FilamentDataset(BasicDataset):
    """
    FILAMENT-TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(FilamentDataset, self).__init__(**kwargs)

    def __getitem__(self,
                    i: int) -> Tuple[list, list, list, list, list]:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            coord, _ = preprocess_data(coord=coord_file)

            try:
                px = float(str(idx).split('_')[0])
            except ValueError:
                px = 1

            # Normalize point cloud to pixel size
            coord[:, 1:] = coord[:, 1:] / px
            coord[:, 1:] = coord[:, 1:] / 20  # in nm know distance between points

            VD = PatchDataSet(max_number_of_points=self.max_point_in_patch,
                              tensor=False)
            coords_idx, df_idx, graph_idx, output_idx, _ = VD.patched_dataset(coord=coord)

            # save data for faster access later
            if not self.benchmark:
                self.save_temp(i=i,
                               coords=coords_idx,
                               graph=graph_idx,
                               out=output_idx,
                               df=df_idx)

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx = self.load_temp(i,
                                                                       coords=True,
                                                                       graph=True,
                                                                       out=True,
                                                                       df=True)

        coords_idx, graph_idx, output_idx, df_idx = self.list_to_tensor(coord=coords_idx,
                                                                        graph=graph_idx,
                                                                        output=output_idx,
                                                                        df=df_idx)

        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, df_idx, graph_idx, output_idx, df_idx


class PartnetDataset(BasicDataset):
    """
    PARTNET TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(PartnetDataset, self).__init__(**kwargs)

    def __getitem__(self,
                    i: int) -> Tuple[list, list, list, list, list]:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            coord = load_ply_partnet(coord_file, downscaling=0.05)

            VD = PatchDataSet(drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              tensor=False)

            coords_idx, df_idx, graph_idx, output_idx, _ = VD.patched_dataset(coord=coord,
                                                                              mesh=True,
                                                                              dist_th=0.05)
            # save data for faster access later
            if not self.benchmark:
                self.save_temp(i=i,
                               coords=coords_idx,
                               graph=graph_idx,
                               out=output_idx,
                               df=df_idx)

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx = self.load_temp(i=i,
                                                                       coords=True,
                                                                       graph=True,
                                                                       out=True,
                                                                       df=True)

        coords_idx, graph_idx, output_idx, df_idx = self.list_to_tensor(coord=coords_idx,
                                                                        graph=graph_idx,
                                                                        output=output_idx,
                                                                        df=df_idx)

        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, df_idx, graph_idx, output_idx, df_idx


class ScannetDataset(BasicDataset):
    """
    SCANNET V2 TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(ScannetDataset, self).__init__(**kwargs)

    def __getitem__(self,
                    i: int) -> Tuple[list, list, list, list, list]:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            coord = load_ply_scannet(coord_file, downscaling=0.1)

            VD = PatchDataSet(drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=coord[:, 0],
                              tensor=False)

            coords_idx, df_idx, graph_idx, output_idx, cls_idx = VD.patched_dataset(coord=coord,
                                                                                    mesh=True,
                                                                                    dist_th=0.15)

            if not self.benchmark:
                # save data for faster access later
                self.save_temp(i=i,
                               coords=coords_idx,
                               graph=graph_idx,
                               out=output_idx,
                               df=df_idx,
                               cls=cls_idx)

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.load_temp(i,
                                                                                coords=True,
                                                                                graph=True,
                                                                                out=True,
                                                                                df=True,
                                                                                cls=True)

        coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.list_to_tensor(coord=coords_idx,
                                                                                 graph=graph_idx,
                                                                                 output=output_idx,
                                                                                 df=df_idx,
                                                                                 cls=cls_idx)

        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, df_idx, graph_idx, output_idx, cls_idx


class ScannetColorDataset(BasicDataset):
    """
    SCANNET V2 + COLORS TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        rgb_idx: Numpy or Tensor list of RGB values (N, 3).

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(ScannetColorDataset, self).__init__(**kwargs)
        self.color_dir = join(self.coord_dir, '../../', 'color')

    def __getitem__(self,
                    i: int) -> Tuple[list, list, list, list, list]:
        # Check if color folder exist
        if not isdir(self.color_dir):
            TardisError('12',
                        'tardis/dist_pytorch/datasets/dataloader.py',
                        f'Given dir: {self.color_dir} is not a directory!')

        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            coord, rgb = load_ply_scannet(coord_file,
                                          downscaling=0.1,
                                          color=join(self.color_dir, f'{idx[:-11]}.ply'))

            classes = coord[:, 0]
            VD = PatchDataSet(drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=classes,
                              rgb=rgb,
                              tensor=False)

            coords_idx, rgb_idx, graph_idx, output_idx, cls_idx = VD.patched_dataset(coord=coord,
                                                                                     mesh=True,
                                                                                     dist_th=0.05)

            if not self.benchmark:
                # save data for faster access later
                self.save_temp(i=i,
                               coords=coords_idx,
                               graph=graph_idx,
                               out=output_idx,
                               rgb=rgb_idx,
                               cls=cls_idx)

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, rgb_idx, cls_idx = self.load_temp(i,
                                                                                 coords=True,
                                                                                 graph=True,
                                                                                 out=True,
                                                                                 rgb=True,
                                                                                 cls=True)

        coords_idx, graph_idx, output_idx, rgb_idx, cls_idx = self.list_to_tensor(coord=coords_idx,
                                                                                  graph=graph_idx,
                                                                                  output=output_idx,
                                                                                  rgb=rgb_idx,
                                                                                  cls=cls_idx)

        # Output edge_f,   node_f,  graph,     node_idx,   node_class
        return coords_idx, rgb_idx, graph_idx, output_idx, cls_idx


class Stanford3DDataset(BasicDataset):
    """
    S3DIS TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        rgb_idx: Numpy or Tensor list of RGB values (N, 3).

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 rgb=False,
                 **kwargs):
        super(Stanford3DDataset, self).__init__(**kwargs)

        self.rgb = rgb

        # Modified self.ids to recognise folder for .txt
        area_list = [d for d in listdir(self.coord_dir) if isdir(join(self.coord_dir, d))]
        area_list = [f for f in area_list if not f.startswith('.')]

        self.ids = []
        for i in area_list:
            area = join(self.coord_dir, i)
            room_list = [d for d in listdir(area) if isdir(join(area, d))]
            room_list = [join(i, f) for f in room_list if not f.startswith('.')]

            self.ids.append(room_list)
        self.ids = [item for sublist in self.ids if sublist for item in sublist]

        # Save patch size value for speed-up
        self.patch_size = np.zeros((len(self.ids), 1))

    def __getitem__(self,
                    i: int):
        """ Get list of all coordinates and image patches"""
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Iterate throw every folder and generate S3DIS scene
        coord_file = join(self.coord_dir, idx, 'Annotations')

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            if self.rgb:
                coord, rgb_v = load_s3dis_scene(dir=coord_file, downscaling=0.05,
                                                rgb=True)
            else:
                coord = load_s3dis_scene(dir=coord_file, downscaling=0.05)
            coord[:, 1:] = coord[:, 1:] / 0.05

            if self.rgb:
                VD = PatchDataSet(max_number_of_points=self.max_point_in_patch,
                                  overlap=0,
                                  rgb=rgb_v,
                                  drop_rate=0.1,
                                  tensor=False)
            else:
                VD = PatchDataSet(max_number_of_points=self.max_point_in_patch,
                                  overlap=0,
                                  drop_rate=0.1,
                                  tensor=False)

            coords_idx, df_idx, graph_idx, output_idx, cls_idx = VD.patched_dataset(coord=coord,
                                                                                    mesh=True,
                                                                                    dist_th=0.125)
            graph_idx = [np.where(g >= 2, 1, 0) for g in graph_idx]

            # save data for faster access later
            if not self.benchmark:
                self.save_temp(i=i,
                               coords=coords_idx,
                               graph=graph_idx,
                               out=output_idx,
                               df=df_idx,
                               cls=cls_idx)

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.load_temp(i,
                                                                                coords=True,
                                                                                graph=True,
                                                                                out=True,
                                                                                df=True,
                                                                                cls=True)

        coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.list_to_tensor(coord=coords_idx,
                                                                                 graph=graph_idx,
                                                                                 output=output_idx,
                                                                                 df=df_idx,
                                                                                 cls=cls_idx)

        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, df_idx, graph_idx, output_idx, cls_idx


def build_dataset(dataset_type: str,
                  dirs: list,
                  max_points_per_patch: int,
                  benchmark=False):
    """
    Wrapper for DataLoader

    Function that wraps all data loader and outputs only one asked for depending
    on a dataset

    Args:
        dataset_type (str):  Ask to recognize and process the dataset.
        dirs (list): Ask for a list with the directory given as [train, test].
        max_points_per_patch (int):  Max number of points per patch.
        benchmark (bool): If True construct data for benchmark.er

    Returns:
        Tuple[torch.DataLoader, torch.DataLoader]: Output DataLoader with
        the specified dataset for training and evaluation.
    """

    if dataset_type in ['filament', 'MT', 'Mem']:
        if not benchmark:
            dl_train = FilamentDataset(coord_dir=dirs[0],
                                       coord_format=('.CorrelationLines.am', '.csv'),
                                       patch_if=max_points_per_patch,
                                       train=True)
        dl_test = FilamentDataset(coord_dir=dirs[1],
                                  coord_format=('.CorrelationLines.am', '.csv'),
                                  patch_if=max_points_per_patch,
                                  benchmark=benchmark,
                                  train=False)
    elif dataset_type in ['partnet', 'PartNet']:
        if not benchmark:
            dl_train = PartnetDataset(coord_dir=dirs[0],
                                      coord_format='.ply',
                                      patch_if=max_points_per_patch,
                                      train=True)
        dl_test = PartnetDataset(coord_dir=dirs[1],
                                 coord_format='.ply',
                                 patch_if=max_points_per_patch,
                                 benchmark=benchmark,
                                 train=False)
    elif dataset_type in ['scannet', 'ScanNetV2']:
        if not benchmark:
            dl_train = ScannetDataset(coord_dir=dirs[0],
                                      coord_format='.ply',
                                      patch_if=max_points_per_patch,
                                      train=True)
        dl_test = ScannetDataset(coord_dir=dirs[1],
                                 coord_format='.ply',
                                 patch_if=max_points_per_patch,
                                 benchmark=benchmark,
                                 train=False)
    elif dataset_type == 'scannet_color':
        if not benchmark:
            dl_train = ScannetColorDataset(coord_dir=dirs[0],
                                           coord_format='.ply',
                                           patch_if=max_points_per_patch,
                                           train=True)
        dl_test = ScannetColorDataset(coord_dir=dirs[1],
                                      coord_format='.ply',
                                      patch_if=max_points_per_patch,
                                      benchmark=benchmark,
                                      train=False)
    elif dataset_type in ['stanford', 'S3DIS', 's3dis']:
        if not benchmark:
            dl_train = Stanford3DDataset(coord_dir=dirs[0],
                                         coord_format='.txt',
                                         patch_if=max_points_per_patch,
                                         train=True)
        dl_test = Stanford3DDataset(coord_dir=dirs[1],
                                    coord_format='.txt',
                                    patch_if=max_points_per_patch,
                                    benchmark=benchmark,
                                    train=False)
    elif dataset_type in ['stanford_color', 'S3DIS_color', 's3dis']:
        if not benchmark:
            dl_train = Stanford3DDataset(coord_dir=dirs[0],
                                         coord_format='.txt',
                                         rgb=True,
                                         patch_if=max_points_per_patch,
                                         train=True)
        dl_test = Stanford3DDataset(coord_dir=dirs[1],
                                    coord_format='.txt',
                                    rgb=True,
                                    patch_if=max_points_per_patch,
                                    benchmark=benchmark,
                                    train=False)
    else:
        # TODO General dataloader
        # if not benchmark:
        #     dl_train = GeneralDataset(coord_dir=dirs[1],
        #                               coord_format=('.ply'),
        #                               downsampling_if=downsampling_if,
        #                               downsampling_rate=downsampling_rate,
        #                               train=True)
        # dl_test = GeneralDataset(coord_dir=dirs[1],
        #                          coord_format=('.ply'),
        #                          downsampling_if=downsampling_if,
        #                          downsampling_rate=downsampling_rate,
        #                          train=False)
        pass

    if not benchmark:
        dl_train = DataLoader(dataset=dl_train, shuffle=True, pin_memory=True)
        return dl_train, DataLoader(dataset=dl_test, shuffle=False, pin_memory=True)

    return dl_test
