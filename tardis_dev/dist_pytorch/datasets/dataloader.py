from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis_dev.dist_pytorch.datasets.augmentation import preprocess_data
from tardis_dev.dist_pytorch.datasets.patches import PatchDataSet
from tardis_dev.dist_pytorch.utils.utils import pc_median_dist
from tardis_dev.utils.load_data import load_ply_partnet, load_ply_scannet
from torch.utils.data import DataLoader, Dataset


class BasicDataset(Dataset):
    """
    BASIC CLASS FOR STANDARD DATASET CONSTRUCTION

    Args:
        coord_dir (str): Dataset directory.
        coord_format (tuple): A tuple of allowed coord formats.
        patch_if (int): Max number of points per patch.
        train (bool): If True, compute as for training dataset, else test.
    """

    def __init__(self,
                 coord_dir: str,
                 coord_format=(".csv"),
                 patch_if=500,
                 train=True):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format

        self.train = train

        # Setup environment
        self.cwd = getcwd()
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
        self.patch_size = np.zeros((len(self.ids), 1))  # Save patch size value for speed-up

    def __len__(self):
        return len(self.ids)

    def save_temp(self, i: int, **kwargs):
        """
        General class function to save temp data.

        Args:
            i (int): Temp index value.
            kwargs (np.ndarray): Dictionary of all arrays to save.
        """

        for key, values in kwargs.items():
            np.save(join(self.cwd, self.temp, f'{key}_{i}.npy'),
                    np.asarray(values, dtype=object))

    def load_temp(self, i: int, **kwargs) -> list:
        """
        General class function to load temp data

        Args:
            i (int): Temp index value.
            kwargs (np.ndarray): Dictionary of all arrays to load.

        Returns:
            List (list[np.ndarray]): List of kwargs arrays as a tensor arrays.
        """

        return [np.load(join(self.cwd, self.temp, f'{key}_{i}.npy'), allow_pickle=True)
                for key, _ in kwargs.items()]

    @staticmethod
    def list_to_tensor(**kwargs) -> list:
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


class FilamentDataset(BasicDataset):
    """
    FILAMENT-TYPE DATASET CONSTRUCTION

    Returns:
        List ((list[list[np.ndarray]] or list[list[torch.Tensor]])):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(FilamentDataset, self).__init__(**kwargs)

    def __getitem__(self, i) -> list:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre process coord and image data also, if exist remove duplicates
            coord, _ = preprocess_data(coord=coord_file,
                                       include_label=True,
                                       size=None,
                                       normalization=None)

            # Normalize point cloud
            dist = pc_median_dist(coord[:, 1:])

            if dist is not None:
                coord[:, 1:] = coord[:, 1:] / dist  # Normalize point cloud to px unit

            VD = PatchDataSet(coord=coord,
                              init_patch_size=0,
                              drop_rate=1,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=None,
                              graph=True,
                              tensor=False)
            coords_idx, df_idx, graph_idx, output_idx, _ = VD.patched_dataset()

            # save data for faster access later
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
        List ((list[list[np.ndarray]] or list[list[torch.Tensor]])):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(PartnetDataset, self).__init__(**kwargs)

    def __getitem__(self, i) -> list:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre process coord and image data also, if exist remove duplicates
            coord = load_ply_partnet(coord_file, downsample=0.035)

            VD = PatchDataSet(coord=coord,
                              init_patch_size=0,
                              drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=None,
                              graph=True,
                              tensor=False)

            coords_idx, df_idx, graph_idx, output_idx, _ = VD.patched_dataset(mesh=True,
                                                                              dist_th=0.07)
            # save data for faster access later
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
        List ((list[list[np.ndarray]] or list[list[torch.Tensor]])):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self,
                 **kwargs):
        super(ScannetDataset, self).__init__(**kwargs)

    def __getitem__(self, i) -> list:
        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre process coord and image data also, if exist remove duplicates
            coord = load_ply_scannet(coord_file, downsample=0.1)

            VD = PatchDataSet(coord=coord,
                              init_patch_size=0,
                              drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=coord[:, 0],
                              graph=True,
                              tensor=False)

            coords_idx, df_idx, graph_idx, output_idx, cls_idx = VD.patched_dataset(mesh=True,
                                                                                    dist_th=0.2)
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
        List ((list[list[np.ndarray]] or list[list[torch.Tensor]])):
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

    def __getitem__(self, i):
        assert isdir(self.color_dir)  # Check if color folder exist

        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = 'temp_train'
        else:
            self.temp = 'temp_test'

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre process coord and image data also, if exist remove duplicates
            coord, rgb = load_ply_scannet(coord_file,
                                          downsample=0.1,
                                          color=join(self.color_dir, f'{idx[:-11]}.ply'))

            classes = coord[:, 0]
            VD = PatchDataSet(coord=coord,
                              init_patch_size=0,
                              drop_rate=0.01,
                              max_number_of_points=self.max_point_in_patch,
                              label_cls=classes,
                              rgb=rgb,
                              graph=True,
                              tensor=False)

            coords_idx, rgb_idx, graph_idx, output_idx, cls_idx = VD.patched_dataset(mesh=True,
                                                                                     dist_th=0.2)

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


def build_dataset(dataset_type: list,
                  dirs: str,
                  max_points_per_patch: int):
    """
    WRAPPER FOR DATALOADER

    Function that wraps all data loader and outputs only one asked for depending
    of a dataset

    Args:
        dataset_type (list):
            Ask to recognize and process the dataset.
        dirs (list):
            Ask for a list with the directory given as [train, test].
        patch_if (int):
            Max number of points per patch.

    Returns:
        (torch.DataLoader):

        dl_train:
            Output DataLoader with the specified dataset for training.

        dl_test:
            Output DataLoader with the specified dataset for evaluation.
    """

    if dataset_type == 'filament':
        dl_train = FilamentDataset(coord_dir=dirs[0],
                                   coord_format=('.CorrelationLines.am', '.csv'),
                                   patch_if=max_points_per_patch,
                                   train=True)
        dl_test = FilamentDataset(coord_dir=dirs[1],
                                  coord_format=('.CorrelationLines.am', '.csv'),
                                  patch_if=max_points_per_patch,
                                  train=False)
    elif dataset_type == 'partnet':
        dl_train = PartnetDataset(coord_dir=dirs[0],
                                  coord_format=('.ply'),
                                  patch_if=max_points_per_patch,
                                  train=True)
        dl_test = PartnetDataset(coord_dir=dirs[1],
                                 coord_format=('.ply'),
                                 patch_if=max_points_per_patch,
                                 train=False)
    elif dataset_type == 'scannet':
        dl_train = ScannetDataset(coord_dir=dirs[0],
                                  coord_format=('.ply'),
                                  patch_if=max_points_per_patch,
                                  train=True)
        dl_test = ScannetDataset(coord_dir=dirs[1],
                                 coord_format=('.ply'),
                                 patch_if=max_points_per_patch,
                                 train=False)
    elif dataset_type == 'scannet_color':
        dl_train = ScannetColorDataset(coord_dir=dirs[0],
                                       coord_format=('.ply'),
                                       patch_if=max_points_per_patch,
                                       train=True)
        dl_test = ScannetColorDataset(coord_dir=dirs[1],
                                      coord_format=('.ply'),
                                      patch_if=max_points_per_patch,
                                      train=False)
    else:
        # TODO General dataloader
        # dl_train = GeneralDataset(coord_dir=dirs[1],
        #                           coord_format=('.ply'),
        #                           downsampling_if=downsampling_if,
        #                           downsampling_rate=downsampling_rate,
        #                           train=True)
        # dl_test = GeneralDataset(coord_dir=dirs[1],
        #                          coord_format=('.ply'),
        #                          downsampling_if=downsampling_if,
        #                          downsampling_rate=downsampling_rate,
        #                          train=False)
        pass

    dl_train = DataLoader(dataset=dl_train,
                          batch_size=1,
                          shuffle=True,
                          pin_memory=True)
    dl_test = DataLoader(dataset=dl_test,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True)
    return dl_train, dl_test
