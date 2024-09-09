#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tardis_em.dist_pytorch.datasets.augmentation import preprocess_data
from tardis_em.dist_pytorch.datasets.patches import PatchDataSet
from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import (
    load_ply_partnet,
    load_ply_scannet,
    load_s3dis_scene,
)
from tardis_em.dist_pytorch.utils.utils import (
    RandomDownSampling,
    VoxelDownSampling,
)
from tardis_em.dist_pytorch.utils.utils import pc_median_dist
from tardis_em.dist_pytorch.utils.build_point_cloud import create_simulated_dataset
from tardis_em.analysis.filament_utils import sort_segment


class BasicDataset(Dataset):
    """
    BASIC CLASS FOR STANDARD DATASET CONSTRUCTION

    Args:
        coord_dir (str): Dataset directory.
        coord_format (tuple, str): A tuple of allowed coord formats.
        patch_if (int): Max number of points per patch.
        train (bool): If True, compute as for training dataset, else test.
    """

    def __init__(
        self,
        coord_dir=None,
        coord_format=".csv",
        patch_if=500,
        downscale=None,
        rgb=False,
        benchmark=False,
        train=True,
    ):
        # Coord setting
        self.coord_dir = coord_dir
        self.coord_format = coord_format
        self.rgb = rgb

        self.downscale = downscale

        self.train = train
        self.benchmark = benchmark

        # Setup environment
        self.cwd = getcwd()
        self.temp = ""

        if self.train:
            if isdir(join(self.cwd, "temp_train")):
                rmtree(join(self.cwd, "temp_train"))
            mkdir(join(self.cwd, "temp_train"))
        else:
            if isdir(join(self.cwd, "temp_test")):
                rmtree(join(self.cwd, "temp_test"))
            mkdir(join(self.cwd, "temp_test"))

        # List of detected coordinates files
        if isinstance(coord_dir, str):
            self.ids = [f for f in listdir(coord_dir) if f.endswith(self.coord_format)]

            # Save patch size value for speed-up
            self.patch_size = np.zeros((len(self.ids), 1))
            self.VD = None
        else:
            self.ids = []

        # Patch build setting
        self.max_point_in_patch = patch_if

        self.l_ = len(self.ids)

    def __len__(self):
        if self.l_ > 0:
            return self.l_
        else:
            return 1

    def save_temp(self, i: int, **kwargs):
        """
        General class function to save temp data.

        Args:
            i (int): Temp index value.
            kwargs (np.ndarray): Dictionary of all arrays to save.
        """

        for key, values in kwargs.items():
            np.save(
                join(self.cwd, self.temp, f"{key}_{i}.npy"),
                np.asarray(values, dtype=object),
            )

    def load_temp(self, i: int, **kwargs) -> List[np.ndarray]:
        """
        General class function to load temp data

        Args:
            i (int): Temp index value.
            kwargs (bool): Dictionary of all arrays to load.

        Returns:
            List (list[np.ndarray]): List of kwargs arrays as a tensor arrays.
        """
        if len(kwargs.items()) == 1:
            return np.load(
                join(self.cwd, self.temp, f"{list(kwargs.items())[0][0]}_{i}.npy"),
                allow_pickle=True,
            )
        return [
            np.load(join(self.cwd, self.temp, f"{key}_{i}.npy"), allow_pickle=True)
            for key, _ in kwargs.items()
        ]

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
        return [
            [torch.Tensor(df.astype(np.float32)).type(torch.float32) for df in value]
            for _, value in kwargs.items()
        ]

    def __getitem__(self, i: int):
        pass


class FilamentSimulateDataset(BasicDataset):
    """
    SIMULATED FILAMENT-TYPE DATASET CONSTRUCTION

    Returns:
        Tuple (list[np.ndarray]):
        coords_idx: Numpy or Tensor list of coordinates (N, (2, 3)).

        df_idx: Normalize zero-out output for standardized dummy.

        graph_idx: Numpy or Tensor list of 2D GT graphs.

        output_idx: Numpy or Tensor list (N, 1) of output index value.

        df_idx: Normalize zero-out output for standardized dummy.
    """

    def __init__(self, type_: str, sample_count=50, **kwargs):
        super(FilamentSimulateDataset, self).__init__(**kwargs)

        self.sample_count = sample_count
        self.type = type_

        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.15,
            drop_rate=0.1,
            graph=True,
            tensor=True,
        )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        """Get list of all coordinates and image patches"""
        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Simulate filament dataset
        if self.type in ["mix3d", "membranes"]:
            coord_file = create_simulated_dataset(
                size=list(np.random.randint((50, 512, 512), (60, 640, 640))),
                sim_type=self.type,
            )
            mesh = 8
        else:
            coord_file = create_simulated_dataset(
                size=list(np.random.randint((1, 512, 512), (250, 4096, 4096))),
                sim_type=self.type,
            )
            mesh = 2
        # Pre-process coord and image data also, if exist remove duplicates
        coord, _ = preprocess_data(coord=coord_file)

        # Optional Down-sampling of simulated dataset
        if self.downscale is not None:
            scale = self.downscale.split("_")

            if scale[1] == "random":
                if scale[0] == "v":
                    scale[1] = np.random.randint(1, 15)
                else:
                    scale[1] = np.random.randint(25, 100) / 100
            else:
                scale[1] = float(scale[1])

            if scale[0] == "v":
                down_scale = VoxelDownSampling(voxel=scale[1], labels=True, KNN=True)
            else:
                down_scale = RandomDownSampling(
                    threshold=scale[1], labels=True, KNN=True
                )
            coord = down_scale(coord)

            coord = coord[coord[:, 0].argsort()]
            df_coord = []
            for i in np.unique(coord[:, 0]):
                id_ = i
                idx = np.where(coord[:, 0] == id_)[0]

                if len(idx) > 3:
                    df_coord.append(
                        np.hstack(
                            (
                                np.repeat(id_, len(idx)).reshape(-1, 1),
                                sort_segment(coord[idx, 1:]),
                            )
                        )
                    )
            coord = np.concatenate(df_coord)

        # Normalize distance
        # coord[:, 1:] = coord[:, 1:] / pc_median_dist(coord[:, 1:], True)

        # Patch dataset
        if self.train:
            coords_idx, df_idx, graph_idx, output_idx, _ = self.VD.patched_dataset(
                coord=coord, mesh=mesh, random=True
            )
        else:
            coords_idx, df_idx, graph_idx, output_idx, _ = self.VD.patched_dataset(
                coord=coord, mesh=mesh
            )

        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, df_idx, graph_idx, output_idx, df_idx


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

    def __init__(self, **kwargs):
        super(FilamentDataset, self).__init__(**kwargs)
        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.1,
            drop_rate=0.1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        """Get list of all coordinates and image patches"""
        idx = self.ids[i]

        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            # Pre-process coord and image data also, if exist remove duplicates
            coord, _ = preprocess_data(coord=coord_file)

            try:
                px = float(str(idx).split("_")[0])
            except ValueError:
                px = 1

            # Normalize point cloud to pixel size
            if self.downscale is None:
                scale = 1
            else:
                scale = self.downscale
            coord[:, 1:] = coord[:, 1:] / px  # Normalize for pixel size
            coord[:, 1:] = coord[:, 1:] / scale  # in nm know distance between points

            coords_idx, df_idx, graph_idx, output_idx, _ = self.VD.patched_dataset(
                coord=coord, mesh=2
            )

            # save data for faster access later
            if not self.benchmark:
                self.save_temp(
                    i=i, coords=coords_idx, graph=graph_idx, out=output_idx, df=df_idx
                )

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx = self.load_temp(
                i, coords=True, graph=True, out=True, df=True
            )

        coords_idx, graph_idx, output_idx, df_idx = self.list_to_tensor(
            coord=coords_idx, graph=graph_idx, output=output_idx, df=df_idx
        )

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

    def __init__(self, **kwargs):
        super(PartnetDataset, self).__init__(**kwargs)
        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.1,
            drop_rate=0.1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        """Get list of all coordinates and image patches"""
        idx = self.ids[i]

        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            if self.downscale is None:
                scale = 0.05
            else:
                scale = self.downscale

            # Pre-process coord and image data also, if exist remove duplicates
            coord = load_ply_partnet(coord_file, downscaling=scale)
            coord[:, 1:] = coord[:, 1:] / scale

            coords_idx, df_idx, graph_idx, output_idx, _ = self.VD.patched_dataset(
                coord=coord, mesh=6
            )
            # save data for faster access later
            if not self.benchmark:
                self.save_temp(
                    i=i, coords=coords_idx, graph=graph_idx, out=output_idx, df=df_idx
                )

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx = self.load_temp(
                i=i, coords=True, graph=True, out=True, df=True
            )

        coords_idx, graph_idx, output_idx, df_idx = self.list_to_tensor(
            coord=coords_idx, graph=graph_idx, output=output_idx, df=df_idx
        )

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

    def __init__(self, **kwargs):
        super(ScannetDataset, self).__init__(**kwargs)
        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.1,
            drop_rate=0.1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        """Get list of all coordinates and image patches"""
        idx = self.ids[i]

        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            if self.downscale is None:
                scale = 0.05
            else:
                scale = self.downscale

            # Pre-process coord and image data also, if exist remove duplicates
            coord = load_ply_scannet(coord_file, downscaling=scale)
            coord[:, 1:] = coord[:, 1:] / scale

            (
                coords_idx,
                df_idx,
                graph_idx,
                output_idx,
                cls_idx,
            ) = self.VD.patched_dataset(coord=coord, mesh=6)

            if not self.benchmark:
                # save data for faster access later
                self.save_temp(
                    i=i,
                    coords=coords_idx,
                    graph=graph_idx,
                    out=output_idx,
                    df=df_idx,
                    cls=cls_idx,
                )

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.load_temp(
                i, coords=True, graph=True, out=True, df=True, cls=True
            )

        coords_idx, graph_idx, output_idx, df_idx, cls_idx = self.list_to_tensor(
            coord=coords_idx, graph=graph_idx, output=output_idx, df=df_idx, cls=cls_idx
        )

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

    def __init__(self, **kwargs):
        super(ScannetColorDataset, self).__init__(**kwargs)
        self.color_dir = join(self.coord_dir, "../../", "color")
        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.1,
            drop_rate=0.1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        # Check if color folder exist
        if not isdir(self.color_dir):
            TardisError(
                "12",
                "tardis_em/dist_pytorch/datasets/dataloader.py",
                f"Given dir: {self.color_dir} is not a directory!",
            )

        """ Get list of all coordinates and image patches """
        idx = self.ids[i]

        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Define what coordinate format are available
        coord_file = join(self.coord_dir, str(idx))

        if self.patch_size[i, 0] == 0:
            if self.downscale is None:
                scale = 0.05
            else:
                scale = self.downscale

            # Pre-process coord and image data also, if exist remove duplicates
            coord, rgb = load_ply_scannet(
                coord_file,
                downscaling=scale,
                color=join(self.color_dir, f"{idx[:-11]}.ply"),
            )
            coord[:, 1:] = coord[:, 1:] / scale
            classes = coord[:, 0]

            (
                coords_idx,
                rgb_idx,
                graph_idx,
                output_idx,
                cls_idx,
            ) = self.VD.patched_dataset(coord=coord, label_cls=classes, rgb=rgb, mesh=6)

            if not self.benchmark:
                # save data for faster access later
                self.save_temp(
                    i=i,
                    coords=coords_idx,
                    graph=graph_idx,
                    out=output_idx,
                    rgb=rgb_idx,
                    cls=cls_idx,
                )

                # Store initial patch size for each data to speed up computation
                self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            coords_idx, graph_idx, output_idx, rgb_idx, cls_idx = self.load_temp(
                i, coords=True, graph=True, out=True, rgb=True, cls=True
            )

        coords_idx, graph_idx, output_idx, rgb_idx, cls_idx = self.list_to_tensor(
            coord=coords_idx,
            graph=graph_idx,
            output=output_idx,
            rgb=rgb_idx,
            cls=cls_idx,
        )

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

    def __init__(self, **kwargs):
        super(Stanford3DDataset, self).__init__(**kwargs)

        # Modified self.ids to recognize folder for .txt
        area_list = [
            d for d in listdir(self.coord_dir) if isdir(join(self.coord_dir, d))
        ]
        area_list = [f for f in area_list if not f.startswith(".")]

        self.ids = []
        for i in area_list:
            area = join(self.coord_dir, i)
            room_list = [d for d in listdir(area) if isdir(join(area, d))]
            room_list = [join(i, f) for f in room_list if not f.startswith(".")]

            self.ids.append(room_list)
        self.ids = [item for sublist in self.ids if sublist for item in sublist]

        # Save patch size value for speed-up
        self.patch_size = np.zeros((len(self.ids), 1))

        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0,
            drop_rate=1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int):
        """Get list of all coordinates and image patches"""
        idx = self.ids[i]

        if self.train:
            self.temp = "temp_train"
        else:
            self.temp = "temp_test"

        # Iterate throw every folder and generate S3DIS scene
        coord_file = join(self.coord_dir, idx, "Annotations")

        if self.patch_size[i, 0] == 0:
            print(f"Loading: {idx}")
            # start = time.time()
            # Pre-process coord and image data also, if exist remove duplicates
            if self.rgb:
                if self.downscale is not None:
                    scale = self.downscale.split("_")
                    if scale[0] == "v":
                        coord, rgb_v = load_s3dis_scene(
                            dir_=coord_file, downscaling=float(scale[1]), rgb=True
                        )
                    else:
                        coord, rgb_v = load_s3dis_scene(
                            dir_=coord_file, random_ds=float(scale[1]), rgb=True
                        )
                else:
                    coord, rgb_v = load_s3dis_scene(
                        dir_=coord_file, downscaling=0, rgb=True
                    )
            else:
                if self.downscale is not None:
                    scale = self.downscale.split("_")
                    if scale[0] == "v":
                        coord = load_s3dis_scene(
                            dir_=coord_file, downscaling=float(scale[1])
                        )
                    else:
                        coord = load_s3dis_scene(
                            dir_=coord_file, random_ds=float(scale[1])
                        )
                else:
                    coord = load_s3dis_scene(dir_=coord_file, downscaling=0)

            if self.downscale is not None:
                if scale[0] == "v":
                    coord[:, 1:] = coord[:, 1:] / float(scale[1])
                else:
                    coord[:, 1:] = coord[:, 1:] / round(
                        pc_median_dist(coord[:, 1:], avg_over=True, box_size=0.25), 3
                    )

            if self.rgb:
                (
                    coords_idx,
                    node_idx,
                    graph_idx,
                    output_idx,
                    cls_idx,
                ) = self.VD.patched_dataset(coord=coord, rgb=rgb_v, mesh=8, random=True)
            else:
                (
                    coords_idx,
                    node_idx,
                    graph_idx,
                    output_idx,
                    cls_idx,
                ) = self.VD.patched_dataset(coord=coord, mesh=8, random=True)

            # save data for faster access later
            if not self.benchmark:
                if self.rgb:
                    self.save_temp(i=i, coord=coord, rgb=rgb_v)
                else:
                    self.save_temp(i=i, coord=coord)

            # Store initial patch size for each data to speed up computation
            self.patch_size[i, 0] = 1
        else:
            # Load pre-process data
            if self.rgb:
                coord, rgb_v = self.load_temp(i, coord=True, rgb=True)
            else:
                coord = self.load_temp(i, coord=True)

            if self.rgb:
                (
                    coords_idx,
                    node_idx,
                    graph_idx,
                    output_idx,
                    cls_idx,
                ) = self.VD.patched_dataset(coord=coord, rgb=rgb_v, mesh=8, random=True)
            else:
                (
                    coords_idx,
                    node_idx,
                    graph_idx,
                    output_idx,
                    cls_idx,
                ) = self.VD.patched_dataset(coord=coord, mesh=8, random=True)

        coords_idx, graph_idx, output_idx, node_idx, cls_idx = self.list_to_tensor(
            coord=coords_idx,
            graph=graph_idx,
            output=output_idx,
            df=node_idx,
            cls=cls_idx,
        )

        if self.benchmark:
            # Output file_name, raw_coord, edge_f, node_f, graph, node_idx, node_class
            return idx, coord, coords_idx, node_idx, graph_idx, output_idx, cls_idx
        # Output edge_f,   node_f, graph,     node_idx,   node_class
        return coords_idx, node_idx, graph_idx, output_idx, cls_idx


def build_dataset(
    dataset_type: Union[str, list],
    dirs: list,
    max_points_per_patch: int,
    downscale=None,
    benchmark=False,
):
    """
    Wrapper for DataLoader

    Function that wraps all data loaders and outputs only one asked for depending
    on a dataset

    Args:
        dataset_type (str): Ask to recognize and process the dataset.
        dirs (list): Ask for a list with the directory given as [train, test].
        max_points_per_patch (int): Max number of points per patch.
        downscale (None, float): Overweight downscale factor
        benchmark (bool): If True construct data for benchmark

    Returns:
        Tuple[torch.DataLoader, torch.DataLoader]: Output DataLoader with
        the specified dataset for training and evaluation.
    """

    if isinstance(dataset_type, list):
        assert len(dataset_type) == 4
        if dataset_type[0] == "simulate":
            if not benchmark:
                dl_train = FilamentSimulateDataset(
                    type_=dataset_type[1],
                    sample_count=int(dataset_type[2]),
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = FilamentSimulateDataset(
                type_=dataset_type[1],
                sample_count=int(dataset_type[3]),
                patch_if=max_points_per_patch,
                train=False,
                downscale=downscale,
            )
    else:
        if dataset_type in ["filament", "MT", "Mem"]:
            if not benchmark:
                dl_train = FilamentDataset(
                    coord_dir=dirs[0],
                    coord_format=(".CorrelationLines.am", ".csv"),
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = FilamentDataset(
                coord_dir=dirs[1],
                coord_format=(".CorrelationLines.am", ".csv"),
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
        elif dataset_type in ["partnet", "PartNet"]:
            if not benchmark:
                dl_train = PartnetDataset(
                    coord_dir=dirs[0],
                    coord_format=".ply",
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = PartnetDataset(
                coord_dir=dirs[1],
                coord_format=".ply",
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
        elif dataset_type in ["scannet", "ScanNetV2"]:
            if not benchmark:
                dl_train = ScannetDataset(
                    coord_dir=dirs[0],
                    coord_format=".ply",
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = ScannetDataset(
                coord_dir=dirs[1],
                coord_format=".ply",
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
        elif dataset_type == "scannet_rgb":
            if not benchmark:
                dl_train = ScannetColorDataset(
                    coord_dir=dirs[0],
                    coord_format=".ply",
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = ScannetColorDataset(
                coord_dir=dirs[1],
                coord_format=".ply",
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
        elif dataset_type in ["stanford", "S3DIS"]:
            if not benchmark:
                dl_train = Stanford3DDataset(
                    coord_dir=dirs[0],
                    coord_format=".txt",
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = Stanford3DDataset(
                coord_dir=dirs[1],
                coord_format=".txt",
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
        elif dataset_type in ["stanford_rgb", "S3DIS_rgb"]:
            if not benchmark:
                dl_train = Stanford3DDataset(
                    coord_dir=dirs[0],
                    coord_format=".txt",
                    rgb=True,
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = Stanford3DDataset(
                coord_dir=dirs[1],
                coord_format=".txt",
                rgb=True,
                patch_if=max_points_per_patch,
                benchmark=benchmark,
                train=False,
                downscale=downscale,
            )
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
        return DataLoader(dataset=dl_train, shuffle=True, pin_memory=True), DataLoader(
            dataset=dl_test, shuffle=False, pin_memory=True
        )
    return dl_test
