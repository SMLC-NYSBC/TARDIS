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
    BasicDataset is a dataset handling class, mainly designed to manage and preprocess
    coordinate data for tasks requiring large-scale numerical data handling. This
    class initializes the dataset environment, processes input data, and provides
    several utility functions to save, load, and transform data efficiently.

    This class is particularly useful for managing coordinate files, setting
    downscaled versions of files, and utility methods for temporary storage of
    preprocessed data.
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
        """
        Initializes the object with parameters to set up the coordinate directory, format,
        patch size, downscale setting, RGB flag, benchmark flag, and train flag. It also
        handles creation and clearing of temporary directories for training or testing
        based on the provided train flag. Additionally, prepares lists of coordinate files
        and configurations for future processing.

        :param coord_dir: Directory path containing coordinate files.
        :type coord_dir: str or None
        :param coord_format: Specific file format for coordinate files. Defaults to ".csv".
        :param patch_if: Maximum number of points allowed in a patch. Defaults to 500.
        :param downscale: Parameters for downscaling the input. Defaults to None.
        :param rgb: Flag indicating if RGB mode is activated. Defaults to False.
        :param benchmark: Flag to enable or disable benchmark mode. Defaults to False.
        :param train: Flag indicating whether to set up a training or testing environment. Defaults to True.
        """
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

        self.length_i = len(self.ids)

    def __len__(self):
        if self.length_i > 0:
            return self.length_i
        else:
            return 1

    def save_temp(self, i: int, **kwargs):
        """
        Saves temporary data to numpy files.

        This method takes an integer identifier `i` and saves temporary data
        provided in `kwargs` to `.npy` files within a specified directory. Files
        are named using the format `<key>_<i>.npy`.

        :param i: An integer used in naming the saved files.
        :param kwargs: Keyword arguments where keys represent file name prefixes,
            and values are the data to be saved in the numpy files. Each value
            is converted to a numpy array with `object` dtype before saving.
        :return: None
        """

        for key, values in kwargs.items():
            np.save(
                join(self.cwd, self.temp, f"{key}_{i}.npy"),
                np.asarray(values, dtype=object),
            )

    def load_temp(self, i: int, **kwargs) -> List[np.ndarray]:
        """
        Loads temporary numpy array files from the disk. The function can load a single file
        or multiple files depending on the provided keyword arguments. If one keyword
        argument is given, it will load the corresponding file directly. Otherwise, it will
        load all files corresponding to the provided keyword arguments in a list.

        :param i: Index used to identify the file(s) to be loaded.
        :type i: int
        :param kwargs: A set of keyword arguments where the keys represent the prefix of the
            filename(s) to be loaded. Each key corresponds to a possible file.
        :type kwargs: dict
        :return: A list of numpy ndarray objects if multiple keyword arguments are provided,
            or a single numpy ndarray object if only one keyword argument is given.
        :rtype: List[np.ndarray]
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
    def list_to_tensor(**kwargs) -> List:
        """
        Converts a list of numerical dataframes into a list of PyTorch tensors. This method
        takes variable keyword arguments, where each value is expected to be a list of
        dataframes. It processes each dataframe in these lists to convert them into
        PyTorch tensors with a float32 data type, preserving the structure of the input.

        :param kwargs: Variable keyword arguments where each key corresponds to a list of
            pandas DataFrames. Each dataframe is processed and converted to a PyTorch
            tensor with dtype set to `float32`.
        :return: A list of lists containing the converted PyTorch tensors. The structure
            of the returned list parallels the structure of the input keyword arguments.
        :rtype: List
        """
        return [
            [torch.Tensor(df.astype(np.float32)).type(torch.float32) for df in value]
            for _, value in kwargs.items()
        ]

    def __getitem__(self, i: int):
        pass


class FilamentSimulateDataset(BasicDataset):
    """
    Simulates and manages a filament dataset for machine learning tasks.

    This class extends the BasicDataset and provides functionality to generate,
    preprocess, and retrieve synthetic filament data. It supports multiple
    filament simulation types, dataset down-sampling, and patch-based data
    processing. The dataset can be used for training, testing, or validation
    workflows, depending on the configuration.
    """

    def __init__(self, type_s: str, sample_count=50, **kwargs):
        """
        Initializes the FilamentSimulateDataset instance.

        This constructor sets the sample count and type of the dataset instance, and
        initializes the internal PatchDataSet instance with specific parameters for
        point, overlap, drop rate, and graph support. The instance represents the
        structure and configuration for filament simulation dataset generation.

        :param type_s: Specifies the type of the dataset.
        :param sample_count: The number of samples required in the dataset. Defaults to 50.
        :param kwargs: Additional keyword arguments passed to the parent class's
            initialization method.
        """
        super(FilamentSimulateDataset, self).__init__(**kwargs)

        self.sample_count = sample_count
        self.type = type_s

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
        """
        Retrieves a data sample based on the provided index. The method is designed to work with
        simulated datasets, preprocess the data, perform optional downsampling, and construct
        patched datasets for training or evaluation purposes. The returned data includes various
        features for graph-based learning models.

        :param i: The index of the required sample in the dataset.
        :type i: int
        :return: A tuple containing processed data which includes the following:
                 - coords_idx: Coordinates of the node feature indices.
                 - df_idx: Node features for the data.
                 - graph_idx: Graph connectivity information.
                 - output_idx: Node indices for the dataset patch.
                 - df_idx: Class labels for the respective nodes.
        :rtype: Tuple[list, list, list, list, list]
        """
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
    FilamentDataset class for handling filament data processing.

    This class is designed to process filament datasets, including pre-processing of
    coordinate data, image patch creation, and handling normalized data points. It is
    useful for machine learning tasks requiring patched datasets, graph representations,
    and customized preprocessing.
    """

    def __init__(self, **kwargs):
        """
        Initializes an instance of the FilamentDataset class.

        The constructor sets up the dataset with provided parameters and initializes the
        VD attribute to an instance of PatchDataSet. The PatchDataSet instance uses
        configuration parameters such as `max_number_of_points`, `overlap`, `drop_rate`,
        `graph`, and `tensor`.

        :param kwargs: Optional keyword arguments passed to the parent class constructor.
        """
        super(FilamentDataset, self).__init__(**kwargs)
        self.VD = PatchDataSet(
            max_number_of_points=self.max_point_in_patch,
            overlap=0.1,
            drop_rate=0.1,
            graph=True,
            tensor=False,
        )

    def __getitem__(self, i: int) -> Tuple[list, list, list, list, list]:
        """
        Retrieve pre-processed data or processes point cloud data on-the-fly based
        on the provided index, and returns relevant datasets related to graph
        generation, node features, and connectivity.

        This method primarily handles the normalization of point cloud data,
        determining scales, preprocessing coordinates, and caching pre-processed data
        for efficiency. If data for the given index is not pre-processed and cached,
        it is pre-processed and stored. Otherwise, cached data is loaded for quicker
        access. The processed data is further converted into tensors as the final output
        format.

        :param i: Index to retrieve or process the point cloud and associated datasets.
        :type i: int
        :returns: A tuple of lists representing processed dataset components:
                  (coords_index, data_frame_index, graph_index, output_index, data_frame_index).
        :rtype: Tuple[list, list, list, list, list]
        """
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
    PartnetDataset is a specialized dataset class designed to work with 3D point clouds
    and graph representations. It integrates functionalities to preprocess, manage, and
    retrieve data efficiently. This includes handling downscaling, patch generation, and
    dynamic data loading for both training and testing.

    The class leverages the PatchDataSet to create patches of data points with specified
    parameters like maximum points, overlap, and drop rate. Additionally, it supports
    temporary storage and loading for optimized access.
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
        """
        Retrieve and process data based on the specified index. This function handles the
        loading or computational creation of data patches required for further processing.
        It manages data scaling, duplication removal, and pre-processing for coordinates
        and graphs. Depending on the patch size and given configurations, it either
        generates the relevant data or loads pre-processed data that was saved earlier
        for efficiency.

        :param i: The index of the dataset to be accessed.
        :type i: int
        :returns: A tuple containing processed data:
            - coords_idx: Processed coordinates indexed tensor.
            - df_idx: Dataframe index tensor.
            - graph_idx: Indexed tensor representation of the graph.
            - output_idx: Indexed tensor for output node identification.
            - df_idx: Dataframe index tensor again for consistency.
        :rtype: Tuple[list, list, list, list, list]
        :raises FileNotFoundError: If the coordinate file for the specified dataset index
            cannot be located in the provided directory.
        """
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
    ScannetDataset handles loading and processing of 3D scan data,
    specifically for tasks involving graph and patch-based methods
    in the context of 3D scene understanding.

    This class is a specialized dataset loader that extends the
    BasicDataset. It can handle a range of tasks including
    preprocessing of scan data, handling coordinate formats,
    and managing batch-level operations for training and testing
    workflows. The class also supports saving and retrieving
    preprocessed temporary data for faster data loading.
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
        """
        Retrieves an indexed sample from the dataset and processes it for use in training
        or testing. The processing includes scaling, coordinate handling, patch generation,
        and caching for efficiency. Returns various processed components of the dataset
        including nodes, edges, graph structure, indexes, and classes.

        :param i: The index of the sample to retrieve and process.
        :type i: int
        :return: A tuple containing the following elements:
            - coords_idx (Tensor): Processed coordinates of nodes.
            - df_idx (Tensor): Feature data for each node in the graph.
            - graph_idx (Tensor): Graph structure defining edges and connectivity.
            - output_idx (Tensor): Indexes of nodes relevant for output computation.
            - cls_idx (Tensor): Classes or labels associated with each node.
        :rtype: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        """
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
    Represents a dataset for Scannet, inheriting functionalities from the
    BasicDataset. It supports handling of color information, coordination
    files, and patch-based data processing.

    This dataset class processes 3D point cloud data and corresponding color
    information from Scannet using various custom functionalities, including
    coordinate scaling, patch-based segmentation, and preprocessed data
    caching for efficient usage. It also incorporates graph-based dataset
    representations.
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
        """
        Retrieves a set of pre-processed coordinates, image patches, graph data,
        and labels (including classes) for a specific index in the dataset. This
        allows access to data such as spatial coordinates, corresponding RGB
        values, and graph-related information. The dataset can handle training and
        testing variations, apply downscaling, and load or save intermediate results
        for efficiency. The function supports transformations from list-based to
        tensor-based formats and removes duplicates during pre-processing.

        :param i: Index of the dataset item to retrieve data for.
        :type i: int
        :return: A tuple containing:
            - Coordinates tensor (list): Spatial coordinates tensor for the graph.
            - RGB tensor (list): Color information tensor corresponding to coordinates.
            - Graph tensor (list): Graph adjacency or connectivity tensor for the given dataset item.
            - Node index tensor (list): Tensor representing various node indices.
            - Node class tensor (list): Tensor with the class labels of the nodes.
        :rtype: Tuple[list, list, list, list, list]
        """
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
    This class represents the Stanford 3D dataset, which handles dataset processing, room identification,
    and patch data preparation. It is responsible for organizing room data from directories and subdirectories,
    preprocessing 3D point cloud data for machine learning tasks, and optimizing computational resources
    by leveraging patches. The class integrates with the PatchDataSet for patch extraction and manipulation.
    """

    def __init__(self, **kwargs):
        """
        Represents a dataset class designed for processing and managing 3D Stanford
        dataset files. This class initializes parameters related to directories,
        processes folder structures to recognize valid dataset areas and rooms, and
        builds dataset identifiers. Moreover, it initializes a PatchDataSet instance
        to assist in handling patches with defined configurations for future usage.

        :param kwargs: Arbitrary keyword arguments passed during initialization.
        """
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
        """
        Get the item corresponding to the provided index. This method handles loading
        and processing of S3DIS scene data, including coordinate and optional RGB
        features.

        Depending on whether the train mode is enabled, data is preprocessed or loaded
        from previously saved temporary files. The method either directly loads or
        downscales the data with different scaling approaches. It further splits and
        converts the data into tensors suitable for model input.

        :param i: Index of the item to be retrieved.
        :type i: int
        :return: Depending on context, returns either:
            - idx, coord, coords_idx, node_idx, graph_idx, output_idx, cls_idx (if
              benchmark is enabled)
            - coordinates index tensor, node index tensor, graph tensor, output
              tensor, and class index tensor (if benchmark is disabled).
        :rtype: Tuple containing the described return values.
        """
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
                            dir_s=coord_file, downscaling=float(scale[1]), rgb=True
                        )
                    else:
                        coord, rgb_v = load_s3dis_scene(
                            dir_s=coord_file, random_ds=float(scale[1]), rgb=True
                        )
                else:
                    coord, rgb_v = load_s3dis_scene(
                        dir_s=coord_file, downscaling=0, rgb=True
                    )
            else:
                if self.downscale is not None:
                    scale = self.downscale.split("_")
                    if scale[0] == "v":
                        coord = load_s3dis_scene(
                            dir_s=coord_file, downscaling=float(scale[1])
                        )
                    else:
                        coord = load_s3dis_scene(
                            dir_s=coord_file, random_ds=float(scale[1])
                        )
                else:
                    coord = load_s3dis_scene(dir_s=coord_file, downscaling=0)

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
    Build dataset objects and corresponding data loaders based on the specified dataset
    type and configurations. Supports various dataset types for training and testing such
    as simulated datasets, Filament datasets, PartNet, ScanNet, Stanford datasets, and more.
    The function also handles benchmark mode for testing datasets only. Depending on the
    `dataset_type`, appropriate dataset classes are instantiated, configured with training
    and testing attributes, and wrapped in data loaders for model consumption.

    :param dataset_type: Specifies the type of dataset to load. Can be either a string or
        a list. If it is a list, it must define simulation-related metadata. For strings,
        supported values include "filament", "MT", "Mem", "partnet", "scannet",
        "stanford", and their variations (e.g., "scannet_rgb", "stanford_rgb").
    :param dirs: A list containing directory paths to load training and testing dataset
        files from. The directories depend on the specific type of dataset being used.
    :param max_points_per_patch: The maximum number of points that a single patch within
        the dataset can contain. This value is used to configure dataset-specific patch
        loading and processing.
    :param downscale: Defines optional downscaling operations to be applied during
        dataset loading to alter point cloud resolution. If not provided, defaults are
        used based on the dataset type.
    :param benchmark: If set to True, indicates benchmark testing mode. In this mode,
        only the test dataset loader is returned, and the training loader setup is bypassed.

    :return: For non-benchmark mode, returns a tuple where the first element is the
        training dataset loader instance and the second element is the testing dataset
        loader instance. For benchmark mode, only the testing dataset loader is returned.

    """

    if isinstance(dataset_type, list):
        assert len(dataset_type) == 4
        if dataset_type[0] == "simulate":
            if not benchmark:
                dl_train = FilamentSimulateDataset(
                    type_s=dataset_type[1],
                    sample_count=int(dataset_type[2]),
                    patch_if=max_points_per_patch,
                    train=True,
                    downscale=downscale,
                )
            dl_test = FilamentSimulateDataset(
                type_s=dataset_type[1],
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
