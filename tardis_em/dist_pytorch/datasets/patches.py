#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Tuple, Union, List

import numpy as np
import torch

from tardis_em.dist_pytorch.datasets.augmentation import BuildGraph
from tardis_em.utils.errors import TardisError


class PatchDataSet:
    """
    Represents a class designed for configuring and processing point cloud data. This includes
    functionalities for down-sampling points, defining overlapping feature patches, and specifying
    output formats such as tensor representation or graph structure. It also initializes and manages
    patch size configurations.
    """

    def __init__(
        self,
        max_number_of_points=500,
        overlap=0.15,
        drop_rate=None,
        graph=True,
        tensor=True,
    ):
        """
        Class that initializes the configuration for point cloud down-sampling and patch processing. It sets up
        parameters for down-sampling the point cloud, configuring the feature patches, and determining output
        formats such as tensor or graph. It also provides initial space for the patch size in the configuration.

        :param max_number_of_points: The maximum number of points for point cloud down-sampling.
        :param overlap: A floating-point value specifying the overlap percentage
            for creating patches.
        :param drop_rate: Floating-point value representing the drop rate for feature reduction.
        :param graph: Boolean to enable or disable graph-based output processing.
        :param tensor: Boolean to enable or disable tensor-based output processing.
        """
        # Point cloud down-sampling setting
        self.DOWNSAMPLING_TH = max_number_of_points

        # Patch setting
        self.voxel = None
        self.drop_rate = drop_rate
        self.TORCH_OUTPUT = tensor
        self.GRAPH_OUTPUT = graph
        self.EXPAND = 0.1  # Expand the boundary box by 10%
        self.STRIDE = overlap  # Create 15% overlaps between patches

        # Initialization
        self.INIT_PATCH_SIZE = np.zeros((2, 3))

    def boundary_box(self, coord, offset=None) -> np.ndarray:
        """
        Calculate the boundary box for the given coordinates, optionally applying an
        offset. The function computes the minimum and maximum dimensions along each
        axis (x, y, z if applicable). If the offset is provided, the boundary box is
        expanded proportionally around the center. The result is a NumPy array
        representing the boundary box with two points: the lower and upper bounds.

        :param coord: The input array of coordinates where rows represent points, and
                      columns represent dimensional axes (x, y, and optionally z).
        :type coord: np.ndarray
        :param offset: An optional expansion multiplier for the bounding box dimensions.
                       If provided, it scales the boundary box's size relative to its
                       center.
        :type offset: float, optional
        :return: A NumPy array representing the boundary box with two rows: the minimum
                 (x, y, z) coordinates and the maximum (x, y, z) coordinates.
        :rtype: np.ndarray
        """
        # Define x,y and (z) min and m
        # ax sizes
        if coord.shape[1] == 3:
            min_x, min_y, min_z = np.min(coord, axis=0)
            max_x, max_y, max_z = np.max(coord, axis=0)
        else:
            min_x, min_y = np.min(coord, axis=0)
            max_x, max_y = np.max(coord, axis=0)
            min_z, max_z = 0, 0

        if offset is not None:
            self.EXPAND = offset

        dx = ((min_x + max_x) / 2) - min_x
        min_x, max_x = min_x - dx * self.EXPAND, max_x + dx * self.EXPAND

        dy = ((min_y + max_y) / 2) - min_y
        min_y, max_y = min_y - dy * self.EXPAND, max_y + dy * self.EXPAND

        dz = ((min_z + max_z) / 2) - min_z
        min_z, max_z = min_z - dz * self.EXPAND, max_z + dz * self.EXPAND

        dx, dy = abs(min_x - max_x), abs(min_y - max_y)
        d = dx if max(dx, dy) == 0 else dy

        self.drop_rate = d * 0.01
        return np.array([(min_x, min_y, min_z), (max_x, max_y, max_z)])

    @staticmethod
    def center_patch(bbox, voxel_size=1) -> np.ndarray:
        """
        Calculates the coordinates for the centers of voxels within a given bounding box.
        The bounding box is divided into a three-dimensional grid using the specified voxel size.
        The method ensures a minimum of 2 voxels along each axis to avoid degenerate cases.
        The voxel centers are returned as a NumPy array containing their coordinates.

        :param bbox: The bounding box defined as a 2x3 NumPy array where the first row contains
            the minimum x, y, z coordinates and the second row contains the maximum x, y, z
            coordinates.
        :type bbox: np.ndarray
        :param voxel_size: The size of each voxel along all dimensions. Defaults to 1.
        :type voxel_size: float
        :return: A NumPy array with the coordinates of all voxel centers as rows.
        :rtype: np.ndarray
        """
        # Calculate the number of voxels along each axis
        n_x = int(np.ceil((bbox[1, 0] - bbox[0, 0]) / voxel_size))
        n_y = int(np.ceil((bbox[1, 1] - bbox[0, 1]) / voxel_size))
        n_z = int(np.ceil((bbox[1, 2] - bbox[0, 2]) / voxel_size)) // 2

        # Calculate the coordinates of the voxel centers
        if n_x < 2:
            n_x = 2
        x = np.linspace(bbox[0, 0] - voxel_size / 2, bbox[1, 0] + voxel_size / 2, n_x)

        if n_y < 2:
            n_y = 2
        y = np.linspace(bbox[0, 1] - voxel_size / 2, bbox[1, 1] + voxel_size / 2, n_y)

        if n_z < 2:
            n_z = 2
        z = np.linspace(bbox[0, 2] - voxel_size / 2, bbox[1, 2] + voxel_size / 2, n_z)

        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        voxel_centers = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))

        return voxel_centers

    def points_in_patch(self, coord: np.ndarray, patch_center: np.ndarray) -> bool:
        """
        Determines if specific points represented by their coordinates are located within a
        defined patch based on the patch's center and pre-defined size. The patch is modeled
        as a cuboid, and each coordinate is verified to be within the patch bounds along
        all axes.

        :param coord: A numpy array of shape (N, 3), where N is the number of points,
            and each row represents the x, y, z coordinates of a point.
        :param patch_center: A numpy array of shape (3,), representing the x, y, z
            coordinates of the patch center.
        :return: A boolean array of shape (N,), where each element signifies whether
            the corresponding point in `coord` is located within the defined patch limits.
        :rtype: np.ndarray
        """
        patch_size_x = self.INIT_PATCH_SIZE[0]
        patch_size_y = self.INIT_PATCH_SIZE[1]
        patch_size_z = self.INIT_PATCH_SIZE[2]

        # Bounding box for patch center
        patch_x, patch_y, patch_z = patch_center
        patch_min_x, patch_max_x = patch_x - patch_size_x, patch_x + patch_size_x
        patch_min_y, patch_max_y = patch_y - patch_size_y, patch_y + patch_size_y
        patch_min_z, patch_max_z = patch_z - patch_size_z, patch_z + patch_size_z

        coord_idx = (
            (coord[:, 0] <= patch_max_x)
            & (coord[:, 0] >= patch_min_x)
            & (coord[:, 1] <= patch_max_y)
            & (coord[:, 1] >= patch_min_y)
            & (coord[:, 2] <= patch_max_z)
            & (coord[:, 2] >= patch_min_z)
        )

        return coord_idx

    def optimal_patches(self, coord: np.ndarray, random=False) -> List[bool]:
        """
        The `optimal_patches` function calculates and returns a list of boolean
        values representing whether specific patches within a given coordinate
        space contain points that meet certain criteria. It operates in two
        modes: random patch selection or systematic grid-based patch calculation.

        This function first calculates a bounding box for the input coordinates.
        It determines voxel sizes dynamically based on the bounding box dimensions,
        applies strategies for patch creation, and evaluates patches for spatial
        distribution of points. The two modes allow either randomly selected patches
        or systematic patches using voxel downscaling until a threshold is satisfied.

        :param coord: The spatial coordinate input as a numpy array.
        :type coord: np.ndarray

        :param random: Determines if the patch selection should follow
            a random process or systematic method. Default is False.
        :type random: bool, optional

        :return: A list of boolean values indicating which patches meet
            the specific criteria.
        :rtype: List[bool]
        """
        bbox = self.boundary_box(coord)
        if self.voxel is not None:
            voxel = self.voxel + self.drop_rate
        else:
            voxel = abs(bbox[0, 0] - bbox[1, 0]) // 10 + self.drop_rate
            if voxel == 0:
                voxel = abs(bbox[0, 0] - bbox[1, 0]) + self.drop_rate

        all_patch = []

        """ Find points index in patches """
        if random:
            if self.voxel is not None:
                voxel = self.voxel + self.drop_rate
            else:
                voxel = abs(bbox[0, 0] - bbox[1, 0]) // 10
                if voxel == 0:
                    voxel = abs(bbox[0, 0] - bbox[1, 0]) + self.drop_rate

            patch_grid = self.center_patch(bbox=bbox, voxel_size=voxel)

            x_pos = np.sort(np.unique(patch_grid[:, 0]))
            if len(x_pos) > 1:
                x_pos = (x_pos[1] - x_pos[0]) / 2
            else:
                x_pos = bbox[[1, 0]] - x_pos[0]

            y_pos = np.sort(np.unique(patch_grid[:, 1]))
            if len(y_pos) > 1:
                y_pos = (y_pos[1] - y_pos[0]) / 2
            else:
                y_pos = bbox[1, 1] - y_pos[0]

            z_pos = np.sort(np.unique(patch_grid[:, 2]))
            if len(z_pos) > 1:
                z_pos = (z_pos[1] - z_pos[0]) / 2
            else:
                z_pos = bbox[1, 2] - z_pos[0]

            self.INIT_PATCH_SIZE = [
                x_pos + (x_pos * self.STRIDE),
                y_pos + (y_pos * self.STRIDE),
                z_pos + (z_pos * self.STRIDE),
            ]

            """Find points in each patch"""
            for patch in patch_grid:
                point_idx = self.points_in_patch(coord=coord, patch_center=patch)
                all_patch.append(point_idx)
            all_patch_df = [patch for patch in all_patch if np.sum(patch) > 0]
            all_patch_bool = [
                True if np.sum(patch) > 0 else False for patch in all_patch
            ]

            """Pick random patch"""
            df_all = []
            for _ in range(10):
                # Initially random pick
                random_ = np.argwhere(all_patch_bool).flatten()[
                    np.random.choice(len(all_patch_df))
                ]
                all_patch = [
                    self.points_in_patch(coord=coord, patch_center=patch_grid[random_])
                ]

                # Check if picked voxel have more than self.mesh points
                # and less than downsample threshold
                pc_size = np.sum(all_patch[0])

                while (
                    pc_size < self.DOWNSAMPLING_TH * 0.5
                    or pc_size > self.DOWNSAMPLING_TH
                ):
                    if pc_size > self.DOWNSAMPLING_TH:
                        while pc_size > self.DOWNSAMPLING_TH:
                            self.INIT_PATCH_SIZE = [
                                self.INIT_PATCH_SIZE[0] * 0.99,
                                self.INIT_PATCH_SIZE[1] * 0.99,
                                self.INIT_PATCH_SIZE[2] * 0.99,
                            ]
                            all_patch = [
                                self.points_in_patch(
                                    coord=coord, patch_center=patch_grid[random_]
                                )
                            ]
                            pc_size = np.sum(all_patch[0])
                    elif pc_size < self.DOWNSAMPLING_TH * 0.5:
                        while pc_size < self.DOWNSAMPLING_TH * 0.8:
                            self.INIT_PATCH_SIZE = [
                                self.INIT_PATCH_SIZE[0] * 1.01,
                                self.INIT_PATCH_SIZE[1] * 1.01,
                                self.INIT_PATCH_SIZE[2] * 1.01,
                            ]
                            all_patch = [
                                self.points_in_patch(
                                    coord=coord, patch_center=patch_grid[random_]
                                )
                            ]
                            pc_size = np.sum(all_patch[0])

                    if (
                        pc_size > self.DOWNSAMPLING_TH
                        or pc_size < self.DOWNSAMPLING_TH * 0.5
                    ):
                        random_ = np.argwhere(all_patch_bool).flatten()[
                            np.random.choice(len(all_patch_df))
                        ]
                        all_patch = [
                            self.points_in_patch(
                                coord=coord, patch_center=patch_grid[random_]
                            )
                        ]
                        pc_size = np.sum(all_patch[0])
                df_all.append(all_patch[0])
            all_patch = df_all
        else:
            th = 1
            while th != 0:
                all_patch = []

                """Initialize search with new voxel size"""
                voxel = round(voxel - self.drop_rate, 1)
                patch_grid = self.center_patch(bbox=bbox, voxel_size=voxel)
                if len(patch_grid) < 2:
                    continue

                x_pos = np.sort(np.unique(patch_grid[:, 0]))
                if len(x_pos) > 1:
                    x_pos = (x_pos[1] - x_pos[0]) / 2
                else:
                    x_pos = bbox[[1, 0]] - x_pos[0]

                y_pos = np.sort(np.unique(patch_grid[:, 1]))
                if len(y_pos) > 1:
                    y_pos = (y_pos[1] - y_pos[0]) / 2
                else:
                    y_pos = bbox[1, 1] - y_pos[0]

                z_pos = np.sort(np.unique(patch_grid[:, 2]))
                if len(z_pos) > 1:
                    z_pos = (z_pos[1] - z_pos[0]) / 2
                else:
                    z_pos = bbox[1, 2] - z_pos[0]

                self.INIT_PATCH_SIZE = [
                    x_pos + (x_pos * self.STRIDE),
                    y_pos + (y_pos * self.STRIDE),
                    z_pos + (z_pos * self.STRIDE),
                ]

                """Find points in each patch"""
                for patch in patch_grid:
                    point_idx = self.points_in_patch(coord=coord, patch_center=patch)
                    all_patch.append(point_idx)

                all_patch = [patch for patch in all_patch if np.sum(patch) > 0]
                th = sum([True for p in all_patch if np.sum(p) > self.DOWNSAMPLING_TH])

            """ Combine smaller patches with threshold limit """
            new_patch = []
            while len(all_patch) > 0:
                df = all_patch[0]

                if np.sum(df) >= self.DOWNSAMPLING_TH:
                    new_patch.append(df)
                    all_patch.pop(0)
                else:
                    while np.sum(df) <= self.DOWNSAMPLING_TH:
                        if len(all_patch) == 1:
                            break
                        if np.sum(df) + np.sum(all_patch[1]) > self.DOWNSAMPLING_TH:
                            break
                        df += all_patch[1]
                        all_patch.pop(1)
                    new_patch.append(df)
                    all_patch.pop(0)
            all_patch = new_patch

        return all_patch

    @staticmethod
    def normalize_idx(coord_with_idx: np.ndarray) -> np.ndarray:
        """
        Normalizes the index values of coordinates in the given array. The method updates the array
        such that indices in the first column are replaced with normalized indices, starting from zero.
        The normalization process maps unique indices to a consecutive range of indices.

        :param coord_with_idx: A numpy array where the first column contains coordinate indices to be
            normalized.
        :type coord_with_idx: np.ndarray
        :return: The input array with normalized indices in its first column.
        :rtype: np.ndarray
        """
        unique_idx, inverse_idx = np.unique(coord_with_idx[:, 0], return_inverse=True)
        # Hot-fix for numpy 2.0.0
        if inverse_idx.ndim == 2:
            inverse_index = inverse_idx[:, 0]

        norm_idx = np.arange(len(unique_idx))

        for _, id_ in enumerate(unique_idx):
            mask = coord_with_idx[:, 0] == id_
            coord_with_idx[:, 0][mask] = norm_idx[inverse_idx[mask]]
        return coord_with_idx

    def output_format(self, data: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Converts the given data into the desired output format. If the `TORCH_OUTPUT`
        flag is set, the input NumPy array is converted to a PyTorch tensor of type
        `torch.float32`.

        :param data: Input data as a NumPy array.
        :type data: np.ndarray
        :return: The input data converted to the desired format. Returns a PyTorch
            tensor if `TORCH_OUTPUT` is enabled, otherwise returns the original
            NumPy array.
        :rtype: Union[np.ndarray, torch.Tensor]
        """
        if self.TORCH_OUTPUT:
            data = torch.from_numpy(data).type(torch.float32)
        return data

    def patched_dataset(
        self,
        coord: np.ndarray,
        label_cls=None,
        rgb=None,
        mesh=6,
        random=False,
        voxel_size=None,
    ) -> Union[Tuple[List, List, List, List, List], Tuple[List, List, List, List]]:
        """
        Generates patches of spatial data and optional associated properties (class labels,
        graph structures, and RGB values) for deep learning tasks. This function processes
        input spatial coordinates to divide them into smaller, manageable patches, and
        conditionally builds graph structures or assigns additional information like class
        labels and RGB values for each patch. This method supports either a 2D-to-3D
        transformation or segmented and raw data processing, depending on the provided input
        configurations.

        :param coord: An array of spatial coordinates, potentially in 2D, 3D, or segmented
            4D form.
        :param label_cls: Optional array of class labels for spatial coordinates.
        :param rgb: Optional array of RGB values for spatial coordinates.
        :param mesh: An integer indicating the scale or structure of graph connectivity
            (e.g., K in a K-Nearest Neighbors graph).
        :param random: Boolean indicating whether to randomly segment patches or determine
            them based on an optimization algorithm.
        :param voxel_size: Optional numeric value specifying the desired voxel size
            for downsampling spatial data.

        :return: A tuple of lists with various outputs segmented into patches:
            - Point cloud patches for input coordinates.
            - RGB data patches if available.
            - Graph patches if graph output is configured.
            - Indices of original data points corresponding to each patch.
            - Optional class label patches if `label_cls` is provided.
        """
        coord_patch = []
        graph_patch = []
        output_idx = []

        if voxel_size is not None:
            self.voxel = voxel_size
        else:
            self.voxel = None

        if self.GRAPH_OUTPUT:
            if coord.shape[1] not in [3, 4]:
                TardisError(
                    "113",
                    "tardis_em/dist_pytorch/datasets/patches.py",
                    "If graph True, coord must by of shape"
                    f"[Dim x X x Y x (Z)], but is: {coord.shape}",
                )
            segmented_coord = coord
            coord = coord[:, 1:]

            if mesh > 2:
                graph_builder = BuildGraph(K=mesh, mesh=True)
            else:
                graph_builder = BuildGraph(K=mesh, mesh=False)
        else:
            graph_builder = None
            if coord.shape[1] not in [2, 3]:
                TardisError(
                    "113",
                    "tardis_em/dist_pytorch/datasets/patches.py",
                    "If graph False, coord must by of shape"
                    f"[X x Y x (Z)], but is: {coord.shape}",
                )
            segmented_coord = None
            if coord.shape[1] == 2:
                """Transform 2D coord to 3D of shape [X, Y, Z]"""
                coord = np.vstack(
                    (
                        coord[:, 0],
                        coord[:, 1],
                        np.zeros((coord.shape[0],)),
                    )
                ).T
            else:
                coord = coord

        if coord.shape[0] <= self.DOWNSAMPLING_TH:
            coord_ds = [True for _ in list(range(0, coord.shape[0], 1))]

            """ Build point cloud for each patch """
            coord_patch.append(self.output_format(coord[coord_ds, :]))

            """ Optionally - Build graph for each patch """
            if self.GRAPH_OUTPUT:
                coord_label = segmented_coord[coord_ds, :]
                coord_label = self.normalize_idx(coord_label)

                graph_patch.append(self.output_format(graph_builder(coord=coord_label)))

            """ Build output index for each patch """
            output_idx.append(np.where(coord_ds)[0])

            """ Build class label index for each patch """
            if label_cls is not None:
                cls_patch = [label_cls]
            else:
                cls_patch = [self.output_format(np.zeros((1, 1)))]

            """ Build rgb node label index for each patch """
            if rgb is not None:
                rgb_patch = [rgb]
            else:
                rgb_patch = [self.output_format(np.zeros((1, 1)))]
        else:  # Build patches for PC with max num. of point per patch
            cls_patch = []
            rgb_patch = []

            """ Find points index in patches """
            all_patch = self.optimal_patches(coord=coord, random=random)

            """Build embedded feature per patch"""
            for i in all_patch:
                """Find points and optional images for each patch"""
                df_patch_keep = i

                df_patch = coord[df_patch_keep, :]
                output_df = np.where(df_patch_keep)[0]
                coord_ds = [True for _ in list(range(0, df_patch.shape[0], 1))]

                """ Build point cloud for each patch """
                coord_patch.append(self.output_format(df_patch[coord_ds, :]))

                """ Optionally - Build graph for each patch """
                if self.GRAPH_OUTPUT:
                    segment_patch = segmented_coord[df_patch_keep, :]
                    segment_patch = self.normalize_idx(segment_patch[coord_ds, :])

                    graph_patch.append(
                        self.output_format(graph_builder(coord=segment_patch))
                    )

                """ Build output index for each patch """
                output_idx.append(output_df[coord_ds])

                """ Build class label index for each patch """
                if label_cls is not None:
                    cls_df = label_cls[df_patch_keep]
                    cls_new = np.zeros((cls_df.shape[0], 200))
                else:
                    cls_df = [0]
                    cls_new = np.zeros((1, 200))

                for id_, j in enumerate(cls_df):
                    df = np.zeros((1, 200))
                    df[0, int(j)] = 1
                    cls_new[id_, :] = df
                cls_patch.append(self.output_format(cls_new))

                """ Build rbg node label index for each patch """
                if rgb is not None:
                    rgb_df = rgb[df_patch_keep]
                else:
                    rgb_df = np.zeros((1, 1))

                rgb_patch.append(self.output_format(rgb_df))

        if self.GRAPH_OUTPUT:
            return coord_patch, rgb_patch, graph_patch, output_idx, cls_patch
        else:
            return coord_patch, rgb_patch, output_idx, cls_patch
