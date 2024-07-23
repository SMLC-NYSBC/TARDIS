#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors, KDTree

from tardis_em.utils.errors import TardisError


def pc_median_dist(pc: np.ndarray, avg_over=False, box_size=0.15) -> float:
    """
    Calculate the median distance between KNN points in the point cloud.

    Args:
        pc (np.ndarray): 2D/3D array of the point clouds.
        avg_over (bool): If True, calculate the median position of all points
            and calculate average k-NN for a selected set of points in that area
            (speed up for big point clouds).
        box_size (float): Boundary box size for 'avg_over'.

    Returns:
        float: Median distance between points in given point cloud.
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().detach().numpy()

    if avg_over:
        # Build BB and offset by 10% from the border
        box_dim = pc.shape[1]

        if box_dim in [2, 3]:
            min_x = np.min(pc[:, 0])
            max_x = np.max(pc[:, 0])
            offset_x = (max_x - min_x) * box_size

            min_y = np.min(pc[:, 1])
            max_y = np.max(pc[:, 1])
            offset_y = (max_y - min_y) * box_size
        else:
            offset_x = 0
            offset_y = 0

        if box_dim == 3:
            min_z = np.min(pc[:, 2])
            max_z = np.max(pc[:, 2])
            offset_z = (max_z - min_z) * box_size
        else:
            offset_z = 0

        x = np.median(pc[:, 0])
        y = np.median(pc[:, 1])

        if box_dim == 3:
            z = np.median(pc[:, 2])
        else:
            z = 0

        voxel = point_in_bb(
            pc,
            min_x=x - offset_x,
            max_x=x + offset_x,
            min_y=y - offset_y,
            max_y=y + offset_y,
            min_z=z - offset_z,
            max_z=z + offset_z,
        )
        pc = pc[voxel]

    # build a NearestNeighbors object for efficient nearest neighbor search
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(pc)

    if pc.shape[0] < 3:
        return 1.0

    distances, _ = nn.kneighbors(pc)
    distances = distances[:, 1]

    return float(np.mean(distances))


def point_in_bb(
    points: np.ndarray,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    min_z: Optional[np.float32] = None,
    max_z: Optional[np.float32] = None,
) -> np.ndarray:
    """
    Compute a bounding_box filter on the given points

    Args:
        points (np.ndarray): (n,3) array.
        min_i, max_i (int): The bounding box limits for each coordinate.
            If some limits are missing, the default values are - infinite for
            the min_i and infinite for the max_i.

    Returns:
        np.ndarray[bool]: The boolean mask indicates wherever a point should be
            kept or not.
    """
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    if points.shape[0] == 3:
        if min_z is not None or max_z is not None:
            bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
        else:
            bound_z = np.asarray([True for _ in points[:, 2]])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    else:
        bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter


class DownSampling:
    """
    Base down sampling wrapper
    """

    def __init__(self, voxel=None, threshold=None, labels=True, KNN=False):
        if voxel is None:
            self.sample = threshold
        else:
            self.sample = voxel

        # If true downs sample with class ids. expect [ID x X x Y x (Z)] [[N, 3] or [N, 4]]
        self.labels = labels
        self.KNN = KNN

    @staticmethod
    def pc_down_sample(
        coord: np.ndarray,
        sampling,
        rgb=None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if rgb is not None:
            return coord, rgb
        return coord

    def __call__(
        self,
        coord: Optional[np.ndarray] = list,
        rgb: Optional[Union[np.ndarray, list]] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Compute voxel down sampling for entire point cloud at once or if coord is a list
        compute voxel down sampling for each list index.
        """
        ds_pc = []
        ds_rgb = []

        # Assert correct data structure
        if self.labels:
            if coord.shape[1] not in [3, 4]:
                TardisError(
                    "130",
                    "tardis_em/dist_pytorch/utils/utils.py",
                    f"Expected coordinate array with IDs of shape [[N, 3] or [N, 4]], but {coord.shape} was given!",
                )
        else:
            if coord.shape[1] not in [2, 3]:
                TardisError(
                    "130",
                    "tardis_em/dist_pytorch/utils/utils.py",
                    f"Expected coordinate array without IDs of shape [[N, 2] or [N, 3]], but {coord.shape} was given!",
                )

        """Down-sample each instance from list and combine"""
        if isinstance(coord, list):
            # Assert if RGB are of the same structure as coord
            if rgb is not None and not isinstance(rgb, list):
                TardisError(
                    "130",
                    "tardis_em/dist_pytorch/utils/utils.py",
                    "List of coordinates require list of rbg but array was give!",
                )

            # Down sample
            id = 0
            for idx, i in enumerate(coord):
                coord_df = i
                if self.labels:
                    id = coord_df[0, 0]

                if rgb is not None:
                    rgb_df = rgb[idx]

                    coord_df, rgb_df = self.pc_down_sample(
                        coord=coord_df, rgb=rgb_df, sampling=self.sample
                    )
                    ds_rgb.append(rgb_df)
                else:
                    coord_df = self.pc_down_sample(coord=coord_df, sampling=self.sample)

                ds_pc.append(
                    np.hstack((np.repeat(id, len(coord_df)).reshape(-1, 1), coord_df))
                )
                if not self.labels:
                    id += 1
        else:
            """Down-sample entire point cloud at once"""
            if rgb is not None:
                return self.pc_down_sample(coord=coord, rgb=rgb, sampling=self.sample)
            else:
                return self.pc_down_sample(coord=coord, sampling=self.sample)


class VoxelDownSampling(DownSampling):
    """
    Wrapper for down sampling of the point cloud using voxel grid (Based on Open3d library)
    """

    def __init__(self, **kwargs):
        super(VoxelDownSampling, self).__init__(**kwargs)

    def pc_down_sample(
        self, coord: np.ndarray, sampling: float, rgb: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        This function takes a set of 3D points and a voxel size and returns the centroids
        of the voxels in which the points are located.

        Args:
            coord (np.ndarray): A numpy array of shape (N, 3) containing the 3D coordinates
            of the input points.
            rgb (np.ndarray): A numpy array of shape (N, 3) containing RGB values of each point.
            sampling (float): The size of each voxel in each dimension.

        Returns:
            voxel_centers (np.ndarray): A numpy array of shape (M, 3) containing the centroids
            of the voxels in which the points are located where M is the number
            of unique voxels.
        """
        if self.labels:
            coord_label = coord
            coord = coord[:, 1:]

        # Find the grid cell index for each point
        voxel_index = np.floor(coord / sampling).astype(np.int32)

        # Compute the unique set of voxel indices
        unique_voxel_index, inverse_index, voxel_counts = np.unique(
            voxel_index, axis=0, return_inverse=True, return_counts=True
        )

        # Hot-fix for numpy 2.0.0
        if inverse_index.ndim == 2:
            inverse_index = inverse_index[:, 0]

        # Compute the centroids of each voxel
        voxel_centers = np.zeros((len(unique_voxel_index), 3))
        np.add.at(voxel_centers, inverse_index, coord)
        voxel_centers /= voxel_counts[:, np.newaxis]

        # Retrieve ID value for down sampled point cloud
        if self.labels or rgb is not None or self.KNN:
            # Build a KDTree from the voxel_centers
            tree = KDTree(coord)

            # Query the KDTree to find the nearest voxel center for each coord point
            _, nearest_voxel_index = tree.query(voxel_centers)
            nearest_voxel_index = np.concatenate(nearest_voxel_index)

            if self.labels and not self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = np.hstack(
                    (coord_label[nearest_voxel_index, 0].reshape(-1, 1), voxel_centers)
                )
            elif self.labels and self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = np.hstack(
                    (
                        coord_label[nearest_voxel_index, 0].reshape(-1, 1),
                        coord[nearest_voxel_index, :],
                    )
                )
            elif not self.labels and self.KNN:
                # Compute the color of the nearest voxel center for each down-sampled point
                voxel_centers = coord[nearest_voxel_index, :]

        if rgb is not None:
            # Compute the color of the nearest voxel center for each down-sampled point
            return voxel_centers, rgb[nearest_voxel_index]
        return voxel_centers


class RandomDownSampling(DownSampling):
    """
    Wrapper for random sampling of the point cloud
    """

    def __init__(self, **kwargs):
        super(RandomDownSampling, self).__init__(**kwargs)

    @staticmethod
    def pc_down_sample(
        coord: np.ndarray, sampling, rgb: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Random picked point to down sample point cloud.
        Return correctly preserve ID class.

        Args:
            coord: Array of point to downs-sample
            rgb: Extra node feature like e.g. RGB values do sample with coord.
            sampling: Lambda function to calculate down-sampling rate or fixed float ratio.

        Returns:
            random_sample (np.ndarray): Down-sample array of points.
        """
        if isinstance(sampling, int) or isinstance(sampling, float):
            rand_keep = int(len(coord) * sampling)
        else:
            rand_keep = sampling(coord)

        if rand_keep != len(coord):
            rand_keep = random.sample(
                range(len(coord)), rand_keep
            )  # Randomly select coords
        else:
            rand_keep = list(range(len(coord)))  # Keep all

        if rgb is None:
            return coord[rand_keep]
        else:
            return coord[rand_keep], rgb[rand_keep]


def check_model_dict(model_dict: dict) -> dict:
    """
    Check and rebuild model structure dictionary to ensure back-compatibility.

    Args:
        model_dict (dict): Model structure dictionary.

    Returns:
        dict: Standardize model structure dictionary.
    """
    new_dict = {}

    for key, value in model_dict.items():
        if key.endswith("type"):
            new_dict["dist_type"] = value
        if key.endswith("out"):
            new_dict["n_out"] = value
        if key.endswith("node_input"):
            new_dict["node_input"] = value
        if key.endswith("node_dim"):
            new_dict["node_dim"] = value
        if key.endswith("edge_dim"):
            new_dict["edge_dim"] = value
        if key.endswith("layers"):
            new_dict["num_layers"] = value
        if key.endswith("heads"):
            new_dict["num_heads"] = value
        if key.endswith("cls"):
            new_dict["num_cls"] = value
        if key.endswith("sigma"):
            if key.startswith("coord"):
                new_dict["coord_embed_sigma"] = value
            if key.startswith("rgb"):
                new_dict["rgb_embed_sigma"] = value
        if key.endswith("dropout") or key.startswith("dropout"):
            new_dict["dropout_rate"] = value
        if key.endswith("structure"):
            new_dict["structure"] = value
        if key.endswith("knn"):
            new_dict["num_knn"] = value
    if "num_cls" not in new_dict:
        new_dict["num_cls"] = None

    if "rgb_embed_sigma" not in new_dict:
        new_dict["rgb_embed_sigma"] = 1.0

    if "num_knn" not in new_dict:
        new_dict["num_knn"] = None

    return new_dict
