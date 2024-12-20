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
    Computes the median nearest neighbor distance for a given point cloud.

    This function calculates the median nearest neighbor distance between points
    in a given 2D or 3D point cloud array. Optionally, it can restrict the computation
    to a subset of points that are within a bounding box region centered around the
    median position of the point cloud. The bounding box dimensions can be scaled
    based on a user-defined `box_size`.

    :param pc: A 2D or 3D point cloud array of shape (N, D), where N is the number
        of points and D is the spatial dimensionality (2 or 3).
    :type pc: np.ndarray
    :param avg_over: Flag to indicate whether to compute the distances over a
        subset of the point cloud within a bounding box. Defaults to False.
    :type avg_over: bool, optional
    :param box_size: Fraction of the bounding box size relative to the point cloud
        extents. Only applicable when `avg_over` is True. Defaults to 0.15.
    :type box_size: float, optional
    :return: The mean of the median nearest neighbor distances.
    :rtype: float
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
    Determines whether points in a given array fall within a specified bounding box.

    The function evaluates if points, provided as an array, lie within the
    boundaries defined by minimum and maximum values for x, y, and optionally z coordinates.
    It enables the filtering of points based on inclusion within a 2D or 3D bounding box.

    :param points: Array representing the coordinates of points, where each row is a point
        and columns correspond to x, y, and optionally z coordinates.
    :type points: numpy.ndarray
    :param min_x: Minimum x-coordinate boundary of the bounding box.
    :type min_x: int
    :param max_x: Maximum x-coordinate boundary of the bounding box.
    :type max_x: int
    :param min_y: Minimum y-coordinate boundary of the bounding box.
    :type min_y: int
    :param max_y: Maximum y-coordinate boundary of the bounding box.
    :type max_y: int
    :param min_z: Optional, minimum z-coordinate boundary of the bounding box.
    :type min_z: numpy.float32, optional
    :param max_z: Optional, maximum z-coordinate boundary of the bounding box.
    :type max_z: numpy.float32, optional
    :return: A boolean array where each element corresponds to the inclusion of a point in the bounding box.
    :return type: numpy.ndarray
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
    Provides functionality for downsampling point cloud data, allowing optional inclusion of RGB values and
    support for various data formats.

    The class is designed to aid in voxel downsampling operations on point cloud data. It supports two
    primary methods of operation: processing the entire point cloud at once or downsampling each index
    when the input is a list. Additional features include support for sampling point clouds with or
    without assigned class IDs (labels) and optional K-Nearest Neighbors (KNN) consideration.
    """

    def __init__(self, voxel=None, threshold=None, labels=True, KNN=False):
        """
        Initializes a parameterized object with configurable sample source, labeling preference,
        and K-Nearest Neighbors (KNN) usage.

        This class is designed to handle voxel data with a specified threshold for sampling.
        Optionally, labeling of class IDs and KNN can be enabled to tailor the behavior
        of the object according to user requirements.

        :param voxel: Sampling data to be used. If None, the threshold value is employed instead.
        :param threshold: Fallback sampling value when no voxel data is provided.
        :param labels: Specifies whether to assign class IDs during down sampling. Expects input
            in formats [ID x X x Y x (Z)] or [[N, 3], [N, 4]]. Defaults to True.
        :param KNN: Determines whether K-Nearest Neighbors (KNN) is utilized. Defaults to False.
        """
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
        """
        Perform down-sampling for the given coordinates and optionally their corresponding
        RGB values. If RGB values are provided, the method returns both the downsampled
        coordinates and their RGB values. Otherwise, only the downsampled coordinates
        are returned.

        :param coord: Coordinates to down-sample.
        :type coord: np.ndarray
        :param sampling: Sampling parameters or logic to apply.
        :param rgb: Optional RGB values corresponding to the input coordinates.
        :type rgb: np.ndarray, optional
        :return: Down-sampled coordinates, and optionally RGB values if provided.
        :rtype: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
        """
        if rgb is not None:
            return coord, rgb
        return coord

    def __call__(
        self,
        coord: Optional[np.ndarray] = list,
        rgb: Optional[Union[np.ndarray, list]] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Processes and downsamples the provided point cloud and associated RGB data. This
        method accepts either a single point cloud or a list of point clouds and can also
        accommodate associated RGB data if provided. The intent of the method is to reduce
        the size of the input data through down-sampling, while optionally attaching
        unique identifiers or RGB information to the point cloud.

        If the input is a list of point clouds, each point cloud will be processed
        individually and then combined into a unified structure.

        :param coord:
            A point cloud coordinate array or a list of coordinate arrays. For individual
            arrays, expected dimensions are [N, 2], [N, 3], [N, 4] depending on the context. If a list
            is provided, all elements should adhere to the described dimensional rules.
        :param rgb:
            Optional RGB array or list of RGB arrays corresponding to the input point clouds.
            Must have a compatible structure with the provided coordinate arrays, or a list
            of such RGB arrays if a list of point clouds is passed.
        :return:
            A tuple containing the down-sampled point cloud and corresponding down-sampled RGB
            data if RGB data is provided. If RGB data is not included, only the down-sampled
            point cloud is returned. If the input is a list, the output will be a unified
            structure combining the processed instances.
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
    Provides functionality for down-sampling 3D point clouds by grouping points into voxels
    of a specified size and computing the centroid of each voxel. Optionally supports operations
    with RGB data, labels, and nearest neighbor search using KDTree.

    Useful for reducing the size of large point clouds for computational efficiency while
    preserving their spatial structure.
    """

    def __init__(self, **kwargs):
        super(VoxelDownSampling, self).__init__(**kwargs)

    def pc_down_sample(
        self, coord: np.ndarray, sampling: float, rgb: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Down-sample a point cloud using a voxel-based approach.

        This function performs down-sampling of a point cloud by dividing the point cloud
        space into uniformly spaced 3D grid cells (voxels) and representing each voxel by
        its centroid. Optionally, it associates additional information such as color or
        label attributes to the down-sampled points.

        :param coord: The 3D coordinates of the input point cloud. It is expected to
                      be a NumPy array with shape (N, 3) where N is the number of points.
        :param sampling: The size of the voxel grid's cube edge. A smaller value results
                         in a finer resolution, retaining more detail.
        :param rgb: Optional. If provided, it should be a NumPy array with shape (N, 3)
                    representing the RGB colors of the input point cloud. The colors
                    of down-sampled points will be computed accordingly.
        :return:
            - If `rgb` is not None, returns a tuple:
                - First element is a NumPy array with shape (M, 3) or (M, 4), where M is
                  the number of down-sampled points. Contains the 3D coordinates of the
                  down-sampled points. If `self.labels` is true, an additional label
                  column is included.
                - Second element is a NumPy array with shape (M, 3), representing the RGB
                  colors associated with the down-sampled points.
            - If `rgb` is None, returns only the down-sampled 3D coordinates as a NumPy
              array with shape (M, 3) or (M, 4) depending on `self.labels`.
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
    A subclass of DownSampling that implements random down-sampling of a point cloud.

    This class leverages random selection to down-sample point cloud data. It is suitable for
    reducing the size of datasets while maintaining a random subset of points. The class retains
    compatibility with additional node features such as RGB values during the down-sampling process.
    """

    def __init__(self, **kwargs):
        super(RandomDownSampling, self).__init__(**kwargs)

    @staticmethod
    def pc_down_sample(
        coord: np.ndarray, sampling, rgb: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Downsamples a point cloud represented by coordinates and optional RGB values based
        on a provided sampling strategy. The function either retains a fraction of the points
        or selects points based on a callable sampling strategy.

        :param coord: The input point cloud coordinates. Each row represents a point.
                       The shape of the numpy array is (N, D), where N is the number
                       of points and D is the dimensionality of each point.
        :type coord: np.ndarray
        :param sampling: The sampling strategy. Can be an integer, float, or callable.
                         If an integer or float, specifies the fraction of points to keep.
                         If a callable, it determines which points to keep dynamically.
        :type sampling: Union[int, float, Callable]
        :param rgb: Optional array representing the RGB colors for each point in the
                    point cloud. It has the same number of rows as `coord`, where each
                    row corresponds to a point's color.
        :type rgb: Optional[np.ndarray]
        :return: A subset of the point cloud coordinates. If `rgb` is provided,
                 the function also returns a subset of the RGB values corresponding
                 to the retained points.
        :rtype: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
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
    Processes the provided `model_dict` by mapping specific key patterns to predefined normalized keys
    and extracting their corresponding values into a new dictionary. This function simplifies the
    original dictionary representation into a standardized configuration dictionary that is ready
    for further use. Default values are also assigned to certain keys if they are not present in the
    provided dictionary.

    :param model_dict: A dictionary containing model configuration parameters with keys following
        specific naming conventions.
    :type model_dict: dict
    :return: A standardized dictionary that includes key-value pairs extracted based on the specified
        naming conventions, along with default values for missing keys such as "num_cls",
        "rgb_embed_sigma", and "num_knn".
    :rtype: dict
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
