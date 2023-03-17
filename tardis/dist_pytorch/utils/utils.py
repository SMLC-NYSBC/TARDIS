#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import KDTree

from tardis.utils.errors import TardisError


def pc_median_dist(pc: np.ndarray,
                   avg_over=False,
                   box_size=0.15) -> float:
    """
    !DEPRECIATED! - Remove in RC3

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
    if avg_over:
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()

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

        voxel = point_in_bb(pc,
                            min_x=x - offset_x,
                            max_x=x + offset_x,
                            min_y=y - offset_y,
                            max_y=y + offset_y,
                            min_z=z - offset_z,
                            max_z=z + offset_z)
        pc = pc[voxel]

        # Calculate KNN dist
        tree = KDTree(pc, leaf_size=pc.shape[0])
    else:
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()
        tree = KDTree(pc, leaf_size=pc.shape[0])

    if pc.shape[0] < 3:
        return 1.0

    knn_df = []
    for id, i in enumerate(pc):
        knn, _ = tree.query(pc[id].reshape(1, -1), k=4)
        knn_df.append(knn[0][1])

    return float(np.mean(knn_df))


def point_in_bb(points: np.ndarray,
                min_x: int,
                max_x: int,
                min_y: int,
                max_y: int,
                min_z: Optional[np.float32] = None,
                max_z: Optional[np.float32] = None) -> np.ndarray:
    """
    !DEPRECIATED! - Remove in RC3

    Compute a bounding_box filter on the given points

    Args:
        points (np.ndarray): (n,3) array.
        min_i, max_i (int): The bounding box limits for each coordinate.
            If some limits are missing, the default values are -infinite for
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


class RandomDownSampling:
    """
    Wrapper for random sampling of the point cloud
    """

    def __init__(self,
                 threshold):
        self.threshold = threshold

    @staticmethod
    def pc_rand_down_sample(coord: np.ndarray,
                            threshold,
                            rgb: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray,
                                                                             np.ndarray],
                                                                       np.ndarray]:
        """
        Random picked point to down sample point cloud

        Args:
            coord: Array of point to downs-sample
            rgb: Extra node feature like e.g. RGB values do sample with coord.
            threshold: Lambda function to calculate down-sampling rate or fixed float ratio.

        Returns:
            np.ndarray: Down-sample array of points.
        """
        if isinstance(threshold, int) or isinstance(threshold, float):
            rand_keep = int(len(coord) * threshold)
        else:
            rand_keep = threshold(coord)

        if rand_keep != len(coord):
            rand_keep = random.sample(range(len(coord)), rand_keep)
        else:
            rand_keep = list(range(len(coord)))

        if rgb is None:
            return coord[rand_keep][:, :3]
        else:
            return coord[rand_keep][:, :3], rgb[rand_keep][:, :3]

    def __call__(self,
                 coord: Optional[np.ndarray] = list,
                 rgb: Optional[Union[np.ndarray, list]] = None) -> Union[Tuple[np.ndarray,
                                                                               np.ndarray],
                                                                         np.ndarray]:
        ds_pc = []
        ds_rgb = []

        if isinstance(coord, list):
            if rgb is not None and not isinstance(rgb, list):
                TardisError('130',
                            'tardis/dist_pytorch/utils/utils.py',
                            'List of coordinates require list of rbg but array was give!')

            id = 0
            for idx, i in enumerate(coord):
                if rgb is not None:
                    rgb_df = rgb[idx]
                    ds_coord, ds_node_f = self.pc_rand_down_sample(coord=i,
                                                                   rgb=rgb_df,
                                                                   threshold=self.threshold)
                    ds_rgb.append(ds_node_f)
                else:
                    ds_coord = self.pc_rand_down_sample(coord=i, threshold=self.threshold)

                ds_pc.append(np.hstack((np.expand_dims(np.repeat(id, len(ds_coord)), 1),
                                        ds_coord)))
                id += 1
        elif coord.shape[1] == 3:
            if rgb is not None:
                return self.pc_rand_down_sample(coord=coord,
                                                rgb=rgb,
                                                threshold=self.threshold)
            else:
                return self.pc_rand_down_sample(coord=coord, threshold=self.threshold)
        else:
            labels = np.unique(coord[:, 0])
            for i in labels:
                peak_id = np.where(coord[:, 0] == i)[0]
                coord_df = coord[peak_id, 1:]

                if rgb is not None:
                    rgb_df = rgb[peak_id, :]
                    ds_coord, ds_node_f = self.pc_rand_down_sample(coord=coord_df,
                                                                   rgb=rgb_df,
                                                                   threshold=self.threshold)
                    ds_rgb.append(ds_node_f)
                else:
                    ds_coord = self.pc_rand_down_sample(coord=coord_df,
                                                        threshold=self.threshold)

                ds_coord = np.hstack((np.expand_dims(np.repeat(i, len(ds_coord)), 1),
                                      ds_coord))
                ds_pc.append(ds_coord)

        if rgb is not None:
            return np.concatenate(ds_pc), np.concatenate(ds_rgb)
        else:
            return np.concatenate(ds_pc)


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
        if key.endswith('type'):
            new_dict['dist_type'] = value
        if key.endswith('out'):
            new_dict['n_out'] = value
        if key.endswith('node_input'):
            new_dict['node_input'] = value
        if key.endswith('node_dim'):
            new_dict['node_dim'] = value
        if key.endswith('edge_dim'):
            new_dict['edge_dim'] = value
        if key.endswith('layers'):
            new_dict['num_layers'] = value
        if key.endswith('heads'):
            new_dict['num_heads'] = value
        if key.endswith('cls'):
            new_dict['num_cls'] = value
        if key.endswith('sigma'):
            if key.startswith('coord'):
                new_dict['coord_embed_sigma'] = value
            if key.startswith('rgb'):
                new_dict['rgb_embed_sigma'] = value
        if key.endswith('dropout') or key.startswith('dropout'):
            new_dict['dropout_rate'] = value
        if key.endswith('structure'):
            new_dict['structure'] = value

    if 'num_cls' not in new_dict:
        new_dict['num_cls'] = None

    if 'rgb_embed_sigma' not in new_dict:
        new_dict['rgb_embed_sigma'] = 1.0

    return new_dict
