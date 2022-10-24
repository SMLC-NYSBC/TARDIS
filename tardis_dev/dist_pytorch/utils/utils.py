from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import KDTree


def pc_median_dist(pc: np.ndarray,
                   avg_over=False,
                   box_size=0.15) -> float:
    """
    Calculate the median distance between KNN points in the point cloud.

    Args:
        pc (np.ndarray): 2D/3D array of the point clouds.
        avg_over (bool): If True, calculate the median position of all points
            and calculate average k-NN for a selected set of points in that area
            (speed up for big point clouds).
        box_size: Boundary box size for 'avg_over'.
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

        if box_dim == 3:
            min_z = np.min(pc[:, 2])
            max_z = np.max(pc[:, 2])
            offset_z = (max_z - min_z) * box_size
        else:
            min_z, max_z = 0, 0
            offset_z = 0

        x = np.median(pc[:, 0])
        y = np.median(pc[:, 1])

        if box_dim == 3:
            z = np.median(pc[:, 2])
        else:
            z = 0

        voxel = point_in_bb(pc,
                            min_x=x - offset_x, max_x=x + offset_x,
                            min_y=y - offset_y, max_y=y + offset_y,
                            min_z=z - offset_z, max_z=z + offset_z)
        voxel = pc[voxel]

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

    return np.mean(knn_df)


def point_in_bb(points: np.ndarray,
                min_x: np.inf, max_x: np.inf,
                min_y: np.inf, max_y: np.inf,
                min_z: Optional[np.float32] = None,
                max_z: Optional[np.float32] = None) -> np.ndarray:
    """
    Compute a bounding_box filter on the given points

    Args:
        points (np.ndarray): (n,3) array.
        min_i, max_i (np.inf or int): The bounding box limits for each coordinate.
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
            bound_z = np.logical_and(
                points[:, 2] > min_z, points[:, 2] < max_z)
        else:
            bound_z = np.asarray([True for _ in points[:, 2]])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    else:
        bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter
