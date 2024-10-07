#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Optional, Union

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from tardis_em.analysis.geometry_metrics import (
    angle_between_vectors,
    total_length,
    tortuosity,
)
from tardis_em.dist_pytorch.utils.utils import pc_median_dist


def resample_filament(points, spacing_size):
    """
    Resamples points for each filament so they have the given spacing size.

    Parameters:
    points (np.array): Array of shape [N, 4] where each column is [ID, X, Y, Z].
    spacing_size (float): Desired spacing size between points.

    Returns:
    np.array: Resampled array with the same structure.
    """
    unique_ids = np.unique(points[:, 0])
    resampled_points = []

    for filament_id in unique_ids:
        filament_points = points[points[:, 0] == filament_id][:, 1:]
        if filament_points.shape[0] < 2:
            resampled_points.append(points[points[:, 0] == filament_id])
            continue

        cumulative_distances = np.cumsum(
            np.sqrt(np.sum(np.diff(filament_points, axis=0) ** 2, axis=1))
        )
        cumulative_distances = np.insert(cumulative_distances, 0, 0)

        num_new_points = int(cumulative_distances[-1] // spacing_size)
        new_points = [filament_points[0]]

        for i in range(1, num_new_points + 1):
            target_distance = i * spacing_size
            idx = np.searchsorted(cumulative_distances, target_distance)
            if idx >= len(cumulative_distances):
                continue

            if cumulative_distances[idx] == target_distance:
                new_points.append(filament_points[idx])
            else:
                ratio = (target_distance - cumulative_distances[idx - 1]) / (
                    cumulative_distances[idx] - cumulative_distances[idx - 1]
                )
                new_point = filament_points[idx - 1] + ratio * (
                    filament_points[idx] - filament_points[idx - 1]
                )
                new_points.append(new_point)

        new_points = np.array(new_points)
        filament_ids = np.full((new_points.shape[0], 1), filament_id)
        resampled_points.append(np.hstack((filament_ids, new_points)))

    return np.vstack(resampled_points)


def sort_segments(coord: np.ndarray) -> np.ndarray:
    df = np.unique(coord[:, 0])

    new_coord = []
    for i in df:
        c = coord[np.isin(coord[:, 0], [i]), 1:]
        c = sort_segment(c)

        id_ = np.repeat(i, len(c))
        c = np.array((id_, c[:, 0], c[:, 1], c[:, 2])).T

        new_coord.append(c)

    return np.concatenate(new_coord)


def sort_segment(coord: np.ndarray) -> np.ndarray:
    """
    Sorting of the point cloud based on number of connections followed by
    searching of the closest point with cdist.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        np.ndarray: Array of point in line order.
    """
    if len(coord) < 2:
        return coord

    new_c = []
    for i in range(len(coord) - 1):
        if i == 0:
            id = np.where(
                [sum(i) for i in cdist(coord, coord)]
                == max([sum(i) for i in cdist(coord, coord)])
            )[0]

            new_c.append(coord[id[0]])
            coord = np.delete(coord, id[0], 0)

        kd = KDTree(coord)
        points = kd.query(np.expand_dims(new_c[len(new_c) - 1], 0))[1][0][0]

        new_c.append(coord[points])
        coord = np.delete(coord, points, 0)

    if len(new_c) > 0:
        return np.stack(new_c)
    else:
        return coord


def reorder_segments_id(
    coord: np.ndarray,
    order_range: Optional[list] = None,
    order_list: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Reorder list of segments to remove missing IDs

    E.g. Change IDs from [1, 2, 3, 5, 6, 8] to [1, 2, 3, 4, 5, 6]

    Args:
        coord: Array of points in 3D or 3D with their segment ID
        order_range: Costume id range for reordering
        order_list: List of reorder IDs to match to coord.

    Returns:
        np.ndarray: Array of points with reordered IDs values
    """
    df = np.unique(coord[:, 0])

    if order_range is None:
        df_range = np.asarray(range(0, len(df)), dtype=df.dtype)
    else:
        df_range = np.asarray(range(order_range[0], order_range[1]), dtype=df.dtype)

    if order_list is None:
        for id, i in enumerate(coord[:, 0]):
            coord[id, 0] = df_range[np.where(df == i)[0][0]]
    else:
        ordered_coord = []
        for i, new_id in zip(df, order_list):
            line = coord[np.where(coord[:, 0] == i)[0], :]
            line[:, 0] = new_id
            ordered_coord.append(line)
        coord = np.concatenate(ordered_coord)
        coord = reorder_segments_id(coord)

    return coord


def smooth_spline(points: np.ndarray, s=0.5):
    """
    Spline smoothing given an 's' smoothness factor.

    Args:
        points (np.ndarray): Point array [(ID), X, Y, Z] with optional ID and Z
        dimension.
        s (float): Smoothness factor.

    Returns:
        Returns: Smooth spline
    """
    if points.shape[1] == 4:  # [ID, X, Y, Z]
        id_ = int(points[0, 0])
        points = points[:, 1:]
        norm_pc = pc_median_dist(points)
        points = points / norm_pc

        t_before = tortuosity(points)
        try:
            tck, u = splprep(points.T, s=s)
            spline = np.stack(splev(u, tck)).T
        except ValueError:
            spline = points

        spline = spline * norm_pc
        t_after = tortuosity(spline)
        ids = np.zeros((len(spline), 1))
        ids[:, 0] = id_

        # Sanity check if spline smoothing failed
        if t_after > t_before:
            return np.hstack((ids, points * norm_pc))
        return np.hstack((ids, spline))
    else:  # [X, Y, Z]
        tck, u = splprep(points.T)

        return np.stack(splev(u, tck)).T


def sort_by_length(coord):
    """
    Sort all splines by their length.

    Args:
        coord: Array of coordinates.

    Returns:
        np.ndarray: sorted and reorder splines.
    """
    length_list = []
    for i in np.unique(coord[:, 0]):
        length_list.append(total_length(coord[np.where(coord[:, 0] == i)[0], 1:]))

    sorted_id = np.argsort(length_list)

    sorted_list = [coord[np.where(coord[:, 0] == i)[0], 1:] for i in sorted_id]
    sorted_list = [
        np.hstack((np.repeat(i, len(sorted_list[i])).reshape(-1, 1), sorted_list[i]))
        for i in range(len(sorted_list))
    ]

    return np.concatenate(sorted_list)


def cut_150_degree(segments_array: np.ndarray):
    """
    Cut segments based on angles between adjacent vectors.

    Given an array of line segments, this function calculates angles between
    adjacent vectors in each line. If the angle is less than or equal to 150
    degrees, the segment is cut into two new segments.

    Args:
    segments_array(np. ndarray): Array of line segments where the first column
    indicates the segment id and the remaining columns represent
    the coordinates of points.

    Args:
        segments_array:

    Returns:
        Tuple[bool, np.ndarray]: Indicates whether any segment was cut,
            and New array of cut segments.
    """

    cut_segments = []
    loop_ = False

    # Loop through unique segment IDs
    for i in np.unique(segments_array[:, 0]):
        pc_ = segments_array[np.where(segments_array[:, 0] == i)[0], 1:]

        angles_ = [180]

        # Calculate angles for each line segment
        for j in range(len(pc_) - 2):
            angles_.append(
                angle_between_vectors(
                    np.array(pc_[j]) - np.array(pc_[j + 1]),
                    np.array(pc_[j + 2]) - np.array(pc_[j + 1]),
                )
            )
        angles_.append(180)

        # Check if any angle is less than or equal to 150 degrees
        if len([id_ for id_, k in enumerate(angles_) if k <= 150]) > 0:
            loop_ = True

            # Find the minimum angle and cut the segment
            min_angle_idx = np.where(angles_ == np.min(angles_))[0][0]
            cut_segments.append(pc_[: min_angle_idx + 1, :])
            cut_segments.append(pc_[min_angle_idx + 1 :, :])
        else:
            cut_segments.append(pc_)

    # Filter out single-point segments
    cut_segments = [c for c in cut_segments if len(c) > 1]

    # Create the output array
    return loop_, np.vstack(
        [
            np.hstack((np.repeat(id_, len(c)).reshape(-1, 1), c))
            for id_, c in enumerate(cut_segments)
        ]
    )
