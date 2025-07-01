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

from tardis_em.analysis.geometry_metrics import length_list
from tardis_em.analysis.geometry_metrics import (
    angle_between_vectors,
    total_length,
    tortuosity,
)
from tardis_em.dist_pytorch.utils.utils import pc_median_dist


def resample_filament(points, spacing_size) -> np.ndarray:
    """
    Resample a collection of 3D coordinates (X, Y, Z) associated with unique IDs to
    uniform spacing along their paths. The input points are grouped by their IDs, and
    each group is interpolated based on the specified spacing size.

    :param points: A numpy array of shape [N, 4], where each row contains [ID, X, Y, Z].
                   The 'ID' column is used to differentiate between different groups of
                   points, while the remaining columns represent the 3D coordinates.
    :param spacing_size: A float sfpecifying the uniform spacing distance to be applied
                         when resampling the 3D points within each group.

    :return: A numpy array of resampled points with shape [M, 4], where each row contains
             [ID, X, Y, Z]. The output maintains the ID for each group, along with its
             uniformly-spaced coordinates.
    :rtype: numpy.ndarray
    """
    # Verify input format
    if points.shape[1] != 4:
        raise ValueError(
            "Input `points` must have shape [N, 4] where columns represent [ID, X, Y, Z]."
        )

    # Initialize list for resampled points
    resampled_points = []

    # Get unique IDs
    unique_ids = np.unique(points[:, 0])

    if spacing_size == 'auto':
        length_max = np.max(length_list(points))
        spacing_size_ = int(0.01 * length_max)
    else:
        spacing_size_ = spacing_size

    # Loop over each unique ID
    for unique_id in unique_ids:
        # Extract points for the current ID
        id_points = points[points[:, 0] == unique_id][
            :, 1:
        ]  # Extract X, Y, Z coordinates

        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(id_points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Start from 0

        # Create a new range of distances for interpolation
        new_distances = np.arange(
            0, cumulative_distances[-1] + spacing_size_, spacing_size_
        )

        # Interpolate X, Y, Z coordinates along the new distance range
        new_x = np.interp(new_distances, cumulative_distances, id_points[:, 0])
        new_y = np.interp(new_distances, cumulative_distances, id_points[:, 1])
        new_z = np.interp(new_distances, cumulative_distances, id_points[:, 2])

        # Combine the new points with their ID
        new_points = np.column_stack(
            (np.full_like(new_x, unique_id), new_x, new_y, new_z)
        )

        # Add the new points to the result list
        resampled_points.append(new_points)

    # Combine all resampled points into a single array
    resampled_points = np.vstack(resampled_points)

    return resampled_points


def sort_segments(coord: np.ndarray) -> np.ndarray:
    """
    Sorts 3D segments by their coordinates. The function first identifies unique
    segment identifiers in the first column of the array. It then collects and
    sorts the associated segments for each unique identifier, constructing a
    final 2D array combining the sorted results.

    :param coord: Input 2D array where each row represents a 3D segment. The first
        column contains unique segment identifiers, and the remaining columns
        contain the segment coordinates.

    :return: A sorted 2D array of the same shape as the input, where the rows
        are sorted segment-wise based on their coordinates.
    """
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
    Sorts a set of coordinates in a specific sequence by iteratively selecting the farthest point
    and proceeding to find the nearest neighbors in subsequent steps.

    This function is designed to work with a two-dimensional array of coordinates. It uses a
    combination of distance calculations (using `cdist`) and nearest-neighbor search (using
    `scipy.spatial.KDTree`) to determine the order in which coordinates are rearranged. The
    process starts by identifying the farthest pair of points in the set and continues from
    the first point in the sequence, progressively adding the nearest neighbor to the sequence.

    :param coord: A numpy array of two-dimensional coordinates. Shape is expected as
                  (n, 2), where n is the number of coordinates provided.

    :return: A numpy array of sorted coordinates in the order as per the described procedure.
    :rtype: np.ndarray
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
    Reorders segment IDs within the provided coordinate array based on the given
    range or a specific order list. This function modifies the segment IDs in the
    coordinate data to either follow a sequence defined by a range or an explicitly
    provided order. If no range or list is provided, default ordering is applied
    based on the unique segment IDs.

    :param coord: A 2D numpy array where the first column contains the segment IDs
        to be reordered.
    :type coord: np.ndarray
    :param order_range: An optional list defining the range of values to be used
        when reordering segment IDs. If None, the IDs are reordered into a default
        range sequence.
    :type order_range: Optional[list]
    :param order_list: An optional list or numpy array defining the specific new
        order of segment IDs. If None, the IDs are reordered using a default range
        sequence.
    :type order_list: Optional[Union[list, np.ndarray]]

    :return: A numpy array with updated segment IDs, reflecting the reordering as
        per the provided range or specific order list.
    :rtype: np.ndarray
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
    Smoothens a given set of 3D points using spline interpolation to reduce variations and noise while
    maintaining the overall shape of the point cloud.

    If the points array includes an identifier column (ID), the function normalizes the coordinates,
    computes the pre- and post-smoothing tortuosity of the points, and ensures that the smoothed spline
    does not increase tortuosity excessively, serving as a failure check for the smoothing process.

    If the points array contains only the 3D coordinates ([X, Y, Z]), it directly computes the spline
    without any tortuosity checks or normalization.

    :param points:
        The array of input 3D points to be smoothened. The input array should have a shape of either
        (N, 3) for 3D points without identifiers, or (N, 4) for 3D points prefixed with an identifier column.
    :param s:
        The smoothness factor for spline interpolation. A smaller value ensures the spline closely conforms
        to the input points, while a larger value produces a smoother spline. Default is 0.5.

    :return:
        An array of smoothened 3D point coordinates. If an identifier column is present, it is preserved
        in the returned result. The output shape is consistent with the input, either (N, 4) for points
        with identifiers or (N, 3) for points without identifiers.
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
    Sorts and reorders segments in the coordinate array by their total length in ascending order.

    This function takes an input array of coordinates, identifies unique segment IDs,
    calculates the total length of each segment, and then sorts the segments by length.
    The output is the reordered array with updated segment IDs, maintaining their relative
    order after sorting.

    :param coord: A numpy array where the first column represents segment IDs and the
        subsequent columns represent coordinates.
    :type coord: numpy.ndarray

    :return: A numpy array with segments reordered by their total length, where segment IDs
        are updated and assigned sequentially in the sorted order.
    :rtype: numpy.ndarray
    """
    coord = reorder_segments_id(coord)

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


def cut_at_degree(segments_array: np.ndarray, cut_at=150):
    """
    Cuts segments based on angles between consecutive line segments and reassigns IDs
    to the newly split segments. The function calculates angles between vectors
    within provided segments, and if any angle is less than or equal to 150 degrees,
    the segment is split at the point with the smallest angle. Single-point segments
    are excluded from the output.

    :param segments_array: A numpy ndarray where each row represents a segment point.
        The first column contains segment IDs, and the remaining columns contain
        point coordinates.
    :type segments_array: numpy.ndarray

    :return: A tuple containing a boolean indicating whether any segments were split,
        and a numpy array of the updated segments with reassigned segment IDs.
    :rtype: tuple
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

        # Check if any angle is less than or equal to cut_at degrees
        if len([id_ for id_, k in enumerate(angles_) if k <= cut_at]) > 0:
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
