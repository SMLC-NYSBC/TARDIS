#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import itertools
from math import sqrt
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from tardis.utils.errors import TardisError


class SpatialGraphCompare:
    """
    Compare two spatial graphs and output filtered-out array's of splines based
    on similarity.

    This class take as na input two arrays of shape [n, 3 or 4] for 2D or 3D
    point cloud. This arrays contain [ID x X x Y x Z] dimensions.

    The comparison is archived by calculating cdist for all splines from one spatial
    graph to all splines from second spatial graph. And for each spline it output
    probability of similarity and splines id's.

    The Probability is calculated as a ration of points (in threshold contact)
    to all points in spline.

    The selection threshold for the spline interaction is given between 0 and 1.
    """

    def __init__(self,
                 distance_threshold: int,
                 interaction_threshold: float):
        self.dist_th = distance_threshold
        self.inter_th = interaction_threshold

    def _compare_spatial_graphs(self,
                                spatial_graph_1: np.ndarray,
                                spatial_graph_2: np.ndarray) -> list:
        """
        Wrapper to compare all MT's between two spatial graphs

        Args:
            spatial_graph_1 (np.ndarray): Spatial graph 1.
            spatial_graph_2 (np.ndarray): Spatial graph 2.

        Returns:
            list: list of MT from spatial graph 1 that match spatial graph 2.
        """
        match_sg1_sg2 = []

        for k in range(int(spatial_graph_1[:, 0].max())):
            tardis_rand = spatial_graph_1[spatial_graph_1[:, 0] == k, :]
            iou = []

            for j in range(int(spatial_graph_2[:, 0].max())):
                amira_rand = spatial_graph_2[spatial_graph_2[:, 0] == j, :]
                iou.append(compare_splines_probability(amira_rand[:, 1:],
                                                       tardis_rand[:, 1:],
                                                       self.dist_th))
            ids = [id for id, i in enumerate(iou) if np.sum(i) > 0 and i > self.inter_th]

            match_sg1_sg2.append([k, ids])

        return match_sg1_sg2

    def __call__(self,
                 amira_sg: np.ndarray,
                 tardis_sg: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
        """
        Compute comparison of Amira and Tardis spatial graphs and output tuple of
        arrays with different selected MTs:
            - Label1: MT taken from the Tardis (matches Amira)
            - Label2: MT taken from the Amira (matches Tardis)
            - Label3: MT in Tardis without match
            - Label4: MT in Amira without match

        Args:
            amira_sg (np.ndarray): Spatial graph [ID, X, Y, Z] from Amira.
            tardis_sg (np.ndarray): Spatial graph [ID, X, Y, Z] from Tardis.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of all arrays
        """
        """Compare Amira with Tardis"""
        amira_tardis = self._compare_spatial_graphs(amira_sg, tardis_sg)

        # Select all splines from Tardis that match Amira
        tardis_match_sg = [x for x in amira_tardis if x[1] != []]
        all_tardis_matches = np.unique(np.concatenate([x[1] for x in tardis_match_sg]))

        # Select all splines from Tardis that do not have match with Amira
        tardis_noise = [y for y in np.unique(tardis_sg[:, 0])
                        if y not in all_tardis_matches]
        tardis_noise = tardis_sg[[id for id, x in enumerate(tardis_sg[:, 0])
                                  if x in tardis_noise], :]

        """Compare Tardis with Amira"""
        tardis_amira = self._compare_spatial_graphs(tardis_sg, amira_sg)

        # Select all splines from Amira that match Tardis
        amira_match_sg = [x for x in tardis_amira if x[1] != []]
        all_amira_matches = np.unique(np.concatenate([x[1] for x in amira_match_sg]))

        # Select all splines from Tardis that do not have match with Amira
        amira_noise = [y for y in np.unique(amira_sg[:, 0])
                       if y not in all_amira_matches]
        amira_noise = amira_sg[[id for id, x in enumerate(amira_sg[:, 0])
                                if x in amira_noise], :]

        # Select MT from comparison
        new_tardis = []
        mt_new_id = 0
        for i in tardis_match_sg:
            df = tardis_sg[[id for id, x in enumerate(tardis_sg[:, 0]) if x in i[1]], :]
            df[:, 1:] = sort_segment(df[:, 1:])
            df[:, 0] = mt_new_id
            mt_new_id += 1
            new_tardis.append(df)
        new_tardis = np.concatenate(new_tardis)

        new_amira = []
        mt_new_id = 0
        for i in amira_match_sg:
            df = amira_sg[[id for id, x in enumerate(amira_sg[:, 0]) if x in i[1]], :]
            df[:, 1:] = sort_segment(df[:, 1:])
            df[:, 0] = mt_new_id
            mt_new_id += 1
            new_amira.append(df)
        new_amira = np.concatenate(new_amira)

        return new_tardis, tardis_noise, new_amira, amira_noise


def compare_splines_probability(spline_1: np.ndarray,
                                spline_2: np.ndarray,
                                threshold=100):
    """
    Compare two splines and calculate probability of how likely given two
    splines are the same line given array of points for same or similar splines
    with no matching coordinates of points.

    Calculates the probability of two splines being similar by comparing
    the distance between their points and taking the mean of the matching
    points below a threshold.

    Parameters:
        spline_1 (np.ndarray): The first spline to compare, represented
        as an array of points.
        spline_2 (np.ndarray): The second spline to compare, represented
        as an array of points.
        threshold (int): The maximum distance between points for them to be
        considered matching.

    Returns:
        float: The probability of the splines being similar, ranging from 0.0 to 1.0.
    """
    if len(spline_1) == 0 or len(spline_2) == 0:
        return 0.0

    # Calculating distance matrix between points of 2 splines
    dist_matrix = cdist(spline_1, spline_2)

    # Calculating the matching point from both splines
    matching_points = np.min(dist_matrix, axis=1)

    # Filtering out distance below threshold
    matching_points = matching_points[matching_points < threshold]

    # If no matching points probability is 0
    if len(matching_points) == 0:
        return 0.0

    # Calculating probability using mean of the matching point below threshold
    probability = len(matching_points) / len(spline_1)

    return probability


class FilterSpatialGraph:
    """
    Calculate length of each spline and distance between all splines ends.

    This clas iterate over all splines in array [ID, X, Y, Z] by given ID.
    Firstly if specified during initialization, class calculate distance
    between all splines ends and use it to define which splines are obviously
    broken.
    Then it calculate length of all the splines (also new one) and filter out
    ones that are too short.
    """
    def __init__(self,
                 connect_seg_if_closer_then=175,
                 filter_short_segments=1000):
        self.connect_seg_if_closer_then = connect_seg_if_closer_then
        self.filter_short_segments = filter_short_segments

    def __call__(self,
                 segments: np.ndarray) -> np.ndarray:
        """
        Connect splines that have their end's in close distance and remove
        splines that are too short.

        Args:
            segments (np.ndarray): Array of points with ID label of shape [ID, X, Y, Z]

        Returns:
            np.ndarray: Filtered array of connected MTs
        """
        if self.connect_seg_if_closer_then > 0:
            # Connect segments with ends close to each other
            segments = filter_connect_near_segment(segments,
                                                   self.connect_seg_if_closer_then)

        if self.filter_short_segments > 0:
            length = []
            for i in np.unique(segments[:, 0]):
                length.append(total_length(segments[np.where(segments[:, 0] == int(i))[0],
                                           1:]))

            length = [id for id, i in enumerate(length) if i > self.filter_short_segments]

            new_seg = []
            for i in length:
                new_seg.append(segments[np.where(segments[:, 0] == i), :])

            segments = np.hstack(new_seg)[0, :]

        return reorder_segments_id(segments)


def reorder_segments_id(coord: np.ndarray,
                        order_range: Optional[list] = None) -> np.ndarray:
    """
    Reorder list of segments to remove missing IDs

    E.g. Change IDs from [1, 2, 3, 5, 6, 8] to [1, 2, 3, 4, 5, 6]

    Args:
        coord: Array of points in 3D or 3D with their segment ID
        order_range: Costume id range for reordering

    Returns:
        np.ndarray: Array of points with reordered IDs values
    """
    df = np.unique(coord[:, 0])

    if order_range is None:
        df_range = np.asarray(range(0, len(df)), dtype=df.dtype)
    else:
        df_range = np.asarray(range(order_range[0], order_range[1]), dtype=df.dtype)

    for id, i in enumerate(coord[:, 0]):
        coord[id, 0] = df_range[np.where(df == i)[0][0]]

    return coord


def sort_segment(coord: np.ndarray) -> np.ndarray:
    """
    Sorting of the point cloud based on number of connections followed by
    searching of the closest point with cdist.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        np.ndarray: Array of point in line order.
    """
    if len(coord) == 0:
        return coord

    new_c = []
    for i in range(len(coord) - 1):
        if i == 0:
            id = np.where([sum(i) for i in cdist(coord, coord)] == max(
                [sum(i) for i in cdist(coord, coord)]
            ))[0]

            new_c.append(coord[id[0]])
            coord = np.delete(coord, id[0], 0)

        kd = KDTree(coord)
        points = kd.query(np.expand_dims(new_c[len(new_c) - 1], 0))[1][0][0]

        new_c.append(coord[points])
        coord = np.delete(coord, points, 0)
    return np.stack(new_c)


def total_length(coord: np.ndarray) -> float:
    """
    Calculate total length of the spline.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        float: Spline length.
    """
    length = 0
    c_len = len(coord) - 1

    for id, _ in enumerate(coord):
        if id == c_len:
            break

        # sqrt((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)
        length += sqrt(pow((coord[id][0] - coord[id + 1][0]), 2) +
                       pow((coord[id][1] - coord[id + 1][1]), 2) +
                       pow((coord[id][1] - coord[id + 1][1]), 2))

    return length


def tortuosity(coord: np.ndarray) -> float:
    """
    Calculate spline tortuosity.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        float: Spline curvature measured with tortuosity.
    """
    if len(coord) == 0:
        return 1.0

    length = total_length(coord)
    end_length = sqrt((coord[0][0] - coord[-1][0]) ** 2 +
                      (coord[0][1] - coord[-1][1]) ** 2 +
                      (coord[0][2] - coord[-1][2]) ** 2)

    return (length + 1e-16) / (end_length + 1e-16)


def filter_connect_near_segment(segments: np.ndarray,
                                dist_th=200) -> np.ndarray:
    """
    Find and connect segments with ends close to each other.

    Args:
        segments (np.ndarray): 3D array of all segments [ID, XYZ].
        dist_th (int): Distance threshold for connecting spline ends.

    Returns:
        np.ndarray: Array of segments with segments connected based on end distance.
    """
    seg_list = [segments[np.where(segments[:, 0] == i), :][0]
                for i in np.unique(segments[:, 0])]

    # Find all segments ends
    ends = [s[0] for s in seg_list] + [s[-1] for s in seg_list]
    ends_coord = [s[0, 1:] for s in seg_list] + [s[-1, 1:] for s in seg_list]

    kd = cdist(ends_coord, ends_coord)

    # Find segments pair with ends in dist_th distance
    df = []
    for i in kd:
        df.append(np.where(i < dist_th)[0])
    idx_connect = sorted([[int(ends[i[0]][0]), int(ends[i[1]][0])]
                          for i in df if len(i) > 1])
    idx_connect = list(k for k, _ in itertools.groupby(idx_connect))

    if len(idx_connect) == 0:
        return segments

    s = set()
    a1 = []
    for t in idx_connect:
        if tuple(t) not in s:
            a1.append(t)
            s.add(tuple(t)[::-1])
    idx_connect = list(s)

    # Select segments without any pair
    new_seg = []
    for i in [int(id) for id in np.unique(segments[:, 0])
              if id not in np.unique(np.concatenate(idx_connect))]:
        new_seg.append(segments[np.where(segments[:, 0] == i), :])
    if len(new_seg) > 0:
        new_seg = np.hstack(new_seg)[0, :]

        # Fix breaks in spline numbering
        new_seg = reorder_segments_id(new_seg)

    connect_seg = []
    for i in [int(id) for id in np.unique(segments[:, 0])
              if id in np.unique(np.concatenate(idx_connect))]:
        connect_seg.append(segments[np.where(segments[:, 0] == i), :])
    connect_seg = np.hstack(connect_seg)[0, :]

    assert len(new_seg) + len(connect_seg) == len(segments), \
        TardisError('116',
                    'tardis/dist/utils/segment_point_cloud.py',
                    f'New segment has incorrect number of points '
                    f'{len(new_seg) + len(connect_seg)} != {len(segments)}')

    # Connect selected segments pairs
    idx = 1000000
    for i in idx_connect:
        for j in i:
            df = np.where(connect_seg[:, 0] == j)[0]
            connect_seg[df, 0] = idx
        idx += 1
    assert len(new_seg) + len(connect_seg) == len(segments), \
        TardisError('116',
                    'tardis/dist/utils/segment_point_cloud.py',
                    f'New segment has incorrect number of points '
                    f'{len(new_seg) + len(connect_seg)} != {len(segments)}')

    # Fix breaks in spline numbering
    if len(new_seg) > 0:
        connect_seg = reorder_segments_id(connect_seg,
                                          order_range=[int(np.max(new_seg[:, 0])) + 1,
                                                       int(np.max(new_seg[:, 0])) + 1 +
                                                       len(np.unique(connect_seg[:, 0]))])
    else:
        connect_seg = reorder_segments_id(connect_seg)

    connect_seg_sort = []
    for i in np.unique(connect_seg[:, 0]):
        connect_seg_sort.append(sort_segment(
            connect_seg[np.where(connect_seg[:, 0] == int(i)), :][0]))
    connect_seg = np.vstack(connect_seg_sort)

    if len(new_seg) > 0:
        new_seg = np.concatenate((new_seg, connect_seg))

        return new_seg
    else:
        return connect_seg
