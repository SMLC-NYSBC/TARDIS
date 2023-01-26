#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from math import sqrt
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree


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
                 tardis_sg: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tardis_noise = [y for y in np.unique(tardis_sg[:, 0]) if
                        y not in all_tardis_matches]
        tardis_noise = tardis_sg[
                       [id for id, x in enumerate(tardis_sg[:, 0]) if x in tardis_noise],
                       :]

        """Compare Tardis with Amira"""
        tardis_amira = self._compare_spatial_graphs(tardis_sg, amira_sg)

        # Select all splines from Amira that match Tardis
        amira_match_sg = [x for x in tardis_amira if x[1] != []]
        all_amira_matches = np.unique(np.concatenate([x[1] for x in amira_match_sg]))

        # Select all splines from Tardis that do not have match with Amira
        amira_noise = [y for y in np.unique(amira_sg[:, 0]) if y not in all_amira_matches]
        amira_noise = amira_sg[
                      [id for id, x in enumerate(amira_sg[:, 0]) if x in amira_noise], :]

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
                 connect_seg_if_closer_then=1000,
                 filter_short_segments=1000):
        self.connect_seg_if_closer_then = connect_seg_if_closer_then
        self.filter_short_segments = filter_short_segments

        self.marge_splines = FilterConnectedNearSegments(
            distance_th=connect_seg_if_closer_then,
            cylinder_radius=200
        )

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
            segments = self.marge_splines(segments)

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
            id = np.where([sum(i) for i in cdist(coord, coord)] == max([sum(i) for i in
                                                                        cdist(coord,
                                                                              coord)]))[0]

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
        length += sqrt((coord[id][0] - coord[id + 1][0]) ** 2 + (
                    coord[id][1] - coord[id + 1][1]) ** 2 + (
                                   coord[id][2] - coord[id + 1][2]) ** 2)

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

    length = total_length(coord) + 1e-16
    end_length = sqrt((coord[0][0] - coord[-1][0]) ** 2 + (
                coord[0][1] - coord[-1][1]) ** 2 + (
                                  coord[0][2] - coord[-1][2]) ** 2) + 1e-16

    return length / end_length


def smooth_spline(points: np.ndarray,
                  s=1000.0):
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
        id = points[0, 0]
        points = points[:, 1:]

        tck, u = splprep(points.T, s=s)
        spline = np.stack(splev(u, tck)).T

        ids = np.zeros((len(spline), 1))
        ids += id

        return np.hstack((ids, spline))
    else:  # [X, Y, Z]
        tck, u = splprep(points.T, s=s)

        return np.stack(splev(u, tck)).T


class FilterConnectedNearSegments:
    """
    Connect splines based on spline trajectory and splines end distances.
    """

    def __init__(self,
                 distance_th=1000,
                 cylinder_radius=150):
        """
        Initialize the class with the distance threshold and cylinder radius parameters.

        Args:
            distance_th (int): Maximum distance between two spline endpoints for them
            to be considered for merging.
            cylinder_radius (int): Maximum distance between spline endpoint and the line
            connecting its neighboring endpoints for them to be considered for merging.
        """
        self.distance_th = distance_th
        self.cylinder_radius = cylinder_radius

    @staticmethod
    def _remove_duplicates(d: dict) -> dict:
        """
        Remove duplicate splines

        Args:
            d (dict): Dictionary containing splines.

        Returns:
            dict: dictionary containing unique splines.
        """
        new_d = {}
        [new_d.update({k: v.tolist()}) for k, v in d.items() if
         v.tolist() not in new_d.values()]

        return new_d

    @staticmethod
    def _find_cylinder_axis(p1: np.ndarray,
                            p2: np.ndarray) -> np.ndarray:
        """
        Find the vector pointing from p1 to p2 and normalize it to make it a
        unit vector.

        Args:
            p1 (np.ndarray): starting point of the vector
            p2 (np.ndarray): end point of the vector

        Returns:
            np.ndarray: normalized vector pointing from p1 to p2
        """
        # Find the vector pointing from p1 to p2
        axis = p2 - p1

        # Normalize the vector to make it a unit vector
        return axis / np.linalg.norm(axis)

    def __call__(self,
                 point_cloud: np.ndarray,
                 omit_border=500) -> np.ndarray:
        """
        Connect splines in the point_cloud that are close enough to each other
         and return the reordered splines.

         Args:
             point_cloud (np.ndarray): Array with segmented point cloud of a shape
            [ID, X, Y, Z]

        Returns:
            np.ndarray: Array of segmented point cloud with connected splines
            that fit the criteria
        """
        # Create a dictionary to store spline information
        splines_list = {}
        MIN_Z, MAX_Z = np.min(point_cloud[:, 3]), np.max(point_cloud[:, 3])

        # Iterate through the point cloud and add points to their respective splines
        for point in point_cloud:
            id, x, y, z = point
            if id not in splines_list:
                splines_list[id] = []
            splines_list[id].append([x, y, z])

        # Iterate through the splines and check for merging conditions
        new_splines = {}
        id = 0
        for id1 in splines_list:
            id1_match = False

            for id2 in splines_list:
                if id1 != id2:
                    # Calculate the distance between the end points of the splines
                    end01 = splines_list[id1][0]
                    end10 = splines_list[id1][-1]

                    end02 = splines_list[id2][0]
                    end20 = splines_list[id2][-1]
                    distance0102 = np.sqrt((end01[0] - end02[0]) ** 2 + (
                            end01[1] - end02[1]) ** 2 + (end01[2] - end02[2]) ** 2)
                    distance0120 = np.sqrt((end01[0] - end20[0]) ** 2 + (
                            end01[1] - end20[1]) ** 2 + (end01[2] - end20[2]) ** 2)

                    distance1002 = np.sqrt((end10[0] - end02[0]) ** 2 + (
                            end10[1] - end02[1]) ** 2 + (end10[2] - end02[2]) ** 2)
                    distance1020 = np.sqrt((end10[0] - end20[0]) ** 2 + (
                            end10[1] - end20[1]) ** 2 + (end10[2] - end20[2]) ** 2)

                    distance = np.min((distance0102, distance0120, distance1002,
                                       distance1020))

                    # Check if the distance is less than 100 nm
                    if self.distance_th >= distance:
                        # Check which ends potentially interact
                        min_dist = np.where((distance0102, distance0120, distance1002,
                                             distance1020) == distance)[0]
                        if len(min_dist) > 1:
                            min_dist = min_dist[0]

                        if min_dist in [0, 1]:
                            id1_end = splines_list[id1][0:2][::-1]

                            # Check if id1 ends are not on the border of tomogram
                            dist_to_border = [np.sqrt((end01[2] - MIN_Z) ** 2),
                                              np.sqrt((end01[2] - MAX_Z) ** 2)]
                            if np.any([True for d in dist_to_border if d <= omit_border]):
                                continue
                        else:
                            id1_end = splines_list[id1][-2:]

                            # Check if id1 ends are not on the border of tomogram
                            dist_to_border = [np.sqrt((end10[2] - MIN_Z) ** 2),
                                              np.sqrt((end10[2] - MAX_Z) ** 2)]
                            if np.any([True for d in dist_to_border if d <= omit_border]):
                                continue

                        if min_dist in [0, 2]:
                            id2_end = splines_list[id2][0:2][::-1]
                        else:
                            id2_end = splines_list[id2][-2:]

                        # Calculate normalized vector of id1
                        cylinder_axis = self._find_cylinder_axis(np.array(id1_end[0]),
                                                                 np.array(id1_end[1]))

                        # Project id2 end to the cylinder
                        projection = (np.array(id2_end[1]) - np.array((id1_end[1]))).dot(
                            cylinder_axis) * cylinder_axis + np.array(id1_end[1])

                        # Calculate distance of id2 end to the cylinder
                        id2_in_cylinder = sqrt(sum((np.array(
                            id2_end[1]) - projection) ** 2))

                        # Merge the splines
                        if id2_in_cylinder <= self.cylinder_radius:
                            df_spline = sort_segment(np.concatenate((splines_list[id1],
                                                                     splines_list[id2])))
                            new_splines[id] = df_spline
                            id += 1
                            id1_match = True
                            continue

            if not id1_match:
                new_splines[id] = np.array(splines_list[id1])
                id += 1

        new_splines = self._remove_duplicates(new_splines)

        splines_array = []
        for key in new_splines:
            for point in new_splines[key]:
                x, y, z = point
                splines_array.append([key, x, y, z])

        return reorder_segments_id(np.stack(splines_array))
