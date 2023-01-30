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
    def _in_cylinder(point: np.ndarray,
                     axis: tuple,
                     r: int,
                     h: int) -> bool:
        """
        Fast check if point is inside a cylinder by projecting point on cylinder
        volume in 3D.

        Args:
            point (np.ndarray): Point in 3D to project on cylinder.
            axis (tuple): Cylinder axis of orientation.
            r (int): Cylinder radius.
            h (int): Cylinder length.

        Returns:
            bool: If True, given point can be found in a cylinder.
        """
        # project the point onto the axis
        d = np.dot(point - axis[0], axis[1] - axis[0]) / np.linalg.norm(axis[1] - axis[0])

        # check if d is within the range [0, h]
        if d < 0 or d > h:
            return False

        # calculate the perpendicular distance
        d = np.linalg.norm(point - axis[0] - d * (axis[1] - axis[0]) /
                           np.linalg.norm(axis[1] - axis[0]))

        # check if d is less than the radius
        return d <= r

    def marge_splines(self,
                      point_cloud: np.ndarray,
                      omit_border: int) -> np.ndarray:
        """
        Connect splines in the point_cloud that are close enough to each other
         and return the reordered splines.

        Example:
            - Get dictionary with {id: [sorted list of points]}

            - While there is more than 1 spline in the dict:
                - Pick first spline in a dict.
                - Select all ends
                - Calc. end distance to all other ends.

                - If any ends within threshold distance:
                    - Calculate initial end vector
                    - Calculate distance to the cylinder

                    @ If any picked ends are within cylinder radius
                        # Pick one with the smallest distance
                        # Save two spline IDs to connected

                @ If Connected ID list not empty:
                    # Marge and sort points from two IDs
                    # Check tortuosity

                    @ If tortuosity <= 1.5:
                        # Add to the connected dict
                        # Remove from dict
                        # start over

                @ If connected ID list is empty or tortuosity > 1.5:
                    # Add only initial spline to the connected dict
                    # Remove from dict
                    # start over

        Args:
            point_cloud (np.ndarray): Array with segmented and sorted point cloud
            of a  shape [ID, X, Y, Z].
            omit_border (int): In A, distance from the border as a limit not to
            connect splines.

        Returns:
            np.ndarray: Array of segmented point cloud with connected splines
            that fit the criteria
        """
        # Find Z bordered to filter out MT connection at the borders
        splines_list = {}
        MIN_Z, MAX_Z = np.min(point_cloud[:, 3]), np.max(point_cloud[:, 3])

        # Create a dictionary to store spline information
        # Iterate through the point cloud and add points to their respective splines
        for point in point_cloud:
            id, x, y, z = point
            if id not in splines_list:
                splines_list[id] = []
            splines_list[id].append([x, y, z])

        # Iterate throw every spline in the list
        merge_splines = {}
        spline_id = 0
        while len(splines_list) > 1:
            key = list(splines_list.keys())[0]
            value = splines_list[key]  # Pick first spline in the dictionary
            end01 = value[0]
            end10 = value[-1]

            end01_list = [list(x)[1][0] for x in splines_list.items()]
            end10_list = [list(x)[1][-1] for x in splines_list.items()]

            # Check if any ends is within threshold distance
            end01_list01 = np.sqrt(np.sum((np.asarray(end01_list) - np.asarray(end01)) ** 2,
                                          axis=1))
            end01_list01 = [{id: dist} for id, dist in zip(list(splines_list.keys()),
                                                           end01_list01)
                            if dist <= 1000 and id != key]
            end01_list10 = np.sqrt(np.sum((np.asarray(end10_list) - np.asarray(end01)) ** 2,
                                          axis=1))
            end01_list10 = [{id: dist} for id, dist in zip(list(splines_list.keys()),
                                                           end01_list10)
                            if dist <= self.distance_th and id != key]
            end10_list01 = np.sqrt(np.sum((np.asarray(end01_list) - np.asarray(end10)) ** 2,
                                          axis=1))
            end10_list01 = [{id: dist} for id, dist in zip(list(splines_list.keys()),
                                                           end10_list01)
                            if dist <= self.distance_th and id != key]
            end10_list10 = np.sqrt(np.sum((np.asarray(end10_list) - np.asarray(end10)) ** 2,
                                          axis=1))
            end10_list10 = [{id: dist} for id, dist in zip(list(splines_list.keys()),
                                                           end10_list10)
                            if dist <= self.distance_th and id != key]

            # Check if any of the point is within the cylinder and get the closest one
            splines_to_merge = []
            end_lists = [end01_list01, end01_list10, end10_list01, end10_list10]

            for end_list in end_lists:
                for i in end_list:
                    m_id = list(i.keys())[-1 if end_list in [end10_list01, end10_list10]
                                          else 0]
                    m_end = list(splines_list[m_id])

                    not_at_the_border = np.all([(m_end[-1 if end_list in [end10_list01,
                                                                          end10_list10]
                                                else 0][2] - MIN_Z) >= omit_border,
                                                (MAX_Z - m_end[-1 if end_list
                                                               in [end10_list01,
                                                                   end10_list10]
                                                 else 0][2]) >= omit_border])

                    if not_at_the_border:
                        in_cylinder = self._in_cylinder(point=np.array(m_end[-1
                                                                       if end_list in [end10_list01,
                                                                                       end10_list10]
                                                                       else 0]),
                                                        axis=(np.array(value[-2 if end_list in [end10_list01,
                                                                                                end10_list10]
                                                                             else 1]),
                                                              np.array(value[-1 if end_list in [end10_list01,
                                                                                                end10_list10]
                                                                             else 0])),
                                                        r=self.cylinder_radius,
                                                        h=self.distance_th)
                        if in_cylinder:
                            splines_to_merge.append([m_id, m_end])

            # Check if any ends fit criteria for connection
            if len(splines_to_merge) == 1:  # Merge
                merged_spline = np.concatenate((value, splines_to_merge[0][1]))
                merge_splines[spline_id] = sort_segment(merged_spline)

                del splines_list[splines_to_merge[0][0]]
            elif len(splines_to_merge) > 1:  # If more than one find best
                if len(np.unique([x[0] for x in splines_to_merge])) == 1:
                    merged_spline = np.concatenate((value, splines_to_merge[0][1]))
                    del splines_list[splines_to_merge[0][0]]
                else:
                    end_lists = {}
                    for d in np.concatenate([end01_list01, end01_list10,
                                             end10_list01, end10_list10]):
                        end_lists.update(d)
                    end_lists_id = min(splines_to_merge, key=lambda x: end_lists[x[0]])[0]
                    end_lists = [x[1] for x in splines_to_merge if x[0] == end_lists_id]

                    merged_spline = np.concatenate((value, end_lists[0]))
                    del splines_list[end_lists_id]

                merge_splines[spline_id] = sort_segment(merged_spline)
            else:  # No merge found
                merge_splines[spline_id] = sort_segment(value)

            del splines_list[key]
            spline_id += 1

        # Add last spline to the new list
        key = list(splines_list.keys())[0]
        merge_splines[spline_id] = splines_list[key]

        return np.concatenate([np.hstack((np.expand_dims(np.repeat(id, len(array)), 1),
                                          array)) for id, array in merge_splines.items()])

    def __call__(self,
                 point_cloud: np.ndarray,
                 omit_border: int):
        past_l = 0

        while len(np.unique(point_cloud[:, 0])) != past_l:
            past_l = len(np.unique(point_cloud[:, 0]))
            point_cloud = self.marge_splines(point_cloud, omit_border)

        return point_cloud


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
                 cylinder_radius=200,
                 filter_short_segments=1000):
        self.connect_seg_if_closer_then = connect_seg_if_closer_then
        self.filter_short_segments = filter_short_segments

        self.marge_splines = FilterConnectedNearSegments(
            distance_th=connect_seg_if_closer_then,
            cylinder_radius=cylinder_radius
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
        """Remove splines with tortuous higher than 1.5"""
        tortuosity_list = []
        for i in np.unique(segments[:, 0]):
            x = segments[np.where(segments[:, 0] == int(i))[0], 1:]
            tortuosity_list.append(tortuosity(x))

        tortuosity_list = [id for id, i in enumerate(tortuosity_list) if i < 1.5]
        new_seg = []
        for i in tortuosity_list:
            new_seg.append(segments[np.where(segments[:, 0] == i), :])
        segments = np.hstack(new_seg)[0, :]

        """Connect segments with ends close to each other"""
        if self.connect_seg_if_closer_then > 0:
            segments = self.marge_splines(segments)

        """Remove too short splines"""
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
            list: List of MT from spatial graph 1 that match spatial graph 2.
        """
        match_sg1_sg2 = []

        for k in range(int(spatial_graph_1[:, 0].max())):
            sg1_spline = spatial_graph_1[spatial_graph_1[:, 0] == k, :]
            iou = []

            for j in range(int(spatial_graph_2[:, 0].max())):
                sg2_spline = spatial_graph_2[spatial_graph_2[:, 0] == j, :]
                iou.append(compare_splines_probability(sg1_spline[:, 1:],
                                                       sg2_spline[:, 1:],
                                                       self.dist_th))

            ids = [id for id, i in enumerate(iou) if np.sum(i) > 0 and i >= self.inter_th]
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

        # Select all splines from Amira that match Tardis
        match_with_tardis = [x for x in amira_tardis if x[1] != []]  # Splines with match
        amira_matches_with_tardis = np.unique([x[0] for x in match_with_tardis])

        # Select all splines from Amira that do not have match with Tardis
        amira_noise = [y for y in np.unique(amira_sg[:, 0]) if y not in amira_matches_with_tardis]
        amira_noise = np.stack([x for x in amira_sg if x[0] in amira_noise])

        """Compare Tardis with Amira"""
        tardis_amira = self._compare_spatial_graphs(tardis_sg, amira_sg)

        # Select all splines from Tardis that match Amira
        match_with_amira = [x for x in tardis_amira if x[1] != []]  # Splines with match
        tardis_matches_with_amira = np.unique([x[0] for x in match_with_amira])

        # Select all splines from Tardis that do not have match with Amira
        tardis_noise = [y for y in np.unique(tardis_sg[:, 0]) if y not in tardis_matches_with_amira]
        tardis_noise = np.stack([x for x in tardis_sg if x[0] in tardis_noise])

        # Select matching splines from comparison
        new_tardis = []
        mt_new_id = 0
        for i in tardis_matches_with_amira:
            df = np.stack([x for x in tardis_sg if x[0] == i])
            df[:, 1:] = sort_segment(df[:, 1:])
            df[:, 0] = mt_new_id
            mt_new_id += 1
            new_tardis.append(df)
        new_tardis = np.concatenate(new_tardis)

        new_amira = []
        mt_new_id = 0
        for i in amira_matches_with_tardis:
            df = np.stack([x for x in amira_sg if x[0] == i])
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
        t_before = tortuosity(points)

        tck, u = splprep(points.T, s=s)
        spline = np.stack(splev(u, tck)).T
        t_after = tortuosity(spline)

        ids = np.zeros((len(spline), 1))
        ids += id

        # Sanity check if spline smoothing failed
        if t_after > t_before:
            return np.hstack((ids, points))
        return np.hstack((ids, spline))
    else:  # [X, Y, Z]
        tck, u = splprep(points.T, s=s)

        return np.stack(splev(u, tck)).T


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
            id = np.where([sum(i)
                           for i in cdist(coord, coord)] == max([sum(i)
                                                                 for i in cdist(coord,
                                                                                coord)]))[0]

            new_c.append(coord[id[0]])
            coord = np.delete(coord, id[0], 0)

        kd = KDTree(coord)
        points = kd.query(np.expand_dims(new_c[len(new_c) - 1], 0))[1][0][0]

        new_c.append(coord[points])
        coord = np.delete(coord, points, 0)
    return np.stack(new_c)


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
    end_length = sqrt((coord[0][0] - coord[-1][0]) ** 2 +
                      (coord[0][1] - coord[-1][1]) ** 2 +
                      (coord[0][2] - coord[-1][2]) ** 2) + 1e-16

    return length / end_length


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
        length += sqrt((coord[id][0] - coord[id + 1][0]) ** 2 +
                       (coord[id][1] - coord[id + 1][1]) ** 2 +
                       (coord[id][2] - coord[id + 1][2]) ** 2)

    return length
