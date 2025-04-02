#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from tardis_em.analysis.filament_utils import (
    cut_at_degree,
    reorder_segments_id,
    sort_segment,
    resample_filament
)
from tardis_em.analysis.geometry_metrics import total_length
from tardis_em.utils.errors import TardisError


class FilterConnectedNearSegments:
    """
    This class is used to manage and process splines in a point cloud by
    connecting nearby segments based on specific cylindrical geometry
    criteria. It handles operations such as determining cylindrical overlap,
    removing duplicates, and merging splines.
    """

    def __init__(self, distance_th=1000, cylinder_radius=150):
        """
        This class is used to configure the parameters for a specific cylindrical
        geometry operation. It allows specifying a distance threshold and a cylinder
        radius to be utilized in the process or calculations related to a cylindrical
        configuration.

        :param distance_th: The threshold distance parameter used for operations,
            which defines the maximum allowable distance.
        :param cylinder_radius: The radius of the cylinder, used for calculations
            related to the cylindrical geometry.
        """
        self.distance_th = distance_th
        self.cylinder_radius = cylinder_radius

    @staticmethod
    def _in_cylinder(point: np.ndarray, axis: tuple, r: int, h: int) -> bool:
        """
        Determines whether a given point lies within a cylinder defined by an axis,
        a radius, and a height. This static method projects the point onto the
        cylinder's axis, checks if it lies within the height boundaries, and
        calculates the perpendicular distance from the axis to verify if it
        is within the radius.

        :param point: A NumPy array representing the point to be checked.
        :param axis: A tuple containing two endpoints of the cylinder's axis,
            where each endpoint is a NumPy array.
        :param r: The cylinder's radius as an integer.
        :param h: The cylinder's height as an integer.

        :return: A boolean value indicating whether the point lies within
            the cylinder.
        """
        # project the point onto the axis
        ax1 = (point - axis[0]) + 1e-16
        ax2 = (axis[1] - axis[0]) + 1e-16

        d = np.dot(ax1, ax2) / np.linalg.norm(ax2)

        # check if d is within the range [0, h]
        if d < 0 or d > h:
            return False

        # calculate the perpendicular distance
        d = np.linalg.norm(ax1 - d * (ax2) / np.linalg.norm(ax2))

        # check if d is less than the radius
        return d <= r

    @staticmethod
    def _remove_duplicates(d: dict) -> dict:
        """
        Removes duplicate values in a given dictionary by converting the values to
        lists and ensuring only unique lists are retained. The method iterates through
        the provided dictionary, converts each value into a list, and updates a new
        dictionary with key-value pairs only if the list version of the value is not
        already present in the new dictionary.

        :param d: A dictionary where values are to be checked for uniqueness after
            conversion to lists.
        :type d: dict

        :return: A new dictionary containing only unique key-value pairs from the
            original dictionary, based on the uniqueness of the list-converted values.
        :rtype: dict
        """
        new_d = {}
        [
            new_d.update({k: v.tolist()})
            for k, v in d.items()
            if v.tolist() not in new_d.values()
        ]

        return new_d

    def splines_direction(self, spline1: list, spline2: list) -> bool:
        """
        Determines whether two splines are facing the same direction within specific spatial
        constraints. This function evaluates the positional relationships of two splines based
        on predefined conditions involving distance and orientation in a cylindrical coordinate
        system.

        :param spline1: The first spline represented as a list of points, where each point is a
            coordinate in 2D or 3D space.
        :param spline2: The second spline represented as a list of points, where each point is
            a coordinate in 2D or 3D space.

        :return: A boolean value indicating whether the two splines face in the same direction
            based on the specified criteria.
        """
        # Check 01 - 01 & Check 01 - 10
        ax = [
            (np.array(spline2[1]), np.array(spline2[0])),
            (np.array(spline2[1]), np.array(spline2[0])),
        ]
        points = [np.array(spline1[0]), np.array(spline1[-1])]
        s201_s101, s201_s110 = [
            self._in_cylinder(
                point=p, axis=a, r=self.cylinder_radius, h=self.distance_th
            )
            for p, a in zip(points, ax)
        ]

        # Check 10 - 01 and Check 10 - 10
        ax = [
            (np.array(spline2[-2]), np.array(spline2[-1])),
            (np.array(spline2[-2]), np.array(spline2[-1])),
        ]
        points = [np.array(spline1[0]), np.array(spline1[-1])]
        s210_s101, s210_s110 = [
            self._in_cylinder(
                point=p, axis=a, r=self.cylinder_radius, h=self.distance_th
            )
            for p, a in zip(points, ax)
        ]

        # Check if splines facing same direction in any way
        return np.any((s201_s101, s201_s110, s210_s101, s210_s110))

    def marge_splines(
        self, point_cloud: np.ndarray, omit_border: int, initial: bool
    ) -> np.ndarray:
        """
        Merge splines from a given point cloud dataset based on certain distance and
        alignment criteria. Each spline in the point cloud is analyzed and combined
        with other splines if they satisfy conditions such as proximity, alignment,
        and not being within a border exclusion zone. The function returns the
        merged splines as a numpy array.

        :param point_cloud: A numpy array representing the point cloud data, where
            each row corresponds to a point with information such as spline ID,
            x-coordinate, y-coordinate, and z-coordinate.
        :type point_cloud: numpy.ndarray

        :param omit_border: An integer value specifying the range from the minimum
            and maximum along the z-axis for excluding points at borders, used to
            eliminate spurious connections near boundaries.
        :type omit_border: int

        :param initial: A boolean indicating whether to process and create an
            initial dictionary of splines based on unique spline IDs. Controls the
            creation and initialization of spline data structures.
        :type initial: bool

        :return: A numpy array representing the merged splines, where each row
            corresponds to a point with updated spline IDs, x-coordinate,
            y-coordinate, and z-coordinate. The merged splines preserve the
            consistency of segment continuity and alignment.
        :rtype: numpy.ndarray
        """
        # Find Z bordered to filter out MT connection at the borders
        MIN_Z, MAX_Z = np.min(point_cloud[:, -1]), np.max(point_cloud[:, -1])

        # Create a dictionary to store spline information
        # Iterate through the point cloud and add points to their respective splines
        if initial:
            splines_list_df = {}
            for point in point_cloud:
                id_, x, y, z = point
                if id_ not in splines_list_df:
                    splines_list_df[id_] = []
                splines_list_df[id_].append([x, y, z])

            splines_list = {}
            for i in splines_list_df:
                value = splines_list_df[i]
                if len(value) > 5:
                    splines_list[i] = value
        else:
            splines_list = {}
            for point in point_cloud:
                id_, x, y, z = point
                if id_ not in splines_list:
                    splines_list[id_] = []
                splines_list[id_].append([x, y, z])

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
            end01_list01 = np.sqrt(
                np.sum((np.asarray(end01_list) - np.asarray(end01)) ** 2, axis=1)
            )
            end01_list01 = [
                {id_i: dist}
                for id_i, dist in zip(list(splines_list.keys()), end01_list01)
                if dist <= self.distance_th and id_i != key
            ]
            end01_list10 = np.sqrt(
                np.sum((np.asarray(end10_list) - np.asarray(end01)) ** 2, axis=1)
            )
            end01_list10 = [
                {id_i: dist}
                for id_i, dist in zip(list(splines_list.keys()), end01_list10)
                if dist <= self.distance_th and id_i != key
            ]
            end10_list01 = np.sqrt(
                np.sum((np.asarray(end01_list) - np.asarray(end10)) ** 2, axis=1)
            )
            end10_list01 = [
                {id_i: dist}
                for id_i, dist in zip(list(splines_list.keys()), end10_list01)
                if dist <= self.distance_th and id_i != key
            ]
            end10_list10 = np.sqrt(
                np.sum((np.asarray(end10_list) - np.asarray(end10)) ** 2, axis=1)
            )
            end10_list10 = [
                {id_i: dist}
                for id_i, dist in zip(list(splines_list.keys()), end10_list10)
                if dist <= self.distance_th and id_i != key
            ]

            # Check if any of the point is within the cylinder and get the closest one
            splines_to_merge = []
            end_lists = [end01_list01, end01_list10, end10_list01, end10_list10]

            for end_list in end_lists:
                for i in end_list:
                    end_bool = end_list in [end10_list01, end10_list10]
                    m_id = list(i.keys())[-1 if end_bool else 0]
                    m_end = list(splines_list[m_id])

                    border_ends = [m_end[0][2], m_end[-1][2]]
                    not_at_the_border = np.all(
                        [
                            np.abs(np.min(border_ends) - MIN_Z) >= omit_border,
                            np.abs(np.max(border_ends) - MAX_Z) >= omit_border,
                        ]
                    )

                    if not_at_the_border:
                        points = np.array(
                            m_end[-1 if end_list in [end10_list01, end10_list10] else 0]
                        )
                        axis = (
                            np.array(
                                value[
                                    (
                                        -2
                                        if end_list in [end10_list01, end10_list10]
                                        else 1
                                    )
                                ]
                            ),
                            np.array(
                                value[
                                    (
                                        -1
                                        if end_list in [end10_list01, end10_list10]
                                        else 0
                                    )
                                ]
                            ),
                        )
                        in_cylinder = self._in_cylinder(
                            point=points,
                            axis=axis,
                            r=self.cylinder_radius,
                            h=self.distance_th,
                        )
                        if in_cylinder:
                            splines_to_merge.append([m_id, m_end])

            # Check if any ends fit criteria for connection
            if len(splines_to_merge) == 1:  # Merge
                # Check if selected spline facing the same direction
                same_direction = self.splines_direction(value, splines_to_merge[0][1])

                # Connect splines
                if same_direction:
                    merged_spline = np.concatenate((value, splines_to_merge[0][1]))
                    merge_splines[spline_id] = sort_segment(merged_spline)

                    del splines_list[splines_to_merge[0][0]]
            elif len(splines_to_merge) > 1:  # If more than one find best
                if len(np.unique([x[0] for x in splines_to_merge])) == 1:
                    # Check if selected spline facing the same direction
                    same_direction = self.splines_direction(
                        value, splines_to_merge[0][1]
                    )

                    # Connect splines
                    if same_direction:
                        merged_spline = np.concatenate((value, splines_to_merge[0][1]))

                        merge_splines[spline_id] = sort_segment(merged_spline)
                        del splines_list[splines_to_merge[0][0]]
                else:
                    end_lists = {}
                    for d in np.concatenate(
                        [end01_list01, end01_list10, end10_list01, end10_list10]
                    ):
                        end_lists.update(d)

                    # Check which splines facing the same direction
                    same_direction = []
                    for d in end_lists:
                        same_direction.append(
                            self.splines_direction(value, splines_list[d])
                        )

                    # Pick splines with the smallest distance that facing same direction
                    end_lists_id = [x for x, b in zip(end_lists, same_direction) if b]
                    same_direction = np.any(same_direction)

                    if len(end_lists_id) > 0:
                        end_lists_id = min(end_lists_id, key=lambda x: end_lists[x])
                        merged_spline = np.concatenate(
                            (value, splines_list[end_lists_id])
                        )

                        merge_splines[spline_id] = sort_segment(merged_spline)
                        del splines_list[end_lists_id]
            else:  # No merge found
                same_direction = True
                merge_splines[spline_id] = sort_segment(value)

            if not same_direction:
                merge_splines[spline_id] = sort_segment(value)

            del splines_list[key]
            spline_id += 1

        # Add the last spline to the new list
        try:
            key = list(splines_list.keys())[0]
            merge_splines[spline_id] = splines_list[key]
        except IndexError:
            pass

        return np.concatenate(
            [
                np.hstack((np.repeat(id_, len(array)).reshape(-1, 1), array))
                for id_, array in merge_splines.items()
            ]
        )

    def __call__(self, point_cloud: np.ndarray, omit_border=0):
        """
        Process and refine a given point cloud to merge splines iteratively until no further
        unique changes occur in the first dimension of the coordinates. The method takes
        in a point cloud and an optional parameter to omit border constraints. It applies
        the `marge_splines` function, starting with an initial pass, followed by iterative
        processing of the data.

        :param point_cloud: A 2D numpy array where each row represents a point
            in the point cloud, and columns represent the spatial dimensions.
        :param omit_border: An integer specifying the level of border omission
            during spline merging. Default value is 0.

        :return: A modified numpy array representing the updated point cloud
            after iterative refinement and spline merging.
        """
        past_l = 0
        while len(np.unique(point_cloud[:, 0])) != past_l:
            past_l = len(np.unique(point_cloud[:, 0]))
            point_cloud = self.marge_splines(
                point_cloud=point_cloud, omit_border=omit_border, initial=True
            )
        return point_cloud


class FilterSpatialGraph:
    """
    A class for filtering and refining spatial graphs by connecting nearby
    segments and removing short segments.

    This class is designed to process a given array of segments representing
    spatial structures, such as microtubules or similar graph-based
    representations. The primary tasks include connecting splines with
    endpoints within a specified distance and removing splines that are below
    a defined length threshold. Additionally, iterative optimization is
    performed to split connections at sharp angles.
    """

    def __init__(
        self,
        connect_seg_if_closer_then=1000,
        cylinder_radius=200,
        filter_short_segments=1000,
    ):
        """
        This class defines the initialization process for configuring parameters related
        to filtering and merging connected segments. It includes settings for connection
        thresholds, segment filtering, and spline merging operations.

        :param connect_seg_if_closer_then: Threshold distance for considering segments
            close enough to connect.
        :type connect_seg_if_closer_then: int
        :param cylinder_radius: Radius of the cylinder used in the connection operation.
        :type cylinder_radius: int
        :param filter_short_segments: Minimum length of segments to retain during the
            filtering process.
        :type filter_short_segments: int
        """
        self.connect_seg_if_closer_then = connect_seg_if_closer_then
        self.filter_short_segments = filter_short_segments

        self.marge_splines = FilterConnectedNearSegments(
            distance_th=connect_seg_if_closer_then, cylinder_radius=cylinder_radius
        )

    def __call__(self, segments: np.ndarray, px=None) -> np.ndarray:
        """
        Performs iterative optimization to modify input segments by applying a series of transformations,
        such as cutting connections at specific angles, merging close segment ends, and filtering short
        segments. The process is repeated to adjust the segments iteratively based on defined criteria.

        :param segments: The input array of segments, where each row represents segment information
                         and typically includes segment ID and its coordinates.
        :type segments: np.ndarray

        :return: The modified segments array after optimization transformations have been applied.
        :rtype: np.ndarray
        """
        """Do iterative optimization split 150 degree connection / marge"""
        # Split 150 degree connections
        if px is None:
            spacing = 5
            spacing_rev = 5
        else:
            spacing = 2500 / (px / 2)
            spacing_rev = 2500 / px
        segments = resample_filament(segments, spacing)

        loop_b = True
        while loop_b:
            loop_b, segments = cut_at_degree(segments, 150)

        # Connect segments with ends close to each other
        border = [np.min(segments[:, -1]), np.max(segments[:, -1])]
        border = np.abs(border[0] - border[1]) * 0.015

        if self.connect_seg_if_closer_then > 0:
            past_l = 0
            while len(np.unique(segments[:, 0])) != past_l:
                past_l = len(np.unique(segments[:, 0]))
                segments = self.marge_splines(point_cloud=segments, omit_border=border)
                segments = reorder_segments_id(segments)

                loop_b = True
                while loop_b:
                    loop_b, segments = cut_at_degree(segments, 150)

            segments = resample_filament(segments, spacing_rev)

        """Remove too short splines"""
        if self.filter_short_segments > 0:
            length = []
            for i in np.unique(segments[:, 0]):
                length.append(
                    total_length(segments[np.where(segments[:, 0] == int(i))[0], 1:])
                )

            length = [
                id_ for id_, i in enumerate(length) if i > self.filter_short_segments
            ]

            new_seg = []
            for i in length:
                new_seg.append(segments[np.where(segments[:, 0] == i), :])

            if len(new_seg) > 0:
                segments = np.hstack(new_seg)[0, :]
                segments = reorder_segments_id(segments)

        return reorder_segments_id(segments)


class SpatialGraphCompare:
    """
    Compares spatial graphs to identify matching microtubules (MTs) and classify
    them based on specific criteria.

    This class is designed to analyze and compare spatial graphs, which represent
    microtubule structures, based on user-defined thresholds. It identifies matching
    microtubules between two spatial graphs by calculating probabilities of alignment.
    The class is used for both comparing spatial structures and filtering instances
    based on pre-defined distance and interaction thresholds.
    """

    def __init__(self, distance_threshold: int, interaction_threshold: float):
        """
        A class to define thresholds for distance and interaction, typically used for
        evaluating data or enforcing certain constraints in various applications. The
        thresholds are initialized during the creation of the class instance and are
        used as fundamental parameters throughout the class usage.

        :param distance_threshold: A distance value that acts as a limiting measure,
                                   specified as an integer.
        :param interaction_threshold: A floating-point value representing the interaction
                                       threshold, used to determine limits or criteria.
        """
        self.dist_th = distance_threshold
        self.inter_th = interaction_threshold

    def _compare_spatial_graphs(
        self, spatial_graph_1: np.ndarray, spatial_graph_2: np.ndarray
    ) -> list:
        """
        Compares two spatial graphs and matches the splines based on the given
        distance and intersection thresholds. For each unique spline in the first
        spatial graph, the function computes the probability of matching splines
        in the second spatial graph. Spline pairs are compared using the
        `compare_splines_probability` function, and matches are recorded if they
        meet the specified thresholds.

        :param spatial_graph_1: A numpy array representing the first spatial
            graph. Each row should contain information about a single spline,
            with the first column indicating the spline index and the remaining
            columns containing spatial data.
        :param spatial_graph_2: A numpy array representing the second spatial
            graph. Each row should contain information about a single spline,
            with the first column indicating the spline index and the remaining
            columns containing spatial data.

        :return: A list of lists, where each inner list contains the
            spline index from `spatial_graph_1` and the corresponding
            indices of splines in `spatial_graph_2` that meet the match criteria.
        """
        match_sg1_sg2 = []

        for i in np.unique(spatial_graph_1[:, 0]):
            sg1_spline = spatial_graph_1[spatial_graph_1[:, 0] == i, :]
            iou = []

            for j in np.unique(spatial_graph_2[:, 0]):
                sg2_spline = spatial_graph_2[spatial_graph_2[:, 0] == j, :]
                iou.append(
                    compare_splines_probability(
                        sg1_spline[:, 1:], sg2_spline[:, 1:], self.dist_th
                    )
                )

            ids = [
                id for id, i in enumerate(iou) if np.sum(i) > 0 and i >= self.inter_th
            ]
            match_sg1_sg2.append([i, ids])

        return match_sg1_sg2

    def __call__(
        self, amira_sg: np.ndarray, tardis_sg: np.ndarray
    ) -> Tuple[list, list]:
        """
        Compares spatial graphs from Amira and Tardis, categorizing their components into
        matched and noise groups based on mutual comparisons. The function processes the
        spatial graphs to output lists of matched and unmatched elements for both input sets.

        :param amira_sg: A numpy array representing the spatial graph data from the
            Amira system.
        :type amira_sg: np.ndarray
        :param tardis_sg: A numpy array representing the spatial graph data from the
            Tardis system.
        :type tardis_sg: np.ndarray

        :return: A tuple containing two lists:
            1. A list of numpy arrays, where each array holds the categorized spatial
               graph components (e.g., matched and noise data for Tardis and Amira).
            2. A list of labels corresponding to each categorized array.
        :rtype: Tuple[list, list]
        """
        label = [
            "TardisFilterBasedOnAmira",
            "TardisNoise",
            "AmiraFilterBasedOnTardis",
            "AmiraNoise",
        ]

        """Amira splines scores"""
        amira_comp = self._compare_spatial_graphs(
            spatial_graph_1=amira_sg, spatial_graph_2=tardis_sg
        )

        """Compare Tardis with Amira"""
        tardis_comp = self._compare_spatial_graphs(
            spatial_graph_1=tardis_sg, spatial_graph_2=amira_sg
        )

        # Select all splines from Tardis that match Amira
        match = [x[0] for x in tardis_comp if x[1] != []]
        tardis_match = np.stack([x for x in tardis_sg if x[0] in match])

        noise = [x[0] for x in tardis_comp if x[1] == []]
        tardis_noise = np.stack([x for x in tardis_sg if x[0] in noise])

        # Select all splines from Amira that match Tardis
        match = [x[0] for x in amira_comp if x[1] != []]
        amira_match = np.stack([x for x in amira_sg if x[0] in match])

        noise = [x[0] for x in amira_comp if x[1] == []]
        amira_noise = np.stack([x for x in amira_sg if x[0] in noise])

        spatial_graphs = [
            g
            for g in (tardis_match, tardis_noise, amira_match, amira_noise)
            if g is not None
        ]
        label = [
            l
            for g, l in zip(
                (tardis_match, tardis_noise, amira_match, amira_noise), label
            )
            if g is not None
        ]

        return spatial_graphs, label


class ComputeConfidenceScore:
    @staticmethod
    def _angle_smoothness(tangents):
        """
        Computes the smoothness of the given array of tangents based on their angular
        deviation. The smoothness is calculated as `1 - standard deviation of angles
        between consecutive tangent vectors`. If the array of tangents is invalid, all
        values are zero, or nearly zero, the method will return 1.0 indicating maximum
        smoothness.

        :param tangents: A 2D numpy array of shape (n, m) where `n` is the number of
            tangent vectors, each in `m` dimensions. Represents the tangents whose
            smoothness will be evaluated.
        :type tangents: numpy.ndarray

        :return: A float value representing the smoothness of the tangent directions.
            A value close to 1.0 indicates high smoothness (low angular deviation),
            while values closer to 0 indicate low smoothness (high angular deviation).
        :rtype: float
        """
        if (
            not isinstance(tangents, np.ndarray)
            or tangents.ndim != 2
            or tangents.shape[0] < 2
        ):
            return 1.0
        else:
            magnitudes = np.linalg.norm(tangents, axis=1)
            non_zero_vectors = tangents[magnitudes > 0]

            # All vectors are approximately zero or there's only one non-zero vector
            if non_zero_vectors.shape[0] < 2:
                return 1.0

            angles = np.arccos(
                np.einsum("ij,ij->i", tangents[:-1], tangents[1:])
                / (
                    np.linalg.norm(tangents[:-1], axis=1)
                    * np.linalg.norm(tangents[1:], axis=1)
                )
            )
            smoothness = 1 - np.std(angles)
            return smoothness

    @staticmethod
    def normalized_length(points: np.ndarray, min_l: float, max_l: float):
        """
        Calculates the normalized length of a path defined by a sequence of points. The normalization
        is based on the minimum and maximum lengths provided. It utilizes the total_length function
        to compute the original length of the path. The returned normalized value lies between 0 and 1.

        :param points: A numpy array representing the sequence of points
            that defines the path.
        :param min_l: The minimum length to be used for normalization.
        :param max_l: The maximum length to be used for normalization.

        :return: Normalized length of the path as a float value.
        """
        length = total_length(points)
        return (length - min_l) / (max_l - min_l + 1e-16)

    def combined_smoothness(self, points: np.ndarray, min_l: float, max_l: float):
        """
        Computes the combined smoothness score for a series of points based on both
        angle-based smoothness and normalized length. The method integrates these
        two metrics into a single smoothness score by averaging their values.

        :param points: The array of points representing a series of coordinates.
                       It should have shape (n, m) where n is the number of points
                       and m is the dimensionality of each point.
        :param min_l: The minimum length used for normalization.
        :param max_l: The maximum length used for normalization.

        :return: The combined smoothness score as a float value.
        """
        tangents = np.diff(points, axis=0)

        scores = [
            self._angle_smoothness(tangents),
            self.normalized_length(points, min_l, max_l),
        ]

        return np.mean(scores)

    def __call__(self, segments: np.ndarray):
        """
        Analyze and compute the combined smoothness scores for segmented arrays.
        The method processes a segmented array and calculates smoothness scores
        for each segment based on its geometry relative to the minimum and maximum
        segment lengths. The lengths are determined by the first and last segments.

        :param segments: A 2D numpy.ndarray representing the segmented array.
                         The first column contains segment ids, and the
                         remaining columns represent spatial coordinates.
        :type segments: numpy.ndarray

        :return: A list of smoothness scores for each unique segment.
        :rtype: list[float]
        :raises ValueError: If the input array does not have the required shape
                            where the second dimension size must equal 4.
        """
        if segments.shape[1] != 4:
            TardisError(
                "145",
                "tardis_em.analysis.spatial_graph_utils.py",
                f"Not segmented array. Expected shape 4 got {segments.shape[1]}",
            )
        unique_ids = np.unique(segments[:, 0])
        min_l = total_length(segments[np.where(segments[:, 0] == 0)[0], 1:])
        max_l = total_length(
            segments[np.where(segments[:, 0] == segments[-1, 0])[0], 1:]
        )

        scores = []
        for i in unique_ids:
            points = segments[np.where(segments[:, 0] == i)[0], 1:]
            scores.append(self.combined_smoothness(points, min_l, max_l))

        return scores


def compare_splines_probability(
    spline_1: np.ndarray, spline_2: np.ndarray, threshold=100
):
    """
    Compares the similarity of two splines based on a distance threshold and returns a
    probability measure of their intersection. The comparison is performed by calculating
    the distances between points on the splines and determining the proportion of
    points on the first spline that match the second spline, within the specified threshold.

    :param spline_1: A numpy array representing the first spline.
    :param spline_2: A numpy array representing the second spline.
    :param threshold: A numeric value representing the distance threshold below which points on
                      the splines are considered matching. Default is 100.

    :return: A floating-point value representing the probability of intersection, i.e., the
             proportion of points on the first spline that match the second spline within
             the specified threshold.
    """
    if len(spline_1) == 0 or len(spline_2) == 0:
        return 0.0

    # Calculating distance matrix between points of 2 splines
    dist_matrix = cdist(spline_1, spline_2)

    # Check how many points on spline1 match spline2
    m_s1 = [1 if x < threshold else 0 for x in np.min(dist_matrix, axis=1)]

    # If no matching points probability is 0
    if sum(m_s1) == 0:
        return 0.0

    # Calculating intersection using % of the matching point below a threshold
    probability = sum(m_s1) / len(spline_1)

    return probability
