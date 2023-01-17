#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from tardis.dist_pytorch.utils.segment_point_cloud import sort_segment


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
