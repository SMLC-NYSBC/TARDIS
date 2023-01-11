#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import numpy as np
from scipy.spatial.distance import cdist


def compare_splines_probability(spline_tardis: np.ndarray,
                                spline_amira: np.ndarray,
                                threshold=100):
    """
    Compare two splines and calculate probability of how likely given two
    splines are the same line given array of points for same or similar splines
    with no matching coordinates of points.

    Calculates the probability of two splines being similar by comparing
    the distance between their points and taking the mean of the matching
    points below a threshold.

    Parameters:
        spline_tardis (np.ndarray): The first spline to compare, represented
        as an array of points.
        spline_amira (np.ndarray): The second spline to compare, represented
        as an array of points.
        threshold (int): The maximum distance between points for them to be
        considered matching.

    Returns:
        float: The probability of the splines being similar, ranging from 0.0 to 1.0.
    """

    # Calculating distance matrix between points of 2 splines
    dist_matrix = cdist(spline_tardis, spline_amira)

    # Calculating the matching point from both splines
    matching_points = np.min(dist_matrix, axis=1)

    # Filtering out distance below threshold
    matching_points = matching_points[matching_points < threshold]

    # If no matching points probability is 0
    if len(matching_points) == 0:
        return 0.0

    # Calculating probability using mean of the matching point below threshold
    probability = np.mean(matching_points) / threshold

    return probability
