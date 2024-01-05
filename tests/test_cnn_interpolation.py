# #####################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
# #####################################################################

import numpy as np

from tardis_em.cnn.data_processing.interpolation import interpolation


def test_interpolation_spline_x():
    points = np.array([[1, 2, 3], [4, 2, 3]])
    expected_result = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]])
    assert np.array_equal(interpolation(points), expected_result)


def test_interpolation_spline_y():
    points = np.array([[1, 2, 3], [1, 4, 3]])
    expected_result = np.array([[1, 2, 3], [1, 3, 3], [1, 4, 3]])
    assert np.array_equal(interpolation(points), expected_result)


def test_interpolation_spline_z():
    points = np.array([[1, 2, 3], [1, 2, 4]])
    expected_result = np.array([[1, 2, 3], [1, 2, 4]])
    assert np.array_equal(interpolation(points), expected_result)
