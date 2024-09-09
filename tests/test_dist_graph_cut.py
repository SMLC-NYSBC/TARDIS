# #####################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
# #####################################################################

from math import sqrt

import numpy as np

from tardis_em.utils.load_data import ImportDataFromAmira
from tardis_em.analysis.spatial_graph_utils import FilterSpatialGraph
from tardis_em.analysis.geometry_metrics import total_length, tortuosity
from tardis_em.analysis.filament_utils import sort_segment, reorder_segments_id


def test_reorder_segments():
    coord = ImportDataFromAmira("./tests/test_data/data_type/am3D.CorrelationLines.am")
    segments = coord.get_segmented_points() / coord.get_pixel_size()
    segments = segments.astype(np.int32)
    segments[np.where(segments[:, 0] == 2)[0], 0] = 5

    reorder_segment = reorder_segments_id(coord=segments, order_range=None)
    assert np.all(np.unique(reorder_segment[:, 0]) == [0, 1, 2])

    reorder_segment = reorder_segments_id(coord=segments, order_range=(3, 6))
    assert np.all(np.unique(reorder_segment[:, 0]) == [3, 4, 5])


def test_sort_segment():
    # Test case 1: Check if the function sorts the points in line order
    coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    expected_output = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    )

    assert np.array_equal(sort_segment(coord), expected_output)

    # Test case 2: Check if the function returns an array of the same shape as the input
    coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    assert sort_segment(coord).shape == coord.shape

    # Test case 3: Check if the function can handle an empty input
    coord = np.array([])
    assert sort_segment(coord).shape == (0,)


def test_total_length():
    coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected_length = 3 * sqrt(27)
    assert (
        total_length(coord) == expected_length
    ), f"For coord={coord}, expected length={expected_length} but got {total_length(coord)}"

    coord = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    expected_length = 2 * sqrt(3)
    assert (
        total_length(coord) == expected_length
    ), f"For coord={coord}, expected length={expected_length} but got {total_length(coord)}"

    coord = np.array([[1, 1, 1]])
    expected_length = 0
    assert (
        total_length(coord) == expected_length
    ), f"For coord={coord}, expected length={expected_length} but got {total_length(coord)}"


def test_tortuosity():
    coord = np.array([[1, 2, 1], [2, 5, 3], [4, 6, 6], [8, 9, 6]])
    expected_tortuosity = round(sqrt(1.268), 3)
    result = round(tortuosity(coord), 3)
    assert (
        expected_tortuosity == result
    ), f"For coord={coord}, expected tortuosity={expected_tortuosity} but got {result}"

    coord = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    expected_tortuosity = 1
    assert (
        tortuosity(coord) == expected_tortuosity
    ), f"For coord={coord}, expected tortuosity={expected_tortuosity} but got {tortuosity(coord)}"

    coord = np.array([[1, 1, 1]])
    expected_tortuosity = 1
    assert (
        tortuosity(coord) == expected_tortuosity
    ), f"For coord={coord}, expected tortuosity={expected_tortuosity} but got {tortuosity(coord)}"


def test_FilterWrapper():
    filter = FilterSpatialGraph(connect_seg_if_closer_then=0, filter_short_segments=0)
    coord = np.array(
        [
            [0, 1, 2, 1],
            [0, 2, 5, 3],
            [0, 4, 6, 6],
            [0, 8, 9, 6],
            [1, 12, 10, 8],
            [1, 15, 12, 9],
            [1, 18, 16, 12],
            [1, 22, 25, 15],
            [10, 102, 100, 80],
            [10, 150, 102, 90],
            [10, 180, 106, 102],
            [10, 220, 205, 105],
        ]
    )
    expect = np.array(
        [
            [0, 1, 2, 1],
            [0, 2, 5, 3],
            [1, 12, 10, 8],
            [1, 15, 12, 9],
            [1, 18, 16, 12],
            [1, 22, 25, 15],
            [2, 102, 100, 80],
            [2, 150, 102, 90],
            [2, 180, 106, 102],
        ]
    )
    filtered_coord = filter(coord)
    assert np.all(filtered_coord.flatten() == expect.flatten())
