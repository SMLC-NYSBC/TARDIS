#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import numpy as np
from tardis_em.analysis.filament_utils import (
    resample_filament,
    sort_segments,
    reorder_segments_id,
    smooth_spline,
    sort_by_length,
    cut_150_degree,
)
from tardis_em.analysis.geometry_metrics import (
    curvature,
    curvature_list,
    tortuosity,
    tortuosity_list,
    total_length,
    length_list,
    angle_between_vectors,
)


def test_resample_filament():
    """Test resampling of filaments to match a given spacing size."""
    points = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 2, 0, 0],
        ]
    )
    spacing_size = 0.5

    result = resample_filament(points, spacing_size)

    # Verify the output length with expected spacing
    assert result.shape[0] > points.shape[0]
    # Verify that IDs are preserved
    assert np.all(result[:, 0] == 1)
    # Verify that spacing is correct
    distances = np.sqrt(np.sum(np.diff(result[:, 1:4], axis=0) ** 2, axis=1))
    assert np.allclose(distances, spacing_size, atol=0.1)


def test_sort_segments():
    """Test sorting of multiple groups of segments."""
    coords = np.array(
        [
            [2, 4, 4, 4],
            [2, 3, 3, 3],
            [1, 2, 2, 2],
            [1, 1, 1, 1],
        ]
    )
    result = sort_segments(coords)

    # Verify that sorting resipects grouping and order within groups
    assert result[0, 0] == 1
    assert result[1, 0] == 1
    assert np.all(result[:2, :] == np.array([[1, 2, 2, 2], [1, 1, 1, 1]]))
    assert np.all(result[2:, :] == np.array([[2, 4, 4, 4], [2, 3, 3, 3]]))


def test_reorder_segments_id():
    """Test reordering of segments by IDs."""
    coords = np.array(
        [
            [1, 0, 0, 0],
            [2, 1, 1, 1],
            [4, 2, 2, 2],
            [6, 2, 2, 2],
        ]
    )
    result = reorder_segments_id(coords)

    # Verify that IDs are contiguous and ordered
    assert np.array_equal(np.unique(result[:, 0]), [0, 1, 2, 3])


def test_smooth_spline():
    """Test smoothing of filament segments using splines."""
    points = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 3, 3, 3],
            [1, 4, 6, 5],
        ]
    )
    result = smooth_spline(points, s=0.5)

    # Assert that IDs are preserved
    assert np.all(result[:, 0] == 1)
    # Assert that the output is smoothed (check if it's not initial input)
    assert not np.allclose(result[:, 1:], points[:, 1:])


def test_sort_by_length():
    """Test sorting of segments by total length."""
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 10, 10, 10],
            [1, 0, 0, 0],
            [1, 5, 5, 5],
            [2, 0, 0, 0],
            [2, 25, 25, 25],
        ]
    )
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 5, 5, 5],
            [1, 0, 0, 0],
            [1, 10, 10, 10],
            [2, 0, 0, 0],
            [2, 25, 25, 25],
        ]
    )
    result = sort_by_length(coords)
    assert np.all(expected == result)

    """Test sorting of segments by total length."""
    coords = np.array(
        [
            [1, 0, 0, 0],
            [1, 10, 10, 10],
            [2, 0, 0, 0],
            [2, 5, 5, 5],
            [4, 0, 0, 0],
            [4, 25, 25, 25],
        ]
    )
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 5, 5, 5],
            [1, 0, 0, 0],
            [1, 10, 10, 10],
            [2, 0, 0, 0],
            [2, 25, 25, 25],
        ]
    )
    result = sort_by_length(coords)
    assert np.all(expected == result)


def test_cut_150_degree():
    """Test cutting segments where angles exceed 150 degrees."""
    segments = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 2, 0, 2],
            [2, 0, 0, 0],
            [2, 1, 0, 1],
            [2, 2, 0, 2],
        ]
    )
    loop, result = cut_150_degree(segments)

    # Verify that cuts are made
    assert loop is True
    # Verify resulting segment format
    assert result.shape[1] == 4
    assert len(np.unique(result[:, 0])) == len(np.unique(segments[:, 0]))


def test_curvature():
    """Test curvature calculation."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [2, 1, 1],
            [3, 0, 1],
        ]
    )
    result = curvature(points)
    assert result.ndim == 1  # Curvature should return array
    assert len(result) == len(points)  # Length matches points


def test_curvature_tortuosity():
    """Test curvature and tortuosity calculation."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [2, 1, 1],
            [3, 0, 1],
        ]
    )
    curvature_value, tortuosity_value = curvature(points, tortuosity_b=True)
    assert tortuosity_value >= 1  # Tortuosity must be >= 1
    assert len(curvature_value) == len(points)  # Curvature for each point


def test_curvature_list():
    """Test curvature list for multiple splines."""
    splines = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 0, 0, 0],
            [2, 1, 1, 1],
        ]
    )
    result = curvature_list(splines)
    assert len(result) == 2  # Two splines
    assert all(type(value) == float for value in result)


def test_tortuosity():
    """Test tortuosity calculation for a single spline."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 1],
        ]
    )
    tort = tortuosity(points)
    assert tort >= 1  # Tortuosity >= 1 for any non-trivial path


def test_tortuosity_list():
    """Test tortuosity list for multiple splines."""
    splines = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 0, 1, 0],
            [2, 1, 1, 0],
        ]
    )
    result = tortuosity_list(splines)
    assert len(result) == 2  # Should handle two splines
    assert all(t >= 1 for t in result)  # All tortuosities >= 1


def test_total_length():
    """Test total length of a curve."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [2, 2, 0],
        ]
    )
    length = total_length(points)
    assert np.isclose(length, np.sqrt(2) + np.sqrt(2))  # Expected length


def test_length_list():
    """Test length calculation of multiple splines."""
    splines = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 0, 0, 0],
            [2, 1, 1, 1],
        ]
    )
    result = length_list(splines)
    assert len(result) == 2  # Two splines
    assert result[0] > 0 and result[1] > 0


def test_angle_between_vectors():
    """Test angle between vectors."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    angle = angle_between_vectors(v1, v2)
    assert np.isclose(angle, 90.0)  # Orthogonal vectors
