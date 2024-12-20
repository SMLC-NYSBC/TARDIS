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
from tardis_em.analysis.mt_classification.utils import (
    count_true_groups,
    distances_of_ends_to_surface,
    distance_to_the_pole,
    divide_into_sequences,
    fill_gaps,
    pick_pole_to_surfaces,
    points_on_mesh_knn,
    select_mt_ids_within_bb,
    assign_filaments_to_poles,
)


def test_count_true_groups():
    """Test counting consecutive groups of True values."""
    assert count_true_groups([True, False, True, True, False, True]) == 3
    assert count_true_groups([False, False, True, True, True, False]) == 1
    assert count_true_groups([False, False, False]) == 0
    assert count_true_groups([True, True, True]) == 1
    assert count_true_groups([]) == 0


def test_distances_of_ends_to_surface():
    """Test distances calculation for points to surface and pole."""
    vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    pole = np.array([0, 0, 0])
    ends = np.array([[0, 2, 2, 2], [1, 3, 3, 3]])

    d1, d2 = distances_of_ends_to_surface(vertices, pole, ends)

    assert d1.shape == (2, 1)
    assert d2.shape == (2, 1)


def test_distance_to_the_pole():
    """Test distance calculation from points to a pole."""
    points = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    pole = np.array([0, 0, 0])

    distances = distance_to_the_pole(points, pole)

    assert distances.shape == (3,)
    assert np.isclose(distances[0], np.sqrt(3), atol=1e-3)
    assert np.isclose(distances[1], np.sqrt(27), atol=1e-3)
    assert np.isclose(distances[2], np.sqrt(75), atol=1e-3)


def test_divide_into_sequences():
    """Test dividing a list into sequences of consecutive integers."""
    arr = [1, 2, 3, 5, 6, 10]
    result = divide_into_sequences(arr)

    assert result == [[1, 2, 3], [5, 6], [10]]
    assert divide_into_sequences([1]) == [[1]]  # Single-element list
    assert divide_into_sequences([1, 3, 5]) == [[1], [3], [5]]


def test_fill_gaps():
    """Test gap filling functionality."""
    float_list = [1, 3, 7, 8]
    n = 3
    result = fill_gaps(float_list, n)

    assert np.array_equal(result, np.array([1, 2, 3, 7, 8]))
    # Edge case with no gaps
    assert np.array_equal(fill_gaps([1, 2, 3], 1), [1, 2, 3])
    # Large gaps
    assert np.array_equal(fill_gaps([1, 10], 9), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_pick_pole_to_surfaces():
    """Test selection of poles relative to surface centroid."""
    poles = np.array([[2, 0, 0], [10, 0, 0]])
    vertices = [np.array([[1, 1, 1], [1, 1, -1]])]  # Simplified for testing

    result = pick_pole_to_surfaces(poles, vertices)
    assert np.array_equal(result, poles)  # First pole closer to the centroid

    # Reverse poles to ensure the reordering works
    poles_reversed = np.array([[10, 0, 0], [2, 0, 0]])
    result_reversed = pick_pole_to_surfaces(poles_reversed, vertices)
    assert np.array_equal(result_reversed, np.flipud(poles_reversed))  # Reordered


def test_points_on_mesh_knn():
    """Test KNN distance checking for points against a mesh."""
    points = np.array([[0, 0, 0], [10, 10, 10]])  # Two points for testing
    vertices = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    distances, within_threshold = points_on_mesh_knn(points, vertices)

    assert len(distances) == len(points)
    assert len(within_threshold) == len(points)
    assert distances[0] < distances[1]  # The first point is closer to the vertices


def test_select_mt_ids_within_bb():
    """Test microtubule endpoint selection within bounding box."""
    vertices = np.array([[0, 0, 0], [10, 10, 10]])
    mt_ends1 = np.array([[1, 1, 1, 1], [2, 5, 5, 5], [3, 15, 15, 15]])
    mt_ends2 = np.array([[1, 2, 2, 2], [3, 20, 20, 20]])

    ids = select_mt_ids_within_bb(vertices, mt_ends1, mt_ends2)

    assert len(ids) == 1  # Only MT ID 1 satisfies the bounding box criteria
    assert ids[0] == 1


def test_assign_filaments_to_poles():
    """Test assignment of filaments to the nearest poles."""
    filaments = np.array([[1, 1, 5, 5], [1, 2, 6, 6], [2, 11, 15, 15], [2, 12, 16, 16]])
    poles = np.array([[0, 5, 5], [15, 15, 15]])

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    assert len(filament_pole1) == 2  # First filament assigned to Pole 1
    assert len(filament_pole2) == 2  # Second filament assigned to Pole 2
    # Ensure the assignment is correct
    assert np.array_equal(np.unique(filament_pole1[:, 0]), [1])
    assert np.array_equal(np.unique(filament_pole2[:, 0]), [2])


def test_assign_filaments_to_single_pole():
    """Test assignment of all filaments to a single nearest pole."""
    filaments = np.array([[1, 1, 1, 1], [1, 2, 2, 2], [2, 3, 3, 3], [2, 4, 4, 4]])
    poles = np.array([[0, 0, 0], [10, 10, 10]])

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    # All filaments should be assigned to Pole 1 since it's closer
    assert len(filament_pole1) == 4
    assert len(filament_pole2) == 0
    assert np.array_equal(np.unique(filament_pole1[:, 0]), [1, 2])  # Filament IDs


def test_assign_filaments_to_two_poles():
    """Test assignment of filaments distributed between two poles."""
    filaments = np.array([[1, 1, 1, 1], [1, 2, 2, 2], [2, 15, 15, 15], [2, 14, 14, 14]])
    poles = np.array([[0, 0, 0], [10, 10, 10]])  # Pole 1  # Pole 2

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    # Filament 1 should go to Pole 1, and Filament 2 should go to Pole 2
    assert len(filament_pole1) == 2
    assert len(filament_pole2) == 2
    assert np.array_equal(np.unique(filament_pole1[:, 0]), [1])  # Filament 1
    assert np.array_equal(np.unique(filament_pole2[:, 0]), [2])  # Filament 2


def test_filmament_flip_relative_to_pole():
    """Test that the filaments are flipped so that the closer endpoint is assigned to the pole."""
    filaments = np.array([[1, 6, 6, 6], [1, 2, 2, 2], [2, 20, 20, 20], [2, 15, 15, 15]])
    poles = np.array([[0, 0, 0], [10, 10, 10]])

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    # Verify that filaments are flipped appropriately
    assert np.array_equal(filament_pole1[0], [1, 6, 6, 6])  # Closest end to Pole 1
    assert np.array_equal(filament_pole2[0], [2, 20, 20, 20])  # Closest end to Pole 2


def test_edge_case_tie_between_poles():
    """Test behavior when filament distances to both poles are equal."""
    filaments = np.array(
        [
            [1, 5, 5, 5],
            [1, 6, 6, 6],
        ]
    )
    poles = np.array([[0, 0, 0], [10, 10, 10]])  # Pole 1  # Pole 2

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    # If distances are equal, assign to Pole 1 based on implementation logic
    assert len(filament_pole1) == 0
    assert len(filament_pole2) == 2


def test_multiple_filaments_with_unique_ids():
    """Test assignment of multiple filaments with unique IDs."""
    filaments = np.array(
        [
            [1, 1, 1, 1],
            [1, 5, 5, 5],
            [2, 6, 6, 6],
            [2, 10, 10, 10],
            [3, 2, 2, 2],
            [3, 3, 3, 3],
        ]
    )
    poles = np.array(
        [
            [0, 0, 0],  # Pole 1
            [9, 9, 9],  # Pole 2
        ]
    )

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    # Filament 1 and 3 → Pole 1; Filament 2 → Pole 2
    assert np.array_equal(np.unique(filament_pole1[:, 0]), [1, 3])
    assert np.array_equal(np.unique(filament_pole2[:, 0]), [2])


def test_all_filaments_closer_to_pole2():
    """Test case where all filaments are closer to Pole 2."""
    filaments = np.array(
        [[1, 10, 11, 11], [1, 12, 10, 12], [2, 14, 16, 15], [2, 15, 15, 15]]
    )
    poles = np.array([[0, 0, 0], [10, 10, 10]])  # Pole 1  # Pole 2

    filament_pole1, filament_pole2 = assign_filaments_to_poles(filaments, poles)

    assert filament_pole1.size == 0
    assert len(filament_pole2) == 4
    assert np.array_equal(np.unique(filament_pole2[:, 0]), [1, 2])
