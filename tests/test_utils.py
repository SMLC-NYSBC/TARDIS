#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import os

import numpy as np
import pytest
import torch

from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.export_data import NumpyToAmira
from tardis_em.utils.load_data import (
    load_am,
    load_tiff,
    ImportDataFromAmira,
    load_mrc_file,
)
from tardis_em.utils.logo import TardisLogo
from tardis_em.analysis.filament_utils import sort_segment, cut_150_degree
from tardis_em.analysis.spatial_graph_utils import compare_splines_probability
from tardis_em.analysis.geometry_metrics import (
    angle_between_vectors,
    tortuosity,
    total_length,
)
from tardis_em.utils.utils import EarlyStopping


# TEST device.py
def test_check_device():
    dev = get_device("cpu")
    assert dev == torch.device("cpu")

    dev = get_device("1")

    if torch.cuda.is_available():
        assert dev == torch.device(type="cuda", index=1)
    else:
        assert dev == torch.device("cpu")


# TEST error.py and logo.py
def test_logo():
    logo = TardisLogo()
    # Test short
    logo(title="Test_pytest")

    # Test long
    logo(
        title="Test_pytest",
        text_1="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        + "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )


def test_error():
    TardisError(id_="20", py="tests/test_utils.py", desc="PyTest Failed!")


# TEST export fiels
def test_am_single_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    exporter = NumpyToAmira()
    exporter.export_amira(coords=df, file_dir="./test.am")

    assert os.path.isfile("./test.am")
    os.remove("./test.am")


def test_am_multi_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    df_1 = np.array(df)
    df_2 = np.array(df_1)

    exporter = NumpyToAmira()
    exporter.export_amira(coords=(df_1, df_2), file_dir="./test.am")

    assert os.path.isfile("./test.am")
    os.remove("./test.am")


def test_am_label_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    df_1 = np.array(df)
    df_2 = np.array(df_1)

    exporter = NumpyToAmira()
    exporter.export_amira(
        coords=(df_1, df_2), file_dir="./test.am", labels=["test1", "test2"]
    )

    assert os.path.isfile("./test.am")
    os.remove("./test.am")


# TEST file readers
def test_tif():
    tif, px = load_tiff(tiff="./tests/test_data/data_type/tif2D.tif")
    assert tif.shape == (64, 32)

    tif, px = load_tiff(tiff="./tests/test_data/data_type/tif3D.tif")
    assert tif.shape == (78, 64, 32)


def test_rec_mrc():
    mrc, px = load_mrc_file(mrc="./tests/test_data/data_type/mrc2D.mrc")
    assert mrc.shape == (64, 32)
    assert px == 23.2

    rec, px = load_mrc_file(mrc="./tests/test_data/data_type/rec2D.rec")
    assert rec.shape == (64, 32)
    assert px == 23.2

    mrc, px = load_mrc_file(mrc="./tests/test_data/data_type/mrc3D.mrc")
    assert mrc.shape == (32, 64, 78)
    assert px == 23.2

    rec, px = load_mrc_file(mrc="./tests/test_data/data_type/rec3D.rec")
    assert rec.shape == (32, 64, 78)
    assert px == 23.2


def test_am():
    am, px, ps, trans = load_am(am_file="./tests/test_data/data_type/am2D.am")

    assert am.shape == (64, 32)
    assert am.dtype == np.uint8
    assert px == 23.2
    assert np.all(trans == np.array((0, 0, 4640)))

    am, px, ps, trans = load_am(am_file="./tests/test_data/data_type/am3D.am")

    assert am.shape == (8, 256, 256)
    assert am.dtype == np.uint8
    assert px == 92.8


def test_am_sg():
    am = ImportDataFromAmira(
        src_am="./tests/test_data/data_type/am3D.CorrelationLines.am",
        src_img="./tests/test_data/data_type/am3D.am",
    )
    segments = am.get_segmented_points()
    assert segments.shape == (10, 4)

    point = am.get_points()
    assert point.shape == (10, 3)

    image, px = am.get_image()
    assert image.shape == (8, 256, 256)
    assert image.dtype == np.uint8

    px = am.get_pixel_size()
    assert px == 92.8


# TEST utils.py
def test_early_stop():
    er_stop = EarlyStopping()
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 0

    er_stop(val_loss=0.09)
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 1

    er_stop = EarlyStopping()
    assert er_stop.counter == 0

    er_stop(f1_score=0.1)
    assert er_stop.counter == 0

    er_stop(f1_score=0.15)
    assert er_stop.counter == 0

    er_stop(f1_score=0.1)
    assert er_stop.counter == 1


def test_compare_splines_probability():
    # Test with matching splines
    spline_tardis = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    spline_amira = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    threshold = 1
    assert (
        round(compare_splines_probability(spline_tardis, spline_amira, threshold), 2)
        == 0.67
    )

    # Test with non-matching splines
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([[4, 4], [5, 5], [6, 6]])
    threshold = 1
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 0.0

    # Test with matching splines and threshold set too high
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([[1, 1], [2, 2], [3, 3]])
    threshold = 10
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 1.0

    # Test with matching splines and threshold set too low
    spline_tardis = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
    )
    spline_amira = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    threshold = 1
    assert (
        round(compare_splines_probability(spline_tardis, spline_amira, threshold), 2)
        == 0.33
    )

    # Test with matching splines and threshold set too low
    spline_tardis = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]])
    spline_amira = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
    )
    threshold = 1
    assert (
        round(compare_splines_probability(spline_tardis, spline_amira, threshold), 2)
        == 1.0
    )

    # Test with empty spline
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([])
    threshold = 1
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 0.0


# TEST spatial_graph_utils.py
def test_sort_segment():
    # Define an example point cloud with unsorted points
    unsorted_points = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4], [3, 4, 5]])

    # Call the sort_segment function
    sorted_points = sort_segment(unsorted_points)

    # Define what the expected output should be for the given input
    expected_output = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

    # Assert that the output from the function matches the expected output
    np.testing.assert_array_equal(sorted_points, expected_output)


def test_tortuosity():
    # Define an example set of coordinates representing a straight line
    straight_line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    # Tortuosity of a straight line should be 1
    assert tortuosity(straight_line) == pytest.approx(1.0, abs=1e-6)

    # Define an example set of coordinates representing a curved path
    curved_path = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
    # Tortuosity of a curved path should be greater than 1
    assert tortuosity(curved_path) > 1.0

    # Test with only one point (should return 1.0)
    one_point = np.array([[0, 0, 0]])
    assert tortuosity(one_point) == 1.0

    # Test with no points (should return 1.0)
    no_points = np.array([])
    assert tortuosity(no_points) == 1.0


def test_total_length():
    # Define an example set of coordinates representing a straight line
    straight_line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    # Total length of a straight line should be equal to the distance between the first and last point
    assert total_length(straight_line) == pytest.approx(2.0, abs=1e-6)

    # Define an example set of coordinates representing a curved path
    curved_path = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
    # Total length of a curved path should be greater than the distance between the first and last point
    assert total_length(curved_path) > 2.0

    # Test with only one point (should return 0.0)
    one_point = np.array([[0, 0, 0]])
    assert total_length(one_point) == 0.0


def test_angle_between_vectors():
    # Test with perpendicular vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert angle_between_vectors(v1, v2) == 90.0

    # Test with parallel vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([2, 0, 0])
    assert angle_between_vectors(v1, v2) == 0.0

    # Test with anti-parallel vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    assert angle_between_vectors(v1, v2) == 180.0

    # Test with vectors at 45 degrees
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 1, 0])
    assert angle_between_vectors(v1, v2) == pytest.approx(45.0, abs=1e-6)

    # Test with negative values
    v1 = np.array([-1, 0, 0])
    v2 = np.array([0, -1, 0])
    assert angle_between_vectors(v1, v2) == 90.0


def test_cut_150_degree():
    # Define a segment with a sharp angle that should be cut
    segments = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 1, 0],
        ]
    )

    was_cut, cut_segments = cut_150_degree(segments)
    assert was_cut
    assert cut_segments.shape == (4, 4)
    assert np.all(cut_segments[0, :] == [0, 0, 0, 0])
    assert np.all(cut_segments[1, :] == [0, 1, 0, 0])
    assert np.all(cut_segments[2, :] == [1, 1, 1, 0])
    assert np.all(cut_segments[3, :] == [1, 2, 1, 0])

    # Define a segment with all angles greater than 150 degrees
    segments = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 3, 0, 0],
        ]
    )

    was_cut, cut_segments = cut_150_degree(segments)
    assert not was_cut
    assert cut_segments.shape == segments.shape
    assert np.all(cut_segments == segments)
