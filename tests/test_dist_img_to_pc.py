import numpy as np
import pytest

from tardis.dist_pytorch.utils.build_point_cloud import BuildPointCloud


def test_BuildPointCloud():
    """
    Test the BuildPointCloud class.
    """
    # Test check_data method
    image = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]],
                     dtype=np.uint8)
    assert (BuildPointCloud.check_data(image) == image).all()

    # Check if binary
    image = np.array([[0, 0, 0],
                      [0, 2, 0],
                      [0, 0, 0]], dtype=np.uint8)
    with pytest.raises(AssertionError):
        BuildPointCloud.check_data(image)

    # Test build_point_cloud method
    image = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.uint8)

    point_cloud = BuildPointCloud().build_point_cloud(image)
    expected_point_cloud = np.array([[0, 0, 0],
                                     [1, 1, 0],
                                     [2, 2, 0],
                                     [3, 3, 0]])
    assert (point_cloud == expected_point_cloud).all()

    with pytest.raises(AssertionError):
        BuildPointCloud().build_point_cloud(np.array([[0, 0, 0],
                                                      [0, 2, 0],
                                                      [0, 0, 0]],
                                                     dtype=np.uint8))
