# #####################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
# #####################################################################

from os.path import join
from typing import Tuple

import numpy as np

from tardis_em.dist_pytorch.datasets.augmentation import Crop2D3D, preprocess_data
from tardis_em.utils.normalization import RescaleNormalize, SimpleNormalize


def test_preprocess_data_general():
    # Test for csv file
    coord = join(
        "tests",
        "test_data",
        "data_loader",
        "filament_mem",
        "train",
        "masks",
        "20dec04c_Grey1_00028gr_00012sq_v02_00003hln_00003enn2-a-DW-contours-tb.csv",
    )
    image = join("tests", "test_data", "data_type", "tif2D.tif")

    size = 32
    include_label = True
    normalization = "simple"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 2

    coords, img = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)


def test_preprocess_data_npy():
    # Test for npy file
    coord = join("tests", "test_data", "data_type", "coord2D.npy")

    image = join("tests", "test_data", "data_type", "tif2D.tif")

    size = 32
    include_label = True
    normalization = "simple"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 2

    coords, img = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)


def test_preprocess_data_am():
    # Test for am file
    coord = join("tests", "test_data", "data_type", "am3D.CorrelationLines.am")

    image = join("tests", "test_data", "data_type", "am3D.am")

    size = 32
    include_label = True
    normalization = "simple"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 2

    coords, img = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)


def test_preprocess_data_normalization():
    # Test for normalization
    coord = join("tests", "test_data", "data_type", "am3D.CorrelationLines.am")
    image = join("tests", "test_data", "data_type", "tif3D.tif")

    size = 32
    include_label = True
    normalization = "minmax"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 2

    coords, img = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)


def test_preprocess_data_size():
    # Test for size
    coord = join("tests", "test_data", "data_type", "am3D.CorrelationLines.am")
    image = join("tests", "test_data", "data_type", "tif3D.tif")

    size = None
    include_label = True
    normalization = "minmax"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 2

    coords, img = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)


def test_preprocess_data_label():
    # Test for include label
    coord = join("tests", "test_data", "data_type", "am3D.CorrelationLines.am")
    image = join("tests", "test_data", "data_type", "tif3D.tif")

    size = None
    include_label = False
    normalization = "minmax"

    result = preprocess_data(coord, image, size, include_label, normalization)

    assert isinstance(result, Tuple)
    assert len(result) == 3

    coords, img, graph = result
    assert isinstance(coords, np.ndarray)
    assert isinstance(img, np.ndarray)
    assert isinstance(graph, np.ndarray)


def test_Normalize():
    norm = SimpleNormalize()
    rescale = RescaleNormalize(clip_range=(1, 99))

    # uint8
    x = np.random.randint(0, 255, size=(25, 25), dtype=np.uint8)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= 0
    assert result_rescale.max() <= 255
    assert np.any(x != result_rescale)

    # int8
    x = np.random.randint(-128, 127, size=(25, 25), dtype=np.int8)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= -128
    assert result_rescale.max() <= 127

    # uint16
    x = np.random.randint(0, 65535, size=(25, 25), dtype=np.uint16)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= 0
    assert result_rescale.max() <= 65535

    # int16
    x = np.random.randint(-32768, 32767, size=(25, 25), dtype=np.int16)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= -32768
    assert result_rescale.max() <= 32767

    # uint32
    x = np.random.randint(0, 4294967295, size=(25, 25), dtype=np.uint32)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= 0
    assert result_rescale.max() <= 4294967295

    # int16
    x = np.random.randint(-2147483648, 2147483647, size=(25, 25), dtype=np.int32)

    result_norm = norm(x)
    assert isinstance(result_norm, np.ndarray)
    assert result_norm.min() >= 0
    assert result_norm.max() <= 1

    result_rescale = rescale(x)
    assert isinstance(result_rescale, np.ndarray)
    assert result_rescale.min() >= -2147483648
    assert result_rescale.max() <= 2147483647


class TestCropImage:
    # create test image data
    test_image_2d = np.random.rand(10, 10)
    test_image_3d = np.random.rand(5, 10, 10)

    def test_init(self):
        crop_2d = Crop2D3D(self.test_image_2d, (5, 5), None)
        assert crop_2d.image.shape == (10, 10)
        assert crop_2d.size == (5, 5)
        assert crop_2d.normalization is None
        assert crop_2d.width == 10
        assert crop_2d.height == 10
        assert crop_2d.depth is None

        crop_3d = Crop2D3D(self.test_image_3d, (5, 5, 5), None)
        assert crop_3d.image.shape == (5, 10, 10)
        assert crop_3d.size == (5, 5, 5)
        assert crop_3d.normalization is None
        assert crop_3d.depth == 5
        assert crop_3d.width == 10
        assert crop_3d.height == 10

    def test_get_xyz_position(self):
        crop_2d = Crop2D3D(self.test_image_2d, (5, 5), None)
        assert crop_2d.get_xyz_position(5, 5, 5) == (0, 5)
        assert crop_2d.get_xyz_position(2, 5, 5) == (0, 5)
        assert crop_2d.get_xyz_position(0, 5, 5) == (0, 5)
        assert crop_2d.get_xyz_position(9, 5, 5) == (0, 5)

        crop_3d = Crop2D3D(self.test_image_3d, (5, 5, 5), None)
        assert crop_3d.get_xyz_position(2, 5, 5) == (0, 5)
        assert crop_3d.get_xyz_position(0, 5, 5) == (0, 5)
        assert crop_3d.get_xyz_position(4, 5, 5) == (0, 5)

    def test_call(self):
        crop_2d = Crop2D3D(self.test_image_2d, (5, 5), None)
        crop_2d_result = crop_2d((5, 5))
        assert crop_2d_result.shape == (5, 5)

        crop_3d = Crop2D3D(self.test_image_3d, (5, 5, 5), None)
        crop_3d_result = crop_3d((2, 5, 5))
        assert crop_3d_result.shape == (5, 5, 5)
