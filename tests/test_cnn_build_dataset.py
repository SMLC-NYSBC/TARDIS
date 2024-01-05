#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from os import listdir, mkdir
from os.path import isdir, join
from shutil import copy, rmtree

import numpy as np

from tardis_em.cnn.datasets.build_dataset import (
    build_train_dataset,
)
from tardis_em.utils.load_data import load_image


def test_build_datasets():
    if isdir("./tests/test_data/temp"):
        rmtree("./tests/test_data/temp")

    mkdir("./tests/test_data/temp")
    mkdir("./tests/test_data/temp/train")
    mkdir("./tests/test_data/temp/train/imgs")
    mkdir("./tests/test_data/temp/train/masks")

    copy("./tests/test_data/data_type/am3D.am", "./tests/test_data/temp/am3D.am")
    copy(
        "./tests/test_data/data_type/am3D.CorrelationLines.am",
        "./tests/test_data/temp/am3D.CorrelationLines.am",
    )

    build_train_dataset(
        dataset_dir="./tests/test_data/temp/",
        circle_size=250,
        resize_pixel_size=25,
        trim_xy=64,
        trim_z=64,
    )

    assert len(listdir("./tests/test_data/temp/train/imgs")) > 0
    assert len(listdir("./tests/test_data/temp/train/masks")) > 0
    dir_img = listdir("./tests/test_data/temp/train/imgs")
    img, _ = load_image(join("./tests/test_data/temp/train/imgs", dir_img[5]))
    assert img.shape == (64, 64, 64)
    assert img.min() >= -1 and img.max() <= 1
    assert img.dtype == np.float32

    dir_mask = listdir("./tests/test_data/temp/train/masks")
    img, _ = load_image(join("./tests/test_data/temp/train/masks", dir_mask[5]))
    assert img.shape == (64, 64, 64)
    assert img.dtype == np.uint8

    rmtree("./tests/test_data/temp")
