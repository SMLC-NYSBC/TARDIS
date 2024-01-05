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
from os.path import isdir
from shutil import rmtree

import numpy as np

from tardis_em.cnn.data_processing.stitch import StitchImages
from tardis_em.cnn.data_processing.trim import trim_with_stride


def test_trim_stitch_3d():
    stitch = StitchImages()

    if isdir("./tests/test_data/temp"):
        rmtree("./tests/test_data/temp")
    mkdir("./tests/test_data/temp")

    image_3d_15 = np.zeros((15, 15, 15), dtype=np.float32)
    mask_3d_15 = np.zeros((15, 15, 15), dtype=np.uint8)
    image_3d_150 = np.zeros((150, 150, 150), dtype=np.float32)
    mask_3d_150 = np.zeros((150, 150, 150), dtype=np.uint8)

    trim_with_stride(
        image=image_3d_15,
        trim_size_xy=15,
        trim_size_z=15,
        output="./tests/test_data/temp",
        image_counter=0,
        scale=(15, 15, 15),
        clean_empty=False,
        mask=mask_3d_15,
    )
    assert len(listdir("./tests/test_data/temp/imgs")) == 1
    assert len(listdir("./tests/test_data/temp/masks")) == 1

    stitch_3d_15 = stitch(
        image_dir="./tests/test_data/temp/imgs", mask=False, dtype=np.float32
    )
    assert stitch_3d_15.shape == (15, 15, 15)

    stitch_3d_15 = stitch(
        image_dir="./tests/test_data/temp/masks",
        mask=True,
        prefix="_mask",
        dtype=np.uint8,
    )
    assert stitch_3d_15.shape == (15, 15, 15)

    rmtree("./tests/test_data/temp")
    mkdir("./tests/test_data/temp")

    trim_with_stride(
        image=image_3d_150,
        trim_size_xy=64,
        trim_size_z=64,
        output="./tests/test_data/temp",
        image_counter=0,
        stride=0,
        scale=(150, 150, 150),
        clean_empty=False,
        mask=mask_3d_150,
    )
    assert len(listdir("./tests/test_data/temp/imgs")) == 27
    assert len(listdir("./tests/test_data/temp/masks")) == 27

    stitch_3d_15 = stitch(
        image_dir="./tests/test_data/temp/imgs", mask=False, dtype=np.float32
    )
    assert stitch_3d_15.shape == (192, 192, 192)

    stitch_3d_15 = stitch(
        image_dir="./tests/test_data/temp/masks",
        mask=True,
        prefix="_mask",
        dtype=np.uint8,
    )
    assert stitch_3d_15.shape == (192, 192, 192)

    rmtree("./tests/test_data/temp")


def test_trim_stitch_2d():
    stitch = StitchImages()

    if isdir("./tests/test_data/temp"):
        rmtree("./tests/test_data/temp")
    mkdir("./tests/test_data/temp")

    image_2d_15 = np.zeros((15, 15), dtype=np.float32)
    mask_2d_15 = np.zeros((15, 15), dtype=np.uint8)
    image_2d_150 = np.zeros((150, 150), dtype=np.float32)
    mask_2d_150 = np.zeros((150, 150), dtype=np.uint8)

    trim_with_stride(
        image=image_2d_15,
        trim_size_xy=15,
        trim_size_z=15,
        output="./tests/test_data/temp",
        image_counter=0,
        scale=(15, 15),
        clean_empty=False,
        mask=mask_2d_15,
    )
    assert len(listdir("./tests/test_data/temp/imgs")) == 1
    assert len(listdir("./tests/test_data/temp/masks")) == 1

    stitch_2d_15 = stitch(
        image_dir="./tests/test_data/temp/imgs", mask=False, dtype=np.float32
    )
    assert stitch_2d_15.shape == (15, 15)

    stitch_2d_15 = stitch(
        image_dir="./tests/test_data/temp/masks",
        mask=True,
        prefix="_mask",
        dtype=np.uint8,
    )
    assert stitch_2d_15.shape == (15, 15)

    rmtree("./tests/test_data/temp")
    mkdir("./tests/test_data/temp")

    trim_with_stride(
        image=image_2d_150,
        trim_size_xy=64,
        trim_size_z=64,
        output="./tests/test_data/temp",
        image_counter=0,
        stride=0,
        scale=(150, 150),
        clean_empty=False,
        mask=mask_2d_150,
    )
    assert len(listdir("./tests/test_data/temp/imgs")) == 9
    assert len(listdir("./tests/test_data/temp/masks")) == 9

    stitch_2d_15 = stitch(
        image_dir="./tests/test_data/temp/imgs", mask=False, dtype=np.float32
    )
    assert stitch_2d_15.shape == (192, 192)
    stitch_2d_15 = stitch(
        image_dir="./tests/test_data/temp/masks",
        mask=True,
        prefix="_mask",
        dtype=np.uint8,
    )
    assert stitch_2d_15.shape == (192, 192)

    rmtree("./tests/test_data/temp")
