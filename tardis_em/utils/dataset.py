#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import random
import shutil
from glob import glob
from os import listdir, mkdir
from os.path import isdir, join
from shutil import copyfile, copytree, move
from typing import Optional

from tardis_em.utils.errors import TardisError


def find_filtered_files(directory, prefix="instances_filter", format_="csv"):
    if prefix == "":
        extension = "*"
    else:
        extension = "*_"

    if not isinstance(format_, list) or not isinstance(format_, tuple):
        search_pattern = join(directory, f"{extension}{prefix}.{format_}")

        filtered_files = glob(search_pattern)
    else:
        filtered_files = []
        for i in format_:
            search_pattern = join(directory, f"{extension}{prefix}.{format_}")

            filtered_files += glob(search_pattern)

    return filtered_files


def move_train_dataset(
    dir_: str, coord_format: tuple, with_img: bool, img_format: Optional[tuple] = None
):
    """
    Detect and copy all date to new train directory.

    Train dataset builder. Detected files of specific format and moved to:
    - dir/train/masks
    - dir/train/imgs [optional]

    Args:
        dir_ (str): Directory where the file should be output.
        coord_format (tuple): Format of the coordinate files.
        with_img (bool): If True, expect corresponding image files.
        img_format (tuple, optional): Allowed format that can be used.
    """
    if coord_format == ".txt":
        area_list = [d for d in listdir(dir_) if isdir(join(dir_, d))]
        area_list = [f for f in area_list if not f.startswith((".", "train", "test"))]

        for i in area_list:
            copytree(join(dir_, i), join(dir_, "train", "masks", i))

    if not len([f for f in listdir(dir_) if f.endswith(coord_format)]) > 0:
        TardisError(
            "121",
            "tardis_em/utils/dataset.py",
            f"No coordinate file found in given dir {dir_}",
        )

    idx_coord = [f for f in listdir(dir_) if f.endswith(coord_format)]

    for i in idx_coord:
        copyfile(src=join(dir_, i), dst=join(dir_, "train", "masks", i))

    """Sort coord with images if included"""
    if with_img:
        if not len([f for f in listdir(dir_) if f.endswith(img_format)]) > 0:
            TardisError(
                "121",
                "tardis_em/utils/dataset.py",
                f"No image file found in given dir {dir_}",
            )

        idx_coord = [f for f in listdir(dir_) if f.endswith(img_format)]

        for i in idx_coord:
            copyfile(src=join(dir_, i), dst=join(dir_, "train", "imgs", i))


def build_test_dataset(dataset_dir: str, dataset_no: int, stanford=False):
    """
    Standard builder for test datasets.

    This module builds a test dataset from the training subset by moving random
    files from train to test directory.
    The number of files is specified in %.

    Files are saved in dir/test/imgs and dir/test/masks.

    Args:
        dataset_dir (str): Directory with train test folders.
        dataset_no (int): Number of datasets to iterate throw.
        stanford (bool): Marker for stanford S3DIS dataset
    """
    if "test" not in listdir(dataset_dir) and "train" not in listdir(dataset_dir):
        TardisError(
            "122",
            "tardis_em/utils/dataset.py",
            f"Could not find train or test folder in directory {dataset_dir}",
        )

    if stanford:
        mkdir(join(dataset_dir, "test", "masks", "Area_1"))
        mkdir(join(dataset_dir, "test", "masks", "Area_3"))
        mkdir(join(dataset_dir, "test", "masks", "Area_5"))

        move(
            join(dataset_dir, "train", "masks", "Area_1", "office_1"),
            join(dataset_dir, "test", "masks", "Area_1", "office_1"),
        )
        move(
            join(dataset_dir, "train", "masks", "Area_3", "WC_1"),
            join(dataset_dir, "test", "masks", "Area_3", "WC_1"),
        )
        move(
            join(dataset_dir, "train", "masks", "Area_5", "storage_1"),
            join(dataset_dir, "test", "masks", "Area_5", "storage_1"),
        )

    image_list = listdir(join(dataset_dir, "train", "imgs"))
    image_list = [i for i in image_list if i.endswith((".tif", ".mrc"))]

    if len(image_list) < 1000:
        # Get list of dataset
        images = []
        for i in range(dataset_no):
            df_imgs = [img for img in image_list if img.startswith(f"{i}")]
            images.append(df_imgs)

        # For each image select 20% random patches
        images = [
            i
            for id_, i in enumerate(images)
            if id_ in random.sample(range(0, len(images)), int(len(images) // 5))
        ]

        for i in images:
            list_move = []
            for j in random.sample(range(0, len(i) - 1), 4 if len(i) > 10 else 0):
                list_move.append(i[j])

            for j in list_move:
                shutil.move(
                    join(dataset_dir, "train", "imgs", j),
                    join(dataset_dir, "test", "imgs", j),
                )

                shutil.move(
                    join(dataset_dir, "train", "masks", j[:-4] + "_mask" + j[-4:]),
                    join(dataset_dir, "test", "masks", j[:-4] + "_mask" + j[-4:]),
                )
    else:
        if len(image_list) > 62500:
            images = random.sample(range(0, len(image_list)), 1000)
        else:
            images = random.sample(
                range(0, len(image_list)), int(len(image_list) // 25)
            )

        for i in images:
            j = image_list[i]

            shutil.move(
                join(dataset_dir, "train", "imgs", j),
                join(dataset_dir, "test", "imgs", j),
            )
            shutil.move(
                join(dataset_dir, "train", "masks", j[:-4] + "_mask" + j[-4:]),
                join(dataset_dir, "test", "masks", j[:-4] + "_mask" + j[-4:]),
            )
