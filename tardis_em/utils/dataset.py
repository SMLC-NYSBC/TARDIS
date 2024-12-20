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


def find_filtered_files(directory, prefix="instances_filter", format_b="csv"):
    """
    Finds and retrieves a list of files in the specified directory based on a given
    prefix and format(s). This function can search for files matching the prefix
    along with a single format or multiple formats, returning a list of filtered
    file paths.

    :param directory: The directory path in which the search will be performed.
    :type directory: str
    :param prefix: The prefix that filtered files should match.
        Defaults to 'instances_filter'.
    :type prefix: str, optional
    :param format_b: The format or formats (as string or a list/tuple of strings)
        to search for matching files. Defaults to 'csv'.
    :type format_b: str or list or tuple, optional

    :return: A list of file paths that match the specified prefix and format(s) in
        the directory.
    :rtype: list
    """
    if prefix == "":
        extension = "*"
    else:
        extension = "*_"

    if not isinstance(format_b, list) or not isinstance(format_b, tuple):
        search_pattern = join(directory, f"{extension}{prefix}.{format_b}")

        filtered_files = glob(search_pattern)
    else:
        filtered_files = []
        for i in format_b:
            search_pattern = join(directory, f"{extension}{prefix}.{format_b}")

            filtered_files += glob(search_pattern)

    return filtered_files


def move_train_dataset(
    dir_s: str, coord_format: tuple, with_img: bool, img_format: Optional[tuple] = None
):
    """
    Moves and organizes a training dataset by placing coordinate files and optionally
    image files into appropriate subdirectories.

    :param dir_s: The directory containing dataset files to be moved.
    :type dir_s: str
    :param coord_format: The file extension or format of coordinate files to be processed.
    :type coord_format: tuple
    :param with_img: A flag indicating whether to include associated image files during
        dataset organization.
    :type with_img: bool
    :param img_format: The file extension or format of image files to be processed if
        `with_img` is True. Optional.
    :type img_format: tuple, optional
    :return: None
    :rtype: None

    :raises TardisError: If no coordinate files matching `coord_format` are found in the
        given directory.
    :raises TardisError: If `with_img` is True but no image files matching `img_format`
        are found in the given directory.
    """
    if coord_format == ".txt":
        area_list = [d for d in listdir(dir_s) if isdir(join(dir_s, d))]
        area_list = [f for f in area_list if not f.startswith((".", "train", "test"))]

        for i in area_list:
            copytree(join(dir_s, i), join(dir_s, "train", "masks", i))

    if not len([f for f in listdir(dir_s) if f.endswith(coord_format)]) > 0:
        TardisError(
            "121",
            "tardis_em/utils/dataset.py",
            f"No coordinate file found in given dir {dir_s}",
        )

    idx_coord = [f for f in listdir(dir_s) if f.endswith(coord_format)]

    for i in idx_coord:
        copyfile(src=join(dir_s, i), dst=join(dir_s, "train", "masks", i))

    """Sort coord with images if included"""
    if with_img:
        if not len([f for f in listdir(dir_s) if f.endswith(img_format)]) > 0:
            TardisError(
                "121",
                "tardis_em/utils/dataset.py",
                f"No image file found in given dir {dir_s}",
            )

        idx_coord = [f for f in listdir(dir_s) if f.endswith(img_format)]

        for i in idx_coord:
            copyfile(src=join(dir_s, i), dst=join(dir_s, "train", "imgs", i))


def build_test_dataset(dataset_dir: str, dataset_no: int, stanford=False):
    """
    Builds a test dataset by reorganizing and moving files from the train dataset directory
    to a test dataset directory based on specific selection logic. This function handles
    dataset creation for both the general case and a special case for Stanford datasets.
    Images and corresponding masks are moved into a new test directory, ensuring the train
    directory is properly split into train and test datasets.

    :param dataset_dir: Path to the directory containing the dataset.
    :type dataset_dir: str
    :param dataset_no: Number of datasets or partitions to consider when splitting for test data.
    :type dataset_no: int
    :param stanford: If True, applies specific folder organization and file movement logic
                     for Stanford datasets. Defaults to False.
    :type stanford: bool

    :return: None
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
