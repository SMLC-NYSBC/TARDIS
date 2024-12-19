#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import glob
from os import listdir, mkdir, rename
from os.path import isdir, join
from shutil import rmtree
from typing import Union

from tardis_em.utils.errors import TardisError


def build_new_dir(dir_: str):
    """
    Builds a directory for temporary output files and ensures required files
    are present within the specified directory. If the required conditions
    are not met or the output directory already exists, appropriate actions
    are taken.

    :param dir_: Path to the base directory where the temporary directory will
                 be created.
    :type dir_: str
    :return: None
    """
    """ Build temp files directory """
    output = join(dir_, "output")
    image_list = glob.glob(dir_ + "/*.tif")
    if not len(image_list) > 0:
        TardisError(
            "12",
            "tardis_em/utils/setup_envir.py",
            "At least one .tif image has to be in the directory!",
        )

    if not isdir(output):
        mkdir(output)
    else:
        TardisError(
            "12",
            "tardis_em/utils/setup_envir.py",
            "Output directory already exist! Files moved to new output_old "
            "directory.",
        )

        rename(output, join(dir_, "output_old"))
        mkdir(output)


def build_temp_dir(dir_: str):
    """
    Creates necessary directories in the specified base directory to ensure the required
    structure for processing is in place. The function first checks for the existence of
    a "temp" subdirectory and its nested structure within the provided `dir_`. If the
    "temp" and respective subdirectories already exist, they will be reset to a clean
    state. If not, proper directories will be created as needed. Additionally, verifies
    and creates a top-level "Predictions" directory if it doesn't exist.

    :param dir_: The base directory where the directories need to be created or reset.
    :type dir_: str
    :return: None
    """
    if isdir(join(dir_, "temp")):
        if isdir(join(dir_, "temp", "Patches")) or isdir(
            join(dir_, "temp", "Predictions")
        ):
            clean_up(dir_=dir_)

            mkdir(join(dir_, "temp"))
            mkdir(join(dir_, "temp", "Patches"))
            mkdir(join(dir_, "temp", "Predictions"))
        else:
            mkdir(join(dir_, "temp", "Patches"))
            mkdir(join(dir_, "temp", "Predictions"))
    else:
        mkdir(join(dir_, "temp"))
        mkdir(join(dir_, "temp", "Patches"))
        mkdir(join(dir_, "temp", "Predictions"))

    if not isdir(join(dir_, "Predictions")):
        mkdir(join(dir_, "Predictions"))


def clean_up(dir_: str):
    """
    Removes a directory named "temp" and its subdirectories, including
    "Patches" and "Predictions", from the specified directory path. Ensures
    recursive deletion of all files and subdirectories inside the "temp"
    directory.

    :param dir_: The path to the parent directory containing the "temp"
        directory to be cleaned up. The directory path must be provided as a
        string.
    :return: None. The function performs cleanup operations without
        returning a value.
    """
    if isdir(join(dir_, "temp")):
        if isdir(join(dir_, "temp", "Patches")):
            rmtree(join(dir_, "temp", "Patches"))

        if isdir(join(dir_, "temp", "Predictions")):
            rmtree(join(dir_, "temp", "Predictions"))

        rmtree(join(dir_, "temp"))


def check_dir(
    dir_: str,
    train_img: str,
    train_mask: str,
    test_img: str,
    test_mask: str,
    with_img: bool,
    img_format: Union[tuple, str],
    mask_format: Union[tuple, str, None],
) -> bool:
    """
    Validates the structure and contents of a dataset directory to ensure proper
    organization for training and testing processes. This function checks for
    the existence of specific subdirectories (`train` and `test`), as well as for
    the presence and consistency of files (both images and masks) in the dataset.
    It ensures that the number of image files matches the number of mask files,
    and optionally validates file formats.

    :param dir_: Root directory of the dataset to validate.
    :type dir_: str
    :param train_img: Path to the subdirectory containing training images.
    :type train_img: str
    :param train_mask: Path to the subdirectory containing training masks.
    :type train_mask: str
    :param test_img: Path to the subdirectory containing test images.
    :type test_img: str
    :param test_mask: Path to the subdirectory containing test masks.
    :type test_mask: str
    :param with_img: Indicates whether the validation should check for the
        presence of both images and masks, or just masks alone.
    :type with_img: bool
    :param img_format: File extension(s) for image files. It can be either
        a tuple of extensions or a single extension as a string.
    :type img_format: Union[tuple, str]
    :param mask_format: File extension(s) for mask files. It can be a tuple
        of extensions, a single extension as a string, or None to disable
        mask format validation.
    :type mask_format: Union[tuple, str, None]
    :return: A boolean indicating whether the directory structure and
        contents meet the validation criteria.
    :rtype: bool
    """
    if mask_format is None:
        return True

    dataset_test = False
    if isdir(join(dir_, "train")) and isdir(join(dir_, "test")):
        dataset_test = True

        if with_img:
            # Check if train img and coord exist and have same files
            if isdir(train_img) and isdir(train_mask):
                if len(
                    [f for f in listdir(train_img) if f.endswith(img_format)]
                ) == len([f for f in listdir(train_mask) if f.endswith(mask_format)]):
                    if (
                        len([f for f in listdir(train_img) if f.endswith(img_format)])
                        == 0
                    ):
                        dataset_test = False
                else:
                    dataset_test = False

            # Check if test img and mask exist and have same files
            if isdir(test_img) and isdir(test_mask):
                if len([f for f in listdir(test_img) if f.endswith(img_format)]) == len(
                    [f for f in listdir(test_mask) if f.endswith(mask_format)]
                ):
                    if (
                        len([f for f in listdir(test_img) if f.endswith(img_format)])
                        == 0
                    ):
                        dataset_test = False
                else:
                    dataset_test = False
        else:
            if isdir(train_img) and isdir(train_mask):
                if len([f for f in listdir(train_mask) if f.endswith(mask_format)]) > 0:
                    pass
                else:
                    dataset_test = False
            else:
                dataset_test = False

            if isdir(test_img) and isdir(test_mask):
                if len([f for f in listdir(test_mask) if f.endswith(mask_format)]) > 0:
                    pass
                else:
                    dataset_test = False
            else:
                dataset_test = False
    return dataset_test
