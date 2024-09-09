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
    Standard set-up for creating new directory for storing all data.

    Args:
        dir_ (str): Directory where folder will be build.
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
    Standard set-up for creating new temp dir for cnn prediction.

    Args:
        dir_ (str): Directory where folder will be build.
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
    Clean-up all temp files.

    Args:
        dir_ (str): Main directory where temp dir is located.
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
    Check the list used to evaluate if directory containing dataset for CNN.

    Args:
        dir_ (str): Main directory with all files.
        train_img (str): Directory name with images for training.
        train_mask (str): Directory name with mask images for training.
        img_format (tuple, str): Allowed image format.
        test_img (str): Directory name with images for validation.
        test_mask (str): Directory name with mask images for validation.
        mask_format (tuple, str): Allowed mask image format.
        with_img (bool): GraphFormer bool value for training with/without images.

    Returns:
        bool: Bool value indicating detection of the correct structure dataset
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
