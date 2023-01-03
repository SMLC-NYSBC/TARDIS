import glob
from os import listdir, mkdir, rename
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

from tardis.utils.errors import TardisError


def build_new_dir(dir: str):
    """
    Standard set-up for creating new directory for storing all data.

    Args:
        dir (str): Directory where folder will be build.
    """
    """ Build temp files directory """
    output = join(dir, "output")
    image_list = glob.glob(dir + '/*.tif')
    assert len(image_list) > 0, \
        TardisError('build_new_dir',
                    'tardis/utils/setup_envir.py',
                    'At least one .tif image has to be in the directory!')

    if not isdir(output):
        mkdir(output)
    else:
        print("Output directory already exist! Files moved to new output_old "
              "directory.")
        rename(output, join(dir, "output_old"))
        mkdir(output)


def build_temp_dir(dir: str):
    """
    Standard set-up for creating new temp dir for cnn prediction.

    Args:
        dir (str): Directory where folder will be build.
    """
    if isdir(join(dir, 'temp')):
        if isdir(join(dir, 'temp', 'Patches')) or isdir(join(dir, 'temp', 'Predictions')):
            clean_up(dir=dir)

            mkdir(join(dir, 'temp'))
            mkdir(join(dir, 'temp', 'Patches'))
            mkdir(join(dir, 'temp', 'Predictions'))
        else:
            mkdir(join(dir, 'temp', 'Patches'))
            mkdir(join(dir, 'temp', 'Predictions'))
    else:
        mkdir(join(dir, 'temp'))
        mkdir(join(dir, 'temp', 'Patches'))
        mkdir(join(dir, 'temp', 'Predictions'))

    if not isdir(join(dir, 'Predictions')):
        mkdir(join(dir, 'Predictions'))


def clean_up(dir: str):
    """
    Clean-up all temp files.

    Args:
        dir (str): Main directory where temp dir is located.
    """
    if isdir(join(dir, 'temp', 'Patches')):
        rmtree(join(dir, 'temp', 'Patches'))

    if isdir(join(dir, 'temp', 'Predictions')):
        rmtree(join(dir, 'temp', 'Predictions'))

    rmtree(join(dir, 'temp'))


def check_dir(dir: str,
              train_img: str,
              train_mask: str,
              test_img: str,
              test_mask: str,
              with_img: bool,
              img_format: Optional[tuple] = str,
              mask_format: Optional[tuple] = str):
    """
    Check list used to evaluate if directory containing dataset for CNN.

    Args:
        dir (str): Main directory with all files.
        train_img (str): Directory name with images for training.
        train_mask (str): Directory name with mask images for training.
        img_format (tuple, str): Allowed image format.
        test_img (str): Directory name with images for validation.
        test_mask (str): Directory name with mask images for validation.
        mask_format (tuple, str): Allowed mask image format.
        with_img (bool): GraphFormer bool value for training with/without images.
    """
    if isinstance(img_format, str):
        img_format = [img_format]
    if isinstance(mask_format, str):
        mask_format = [mask_format]

    dataset_test = False
    if isdir(join(dir, 'train')) and isdir(join(dir, 'test')):
        dataset_test = True

        if with_img:
            # Check if train img and coord exist and have same files
            if isdir(train_img) and isdir(train_mask):
                if len([f for f in listdir(train_img)
                        if f.endswith(img_format)]) == len([f for f in listdir(train_mask)
                                                            if f.endswith(mask_format)]):
                    if len([f for f in listdir(train_img)
                            if f.endswith(img_format)]) == 0:
                        dataset_test = False
                else:
                    dataset_test = False

            # Check if test img and mask exist and have same files
            if isdir(test_img) and isdir(test_mask):
                if len([f for f in listdir(test_img)
                        if f.endswith(img_format)]) == len([f for f in listdir(test_mask)
                                                            if f.endswith(mask_format)]):
                    if len([f for f in listdir(test_img)
                            if f.endswith(img_format)]) == 0:
                        dataset_test = False
                else:
                    dataset_test = False
        else:
            if isdir(train_img) and isdir(train_mask):
                if len([f for f in listdir(train_mask)
                        if f.endswith(mask_format)]) > 0:
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
