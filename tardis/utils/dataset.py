"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> Utils - dataset

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2023
"""
from os import listdir
from os.path import join
from shutil import move
from typing import Optional

import numpy as np

from tardis.utils.errors import TardisError


def move_train_dataset(dir: str,
                       coord_format: tuple,
                       with_img: bool,
                       img_format: Optional[tuple] = None):
    """
    Standard builder for train datasets.

    Train dataset builder. Detected files of specific format and moved to:
    - dir/train/masks
    - dir/train/imgs [optional]

    Args:
        dir (str): Directory where the file should be output.
        coord_format (tuple): Format of the coordinate files.
        with_img (bool): If True, expect corresponding image files.
        img_format (tuple, optional): Allowed format that can be used.
    """
    assert len([f for f in listdir(dir) if f.endswith(coord_format)]) > 0, \
        TardisError('121',
                    'tardis/utils/dataset.py',
                    f'No coordinate file found in given dir {dir}')

    idx_coord = [f for f in listdir(dir) if f.endswith(coord_format)]

    for i in idx_coord:
        move(src=join(dir, i),
             dst=join(dir, 'train', 'masks', i))

    """Sort coord with images if included"""
    if with_img:
        assert len([f for f in listdir(dir) if f.endswith(img_format)]) > 0, \
            TardisError('121',
                        'tardis/utils/dataset.py',
                        f'No image file found in given dir {dir}')

        idx_coord = [f for f in listdir(dir) if f.endswith(img_format)]

        for i in idx_coord:
            move(src=join(dir, i),
                 dst=join(dir, 'train', 'imgs', i))


def build_test_dataset(dataset_dir: str,
                       train_test_ration: float):
    """
    Standard builder for test datasets.

    This module building a test dataset from training subset, by moving random
    files from train to test directory.
    Number of files is specified in %.

    Files are saved in dir/test/imgs and dir/test/masks.

    Args:
        dataset_dir (str): Directory with train test folders.
        train_test_ration (int): Percentage of dataset to be moved.
    """
    dataset = dataset_dir

    assert 'test' in listdir(dataset_dir) and 'train' in listdir(dataset_dir), \
        TardisError('122',
                    'tardis/utils/dataset.py',
                    f'Could not find train or test folder in directory {dataset_dir}')

    image_list = sorted(listdir(join(dataset_dir, 'train', 'imgs')))
    mask_list = listdir(join(dataset_dir, 'train', 'masks'))
    mask_list.sort()

    train_test_ratio = (len(mask_list) * train_test_ration) // 1
    train_test_ratio = int(train_test_ratio)

    if train_test_ratio == 0:
        train_test_ratio = 1

    test_idx = []
    if len(image_list) == 0:
        data_no = len(mask_list)
    else:
        data_no = len(image_list)

    for _ in range(train_test_ratio):
        random_idx = np.random.choice(data_no)

        while random_idx in test_idx:
            random_idx = np.random.choice(data_no)

        test_idx.append(random_idx)

    if len(image_list) != 0:
        test_image_idx = list(np.array(image_list)[test_idx])
        test_mask_idx = list(np.array(mask_list)[test_idx])
    else:
        test_image_idx = []
        test_mask_idx = list(np.array(mask_list)[test_idx])

    for i in range(len(test_idx)):
        if len(image_list) != 0:
            # Move image file to test dir
            move(join(dataset, 'train', 'imgs', test_image_idx[i]),
                 join(dataset, 'test', 'imgs', test_image_idx[i]))
            move(join(dataset, 'train', 'masks', test_mask_idx[i]),
                 join(dataset, 'test', 'masks', test_mask_idx[i]))
        elif len(image_list) == 0 and len(mask_list) != 0:
            # Move mask file to test dir
            move(join(dataset, 'train', 'masks', test_mask_idx[i]),
                 join(dataset, 'test', 'masks', test_mask_idx[i]))
