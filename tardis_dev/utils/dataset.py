from os import listdir
from os.path import join
from shutil import move
from typing import Optional

import numpy as np


def move_train_dataset(dir: str,
                       coord_format: tuple,
                       with_img: bool,
                       img_format: Optional[tuple] = None) -> list:
    """
    Standard builder for train datasets.

    Train dataset builder. Detected files of specific format and moved to:
    - dir/train/masks
    - dir/train/imgs [optional]

    Args:
        dir (str): Directory where the file should outputted.
        coord_format (tuple): Format of the coordinate files.
        with_img (bool): If True, expect corresponding image files.
        img_format (tuple, optional): Allowed format that can be used.
    """
    assert len([f for f in listdir(dir) if f.endswith(coord_format)]) > 0, \
        f'No file found in given dir {dir}'

    idx_coord = [f for f in listdir(dir) if f.endswith(coord_format)]

    for i in idx_coord:
        move(src=join(dir, i),
             dst=join(dir, 'train', 'masks', i))

    """Sort coord with images if included"""
    if with_img:
        assert len([f for f in listdir(dir) if f.endswith(img_format)]) > 0, \
            f'No file found in given dir {dir}'

        idx_coord = [f for f in listdir(dir) if f.endswith(img_format)]

        for i in idx_coord:
            move(src=join(dir, i),
                 dst=join(dir, 'train', 'imgs', i))


def build_test_dataset(dataset_dir: str,
                       train_test_ration: int,
                       prefix: str):
    """
    Standard builder for test datasets.

    This module building a test dataset from training subset, by moving random
    files from train to test directory.
    Number of files is specified in %.

    Files are saved in dir/test/imgs and dir/test/masks.

    Args:
        dataset_dir (str): Directory with train test folders.
        train_test_ration (int): Percentage of dataset to be moved.
        prefix (str): Additional prefix name at the end of the file.
    """
    dataset = dataset_dir
    prefix = prefix

    assert 'test' in listdir(dataset_dir) and 'train' in listdir(dataset_dir), \
        f'Could not find train or test folder in directory {dataset_dir}'

    image_list = listdir(join(dataset_dir, 'train', 'imgs'))
    image_list.sort()
    mask_list = listdir(join(dataset_dir, 'train', 'masks'))
    mask_list.sort()

    train_test_ratio = (len(mask_list) * train_test_ration) // 100
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