from os import listdir
from os.path import isdir, join
from shutil import move
from typing import Optional

import numpy as np


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Args:
        patience (int): how many epochs to wait before stopping when loss is
               not improving.
        min_delta (int): minimum difference between new loss and old loss for
               new loss to be considered as an improvement.
    """

    def __init__(self,
                 patience=10,
                 min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self,
                 val_loss: Optional[float] = None,
                 f1_score: Optional[float] = None):
        assert val_loss is not None or f1_score is not None, \
            'Validation loss or F1 score is missing!'

        if val_loss is not None:
            if self.best_loss is None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                self.counter = 0  # Reset counter if validation loss improves
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1

                if self.counter >= self.patience:
                    print('INFO: Early stopping')
                    self.early_stop = True
        elif f1_score is not None:
            if self.best_loss is None:
                self.best_loss = f1_score
            elif self.best_loss - f1_score < self.min_delta:
                self.best_loss = f1_score
                self.counter = 0  # Reset counter if validation loss improves
            elif self.best_loss - f1_score > self.min_delta:
                self.counter += 1

                if self.counter >= self.patience:
                    print('INFO: Early stopping')
                    self.early_stop = True


def check_uint8(image: np.ndarray):
    """
    Simple check for uint8 array.

    If array is not compliant, then data are examine for a type of data (binary
    mask, image with int8 etc.) and converted back to uint8.

    Args:
        image (np.ndarray): Image for evaluation.
    """
    if np.all(np.unique(image) == [0, 1]):
        # Binary mask image
        return image
    elif np.all(np.unique(image) == [0, 254]):
        # Multi-label mask
        return np.array(np.where(image > 1, 1, 0), dtype=np.uint8)
    elif len(np.unique(image) > 2):
        # Raw image data
        return image.astype(np.uint8)
    elif np.all(image == 0):
        # Image is empty
        return image


def check_dir(dir: str,
              train_img: str,
              train_mask: str,
              img_format: tuple,
              test_img: str,
              test_mask: str,
              mask_format: tuple,
              with_img: bool):
    """
    Check list used to evaluate if directory containing dataset for CNN.

    Args:
        dir (str): Main directory with all files.
        train_img (str): Directory name with images for training.
        train_mask (str): Directory name with mask images for training.
        img_format (tuple): Allowed image format.
        test_img (str): Directory name with images for validation.
        test_mask (str): Directory name with mask images for validation.
        mask_format (tuple): Allowed mask image format.
        with_img (bool): GraphFormer bool value for training with/without images.
    """
    dataset_test = False
    if isdir(join(dir, 'train')) and isdir(join(dir, 'test')):
        dataset_test = True

        if with_img:
            # Check if train img and coord exist and have same files
            if isdir(train_img) and isdir(train_mask):
                if len([f for f in listdir(train_img)
                        if f.endswith(img_format)]) == len([f for f in listdir(train_mask)
                                                            if f.endswith(mask_format)]):
                    if len([f for f in listdir(train_img) if f.endswith(img_format)]) == 0:
                        dataset_test = False
                else:
                    dataset_test = False

            # Check if test img and mask exist and have same files
            if isdir(test_img) and isdir(test_mask):
                if len([f for f in listdir(test_img)
                        if f.endswith(img_format)]) == len([f for f in listdir(test_mask)
                                                            if f.endswith(mask_format)]):
                    if len([f for f in listdir(test_img) if f.endswith(img_format)]) == 0:
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


class BuildTestDataSet:
    """
    MODULE FOR BUILDING TEST DATASET

    This module building a test dataset from training subset, by moving random
    files from train to test directory.
    Number of files is specified in %.

    Files are saved in dir/test/imgs and dir/test/masks.

    Args:
        dataset_dir (str): Directory with train test folders.
        train_test_ration (int): Percentage of dataset to be moved.
        prefix (str): Additional prefix name at the end of the file.
    """

    def __init__(self,
                 dataset_dir: str,
                 train_test_ration: int,
                 prefix: str):
        self.dataset = dataset_dir
        self.prefix = prefix

        assert 'test' in listdir(dataset_dir) and 'train' in listdir(dataset_dir), \
            f'Could not find train or test folder in directory {dataset_dir}'

        self.image_list = listdir(join(dataset_dir, 'train', 'imgs'))
        self.image_list.sort()
        self.mask_list = listdir(join(dataset_dir, 'train', 'masks'))
        self.mask_list.sort()

        self.train_test_ratio = (
            len(self.mask_list) * train_test_ration) // 100
        self.train_test_ratio = int(self.train_test_ratio)

        if self.train_test_ratio == 0:
            self.train_test_ratio = 1

    def __builddataset__(self):
        test_idx = []
        if len(self.image_list) == 0:
            data_no = len(self.mask_list)
        else:
            data_no = len(self.image_list)

        for _ in range(self.train_test_ratio):
            random_idx = np.random.choice(data_no)

            while random_idx in test_idx:
                random_idx = np.random.choice(data_no)

            test_idx.append(random_idx)

        if len(self.image_list) != 0:
            test_image_idx = list(np.array(self.image_list)[test_idx])
            test_mask_idx = list(np.array(self.mask_list)[test_idx])
        else:
            test_image_idx = []
            test_mask_idx = list(np.array(self.mask_list)[test_idx])

        for i in range(len(test_idx)):
            if len(self.image_list) != 0:
                # Move image file to test dir
                move(join(self.dataset, 'train', 'imgs', test_image_idx[i]),
                     join(self.dataset, 'test', 'imgs', test_image_idx[i]))
                move(join(self.dataset, 'train', 'masks', test_mask_idx[i]),
                     join(self.dataset, 'test', 'masks', test_mask_idx[i]))
            elif len(self.image_list) == 0 and len(self.mask_list) != 0:
                # Move mask file to test dir
                move(join(self.dataset, 'train', 'masks', test_mask_idx[i]),
                     join(self.dataset, 'test', 'masks', test_mask_idx[i]))
