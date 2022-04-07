from shutil import move
import numpy as np
from os import listdir
from os.path import join, isdir
import tifffile.tifffile as tif

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self,
                 patience=10,
                 min_delta=0):
        """
        patience: how many epochs to wait before stopping when loss is
               not improving
        min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def check_uint8(image: np.ndarray):
    tif.imsave('img.tif', image)
    print(np.unique(image))
    if np.all(np.unique(image) == [0, 1]):
        return image
    elif np.all(np.unique(image) == [0, 254]) or np.all(np.unique(image) == [0, 255]):
        return np.array(np.where(image > 1, 1, 0), dtype=np.int8)
    else:
        return None  # Image is empty


def check_dir(dir: str,
              train_img: str,
              train_mask: str,
              img_format: tuple,
              test_img: str,
              test_mask: str,
              mask_format: tuple,
              with_img: bool):
    dataset_test = False
    if isdir(join(dir, 'train')) and isdir(join(dir, 'test')):
        dataset_test = True

        if with_img:
            # Check if train img and coord exist and have same files
            if isdir(train_img) and isdir(train_mask):
                if len([f for f in listdir(train_img) if f.endswith(img_format)]) == \
                        len([f for f in listdir(train_mask) if f.endswith(mask_format)]):
                    if len([f for f in listdir(train_img) if f.endswith(img_format)]) == 0:
                        dataset_test = False
                else:
                    dataset_test = False

            # Check if test img and mask exist and have same files
            if isdir(test_img) and isdir(test_mask):
                if len([f for f in listdir(test_img) if f.endswith(img_format)]) == \
                        len([f for f in listdir(test_mask) if f.endswith(mask_format)]):
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

    Files are saved in dir/test/imgs and dir/test/masks
    Args:
        dataset_dir: Directory with train test folders
        train_test_ration: Percentage of dataset to be moved
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

        self.train_test_ratio = (len(self.mask_list) * train_test_ration) // 100
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
