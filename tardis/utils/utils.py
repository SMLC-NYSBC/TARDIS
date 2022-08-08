from os import listdir
from os.path import isdir, join
from shutil import move
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist


def pc_median_dist(pc: np.ndarray,
                   avg_over=False,
                   box_size=0.15):
    """
    CALCULATE AVERAGE DISTANCE BETWEEN K-NN POINTS IN POINT CLOUD

    Args:
        pc: 2D/3D array of point cloud
        avg_over: If True, calculate median position of all point and calculate
            average k-NN for selected set of point in that area (speed up for
            big point clouds)
    """
    if avg_over:
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()

        # Build BB and offset by 10% from the border
        box_dim = pc.shape[1]

        if box_dim in [2, 3]:
            min_x = np.min(pc[:, 0])
            max_x = np.max(pc[:, 0])
            offset_x = (max_x - min_x) * box_size

            min_y = np.min(pc[:, 1])
            max_y = np.max(pc[:, 1])
            offset_y = (max_y - min_y) * box_size

        if box_dim == 3:
            min_z = np.min(pc[:, 2])
            max_z = np.max(pc[:, 2])
            offset_z = (max_z - min_z) * box_size
        else:
            min_z, max_z = 0, 0
            offset_z = 0

        x = np.median(pc[:, 0])
        y = np.median(pc[:, 1])

        if box_dim == 3:
            z = np.median(pc[:, 2])
        else:
            z = 0

        voxel = point_in_bb(pc,
                            min_x=x - offset_x, max_x=x + offset_x,
                            min_y=y - offset_y, max_y=y + offset_y,
                            min_z=z - offset_z, max_z=z + offset_z)
        voxel = pc[voxel]

        # Calculate KNN dist
        knn = cdist(voxel, voxel)
    else:
        knn = cdist(pc, pc)

    if knn.shape[0] < 3:
        return 1.0

    df_1 = [sorted(d)[1] for d in knn if sorted(d)[1] != 0]  # Fist KNN
    df_2 = [sorted(d)[2] for d in knn if sorted(d)[2] != 0]  # Second KNN

    return np.mean(df_1 + df_2)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Args:
        patience: how many epochs to wait before stopping when loss is
               not improving
        min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
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
                 val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def check_uint8(image: np.ndarray):
    """
    SIMPLE CHECK FOR UINT8 DATA TYPE

    Args:
        image: Image for evaluation
    """
    if np.all(np.unique(image) == [0, 1]):
        return image
    elif np.all(np.unique(image) == [0, 254]) or len(np.unique(image) > 2):
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
    """
    CHECK LIST USED TO EVALUATE IF DIRECTORY CONTAIN CORRECT DATA TYPE FOR CNN

    Args:
        dir: Main directory with all files
        train_img: Directory name with images for training
        train_mask: Directory name with mask images for training
        img_format: Allowed image format
        test_img: Directory name with images for validation
        test_mask: Directory name with mask images for validation
        mask_format: Allowed mask image format
        with_img: GraphFormer bool value for training with/without images
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

    Files are saved in dir/test/imgs and dir/test/masks
    Args:
        dataset_dir: Directory with train test folders
        train_test_ration: Percentage of dataset to be moved
        prefix: Additional prefix name at the end of the file
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


def point_in_bb(points: np.ndarray,
                min_x: np.inf, max_x: np.inf,
                min_y: np.inf, max_y: np.inf,
                min_z: Optional[np.float32] = None,
                max_z: Optional[np.float32] = None):
    """
    Compute a bounding_box filter on the given points

    Args:
        points: (n,3) array
        min_i, max_i: The bounding box limits for each coordinate. If some
        limits are missing, the default values are -infinite for the min_i and
        infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keep or not.
    """
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    if points.shape[0] == 3:
        if min_z is not None or max_z is not None:
            bound_z = np.logical_and(
                points[:, 2] > min_z, points[:, 2] < max_z)
        else:
            bound_z = np.asarray([True for _ in points[:, 2]])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    else:
        bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter
