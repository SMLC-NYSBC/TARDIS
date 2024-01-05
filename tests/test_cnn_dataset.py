#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from os.path import join

import numpy as np
import torch

from tardis_em.cnn.datasets.augmentation import preprocess
from tardis_em.cnn.datasets.dataloader import (
    CNNDataset,
    PredictionDataset,
)
from tardis_em.utils.normalization import (
    MinMaxNormalize,
    RescaleNormalize,
    SimpleNormalize,
)


def test_dataloader():
    dir_ = "./tests/test_data/data_loader/cnn/test/"
    dataset = CNNDataset(img_dir=join(dir_, "imgs"), mask_dir=join(dir_, "masks"))
    assert len(dataset) == 2

    img, mask = dataset.__getitem__(0)

    assert img.shape == (1, 64, 64, 64) and mask.shape == (1, 64, 64, 64)
    assert img.min() >= 0 and img.max() <= 255
    assert torch.sum(img) != 0

    dataset = PredictionDataset(img_dir=join(dir_, "imgs"))
    assert len(dataset) == 2

    img, idx = dataset.__getitem__(0)

    assert img.shape == (1, 64, 64, 64)
    assert img.min() >= -1 and img.max() <= 255
    assert torch.sum(img) != 0
    assert isinstance(idx, str) is True


def test_normalization():
    img = np.random.rand(64, 64, 64) * 255
    img = img.astype(np.uint8)

    s_norm = SimpleNormalize()
    mm_norm = MinMaxNormalize()
    res_norm = RescaleNormalize(clip_range=(2, 98))

    s_img = s_norm(x=img)
    assert s_img.min() >= 0 and s_img.max() <= 1
    assert s_img.dtype == np.float32

    mm_img = mm_norm(x=img)
    assert mm_img.min() >= -1 and mm_img.max() <= 1
    assert mm_img.dtype == np.float32

    res_img = res_norm(x=img)
    assert res_img.min() >= -1 and res_img.max() <= 255
    assert res_img.dtype == np.uint8


class TestDataSetBuilder2D3D:
    def test_data_augmentation3d(self):
        img = np.random.rand(64, 64, 64)
        img = img.astype(np.float32)

        img_proc, mask = preprocess(image=img, mask=img, transformation=True, size=64)
        assert img_proc.shape == (1, 64, 64, 64)
        assert mask.shape == (1, 64, 64, 64)
        assert img_proc.dtype == np.float32
        assert np.all(img_proc == mask)
        assert img_proc.min() >= -1 and img_proc.max() <= 1
        assert mask.min() >= 0 and mask.max() <= 1

    def test_data_augmentation2d(self):
        img = np.random.rand(64, 64)
        img = img.astype(np.float32)

        img_proc, mask = preprocess(image=img, mask=img, transformation=True, size=64)
        assert img_proc.shape == (1, 64, 64)
        assert mask.shape == (1, 64, 64)
        assert img_proc.dtype == np.float32
        assert np.all(img_proc == mask)
        assert img_proc.min() >= -1 and img_proc.max() <= 1
        assert mask.min() >= 0 and mask.max() <= 1
