from os.path import join

import numpy as np
import torch
from tardis_dev.spindletorch.datasets.augment import (MinMaxNormalize,
                                                      RescaleNormalize,
                                                      SimpleNormalize,
                                                      preprocess)
from tardis_dev.spindletorch.datasets.dataset_loader import (CNNDataset,
                                                             PredictionDataset)


def test_dataloader():
    dir = './tests/test_data/data_loader/cnn/test/'
    dataset = CNNDataset(img_dir=join(dir, 'imgs'),
                         mask_dir=join(dir, 'masks'),
                         size=64,
                         mask_suffix='_mask',
                         normalize='simple',
                         transform=True,
                         out_channels=1)
    assert len(dataset) == 2

    img, mask = dataset.__getitem__(0)

    assert img.shape == (1, 64, 64, 64) and mask.shape == (1, 64, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    assert torch.sum(img) != 0

    dataset = PredictionDataset(img_dir=join(dir, 'imgs'),
                                out_channels=1)
    assert len(dataset) == 2

    img, idx = dataset.__getitem__(0)

    assert img.shape == (1, 64, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    assert torch.sum(img) != 0
    assert idx == '4_0_55_98_25'


def test_normalization():
    img = np.random.rand(64, 64, 64) * 255
    img = img.astype(np.uint8)

    s_norm = SimpleNormalize()
    mm_norm = MinMaxNormalize(min=10, max=256)
    res_norm = RescaleNormalize(range=(2, 98))

    s_img = s_norm(x=img)
    assert s_img.min() >= 0 and s_img.max() <= 1
    assert s_img.dtype == np.float64

    mm_img = mm_norm(x=img)
    assert mm_img.min() >= 0 and mm_img.max() <= 1
    assert mm_img.dtype == np.float64

    res_img = res_norm(x=img)
    assert res_img.min() >= 0 and res_img.max() <= 1
    assert res_img.dtype == np.float64


class TestDataSetBuilder2D3D:
    mask3D = np.zeros((64, 64, 64))
    mask3D[:32, 32, :32] = 1  # draw a line
    mask2D = np.zeros((64, 64))
    mask2D[32, :32] = 1  # draw a line

    def test_data_augmentation3D(self):
        img = np.random.rand(64, 64, 64) * 255
        img = img.astype(np.uint8)

        img_proc, mask = preprocess(image=img,
                                    mask=self.mask3D,
                                    normalization='simple',
                                    transformation=True,
                                    size=64,
                                    output_dim_mask=1)
        assert img_proc.shape == (1, 64, 64, 64)
        assert mask.shape == (1, 64, 64, 64)
        assert img_proc.dtype == np.float64
        assert np.sum(img - img_proc[0, :]) != 0
        assert img_proc.min() >= 0 and img_proc.max() <= 1
        assert mask.min() >= 0 and mask.max() <= 1

    def test_data_augmentation2D(self):
        img = np.random.rand(64, 64) * 255
        img = img.astype(np.uint8)

        img_proc, mask = preprocess(image=img,
                                    mask=self.mask2D,
                                    normalization='simple',
                                    transformation=True,
                                    size=64,
                                    output_dim_mask=1)
        assert img_proc.shape == (1, 64, 64)
        assert mask.shape == (1, 64, 64)
        assert img_proc.dtype == np.float64
        assert np.sum(img - img_proc[0, :]) != 0
        assert img_proc.min() >= 0 and img_proc.max() <= 1
        assert mask.min() >= 0 and mask.max() <= 1
