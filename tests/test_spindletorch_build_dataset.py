from tardis_dev.spindletorch.data_processing.build_training_dataset import build_train_dataset
from os import mkdir, listdir
from os.path import join
from shutil import rmtree, copy
from tardis_dev.utils.load_data import load_image
import numpy as np


def test_build_datasets():
    mkdir('./tests/test_data/temp')
    mkdir('./tests/test_data/temp/train')
    mkdir('./tests/test_data/temp/train/imgs')
    mkdir('./tests/test_data/temp/train/masks')

    copy('./tests/test_data/data_type/am3D.am',
         './tests/test_data/temp/am3D.am')
    copy('./tests/test_data/data_type/am3D.CorrelationLines.am',
         './tests/test_data/temp/am3D.CorrelationLines.am')

    build_train_dataset(dataset_dir='./tests/test_data/temp/',
                        circle_size=250,
                        multi_layer=False,
                        resize_pixel_size=25,
                        trim_xy=64,
                        trim_z=64)

    assert len(listdir('./tests/test_data/temp/train/imgs')) == 60
    assert len(listdir('./tests/test_data/temp/train/masks')) == 60
    dir_img = listdir('./tests/test_data/temp/train/imgs')
    img, _ = load_image(join('./tests/test_data/temp/train/imgs', dir_img[5]))
    assert img.shape == (64, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    assert img.dtype == np.float32

    dir_mask = listdir('./tests/test_data/temp/train/masks')
    img, _ = load_image(join('./tests/test_data/temp/train/masks', dir_mask[5]))
    assert img.shape == (64, 64, 64)
    assert img.dtype == np.uint8

    rmtree('./tests/test_data/temp')
