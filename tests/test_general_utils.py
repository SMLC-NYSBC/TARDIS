import numpy as np
import torch
from tardis_dev.utils.device import get_device
from tardis_dev.utils.load_data import (ImportDataFromAmira, import_am,
                                        import_mrc, import_tiff)
from tardis_dev.utils.utils import EarlyStopping


def test_early_stop():
    er_stop = EarlyStopping(patience=10, min_delta=0)
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 0

    er_stop(val_loss=0.09)
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 1

    er_stop = EarlyStopping(patience=10, min_delta=0)
    assert er_stop.counter == 0

    er_stop(f1_score=0.1)
    assert er_stop.counter == 0

    er_stop(f1_score=0.15)
    assert er_stop.counter == 0

    er_stop(f1_score=0.1)
    assert er_stop.counter == 1


def test_check_device():
    dev = get_device('cpu')
    assert dev == torch.device('cpu')

    dev = get_device(1)
    assert dev == torch.device(type='cuda', index=1)


def test_tif():
    tif, px = import_tiff(tiff='./tests/test_data/data_type/tif2D.tif')
    assert tif.shape == (64, 32)
    assert tif.dtype == np.uint8

    tif, px = import_tiff(tiff='./tests/test_data/data_type/tif3D.tif')
    assert tif.shape == (78, 64, 32)
    assert tif.dtype == np.uint8


def test_rec_mrc():
    mrc, px = import_mrc(mrc='./tests/test_data/data_type/mrc2D.mrc')
    assert mrc.shape == (64, 32)
    assert mrc.dtype == np.uint8
    assert px == 23.2

    rec, px = import_mrc(mrc='./tests/test_data/data_type/rec2D.rec')
    assert rec.shape == (64, 32)
    assert rec.dtype == np.uint8
    assert px == 23.2

    mrc, px = import_mrc(mrc='./tests/test_data/data_type/mrc3D.mrc')
    assert mrc.shape == (78, 64, 32)
    assert mrc.dtype == np.uint8
    assert px == 23.2

    rec, px = import_mrc(mrc='./tests/test_data/data_type/rec3D.rec')
    assert rec.shape == (78, 64, 32)
    assert rec.dtype == np.uint8
    assert px == 23.2


def test_am():
    am, px, ps, trans = import_am(am_file='./tests/test_data/data_type/am2D.am')

    assert am.shape == (64, 32)
    assert am.dtype == np.uint8
    assert px == 23.2
    assert np.all(trans == np.array((0, 0, 4640)))

    am, px, ps, trans = import_am(am_file='./tests/test_data/data_type/am3D.am')

    assert am.shape == (8, 256, 256)
    assert am.dtype == np.uint8
    assert px == 92.8


def test_am_sg():
    am = ImportDataFromAmira(src_am='./tests/test_data/data_type/am3D.CorrelationLines.am',
                             src_img='./tests/test_data/data_type/am3D.am')
    segments = am.get_segmented_points()
    assert segments.shape == (10, 4)

    point = am.get_points()
    assert point.shape == (10, 3)

    image, px = am.get_image()
    assert image.shape == (8, 256, 256)
    assert image.dtype == np.uint8

    px = am.get_pixel_size()
    assert px == 92.8
