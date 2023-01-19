#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import io
import os

import numpy as np
import torch

from tardis.utils.aws import get_weights_aws
from tardis.utils.device import get_device
from tardis.utils.errors import TardisError
from tardis.utils.export_data import NumpyToAmira
from tardis.utils.load_data import (import_am, import_tiff, ImportDataFromAmira,
                                    load_mrc_file)
from tardis.utils.logo import TardisLogo
from tardis.utils.spline_metric import compare_splines_probability
from tardis.utils.utils import EarlyStopping


def test_early_stop():
    er_stop = EarlyStopping()
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 0

    er_stop(val_loss=0.09)
    assert er_stop.counter == 0

    er_stop(val_loss=0.1)
    assert er_stop.counter == 1

    er_stop = EarlyStopping()
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

    tif, px = import_tiff(tiff='./tests/test_data/data_type/tif3D.tif')
    assert tif.shape == (78, 64, 32)


def test_rec_mrc():
    mrc, px = load_mrc_file(mrc='./tests/test_data/data_type/mrc2D.mrc')
    assert mrc.shape == (64, 32)
    assert px == 23.2

    rec, px = load_mrc_file(mrc='./tests/test_data/data_type/rec2D.rec')
    assert rec.shape == (64, 32)
    assert px == 23.2

    mrc, px = load_mrc_file(mrc='./tests/test_data/data_type/mrc3D.mrc')
    assert mrc.shape == (64, 78, 32)
    assert px == 23.2

    rec, px = load_mrc_file(mrc='./tests/test_data/data_type/rec3D.rec')
    assert rec.shape == (64, 78, 32)
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


def test_aws():
    aws = get_weights_aws(network='dist', subtype='triang', model='microtubules')
    assert isinstance(aws, str) or isinstance(aws, io.BytesIO)

    aws = get_weights_aws(network='dist', subtype='triang', model='microtubules')
    assert isinstance(aws, str)


def test_device():
    assert get_device('cpu') == torch.device('cpu')

    assert get_device(0) == torch.device('cpu') or get_device(0) == torch.device('cuda:0')

    assert get_device('mps') == torch.device('cpu') or get_device('mps') == torch.device(
        'mps')


def test_am_single_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    exporter = NumpyToAmira()
    exporter.export_amira(coords=df, file_dir='./test.am')

    assert os.path.isfile('./test.am')
    os.remove('./test.am')


def test_am_multi_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    df_1 = np.array(df)
    df_2 = np.array(df_1)

    exporter = NumpyToAmira()
    exporter.export_amira(coords=(df_1, df_2),
                          file_dir='./test.am')

    assert os.path.isfile('./test.am')
    os.remove('./test.am')


def test_am_label_export():
    df = np.zeros((25, 4))
    df_line = np.linspace(0, 5, 25)
    df_line = np.round(df_line)
    df[:, 0] = df_line

    df_1 = np.array(df)
    df_2 = np.array(df_1)

    exporter = NumpyToAmira()
    exporter.export_amira(coords=(df_1, df_2),
                          file_dir='./test.am',
                          labels=['test1', 'test2'])

    assert os.path.isfile('./test.am')
    os.remove('./test.am')


def test_logo():
    logo = TardisLogo()
    # Test short
    logo(title='Test_pytest')

    # Test long
    logo(title='Test_pytest',
         text_1='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' +
                'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


def test_error():
    TardisError(id='20',
                py='tests/test_general_utils.py',
                desc='PyTest Failed!')


def test_compare_splines_probability():
    # Test with matching splines
    spline_tardis = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    spline_amira = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    threshold = 1
    assert round(compare_splines_probability(spline_tardis, spline_amira, threshold),
                 2) == 0.67

    # Test with non-matching splines
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([[4, 4], [5, 5], [6, 6]])
    threshold = 1
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 0.0

    # Test with matching splines and threshold set too high
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([[1, 1], [2, 2], [3, 3]])
    threshold = 10
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 1.0

    # Test with matching splines and threshold set too low
    spline_tardis = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2],
                              [4, 4, 4], [5, 5, 5], [6, 6, 6]])
    spline_amira = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    threshold = 1
    assert round(compare_splines_probability(spline_tardis, spline_amira, threshold),
                 2) == 0.33

    # Test with matching splines and threshold set too low
    spline_tardis = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]])
    spline_amira = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2],
                             [4, 4, 4], [5, 5, 5], [6, 6, 6]])
    threshold = 1
    assert round(compare_splines_probability(spline_tardis, spline_amira, threshold),
                 2) == 1.0

    # Test with empty spline
    spline_tardis = np.array([[0, 0], [1, 1], [2, 2]])
    spline_amira = np.array([])
    threshold = 1
    assert compare_splines_probability(spline_tardis, spline_amira, threshold) == 0.0
