from os.path import join

import numpy as np
import tardis.slcpy_data_processing.utilis.load_data as loader


class TestLoader:
    dir = 'test_data'
    def test_tif2D(self):
        image, _ = loader.import_tiff(join('tests', self.dir, 'tif2D.tif'),
                                      dtype=np.uint8)
        assert image.dtype == 'uint8', \
            'Error while loading .tif. Incorrect dtype!'
        assert image.shape == (64, 32), \
            'Error while loading .tif. Incorrect image shape or orientation!'

        image, _ = loader.import_tiff(join('tests', self.dir, 'tif2D.tif'),
                                      dtype=np.float32)
        assert image.dtype == 'float32', \
            'Error while loading .tif. Incorrect dtype!'
        assert image.shape == (64, 32), \
            'Error while loading .tif. Incorrect image shape or orientation!'

    def test_tif3D(self):
        image, _ = loader.import_tiff(join('tests', self.dir, 'tif3D.tif'),
                                      dtype=np.uint8)
        assert image.dtype == 'uint8', \
            'Error while loading .tif. Incorrect dtype!'
        assert image.shape == (64, 64, 32), \
            'Error while loading .tif. Incorrect image shape or orientation!'

        image, _ = loader.import_tiff(join('tests', self.dir, 'tif3D.tif'),
                                      dtype=np.float32)
        assert image.dtype == 'float32', \
            'Error while loading .tif. Incorrect dtype!'
        assert image.shape == (64, 64, 32), \
            'Error while loading .tif. Incorrect image shape or orientation!'

    def test_mrc2D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'mrc2D.mrc'))
        assert image.shape == (64, 32), \
            'Error while loading .mrc. Incorrect image shape or orientation!'
        assert pixel_size == 2.32, 'Pixel size was not detected correctly!'

    def test_mrc3D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'mrc3D.mrc'))
        assert image.shape == (64, 64, 32), \
            'Error while loading .mrc. Incorrect image shape or orientation!'
        assert pixel_size == 2.32, 'Pixel size was not detected correctly!'

    def test_rec3D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'rec3D.rec'))
        assert image.shape == (64, 64, 32), \
            'Error while loading .rec. Incorrect image shape or orientation!'
        assert pixel_size == 2.32, 'Pixel size was not detected correctly!'

    def test_am3D(self):
        image, pixel_size = loader.import_am(join('tests',
                                                  self.dir,
                                                  'am3D.am'))
        assert image.shape == (64, 64, 32), \
            'Error while loading .am. Incorrect image shape or orientation!'
        assert pixel_size == 2.32, 'Pixel size was not detected correctly!'
