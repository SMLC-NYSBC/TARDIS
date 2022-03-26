from os.path import join

import numpy as np
import tardis.slcpy_data_processing.utilis.load_data as loader


class TestLoader:
    dir = 'test_data'

    def check_size2D(self,
                     image):
        assert image.shape == (64, 32), \
            'Error while loading file. Incorrect image shape or orientation!'

    def check_size3D(self,
                     image):
        assert image.shape == (78, 64, 32), \
            'Error while loading file. Incorrect image shape or orientation!'

    def check_pixel_size(self,
                         pixel_size):
         assert pixel_size == 23.2, \
            f'Pixel size was not detected correctly. Given {pixel_size}, expected 23.2!'
            
    def test_tif2D(self):
        image, _ = loader.import_tiff(join('tests', self.dir, 'tif2D.tif'),
                                      dtype=np.uint8)
        assert image.dtype == 'uint8', \
            'Error while loading .tif. Incorrect dtype!'
        self.check_size2D(image)

        image, _ = loader.import_tiff(join('tests', self.dir, 'tif2D.tif'),
                                      dtype=np.float32)
        assert image.dtype == 'float32', \
            'Error while loading .tif. Incorrect dtype!'
        self.check_size2D(image)

    def test_tif3D(self):
        image, _ = loader.import_tiff(join('tests', self.dir, 'tif3D.tif'),
                                      dtype=np.uint8)
        assert image.dtype == 'uint8', \
            'Error while loading .tif. Incorrect dtype!'
        self.check_size3D(image)

        image, _ = loader.import_tiff(join('tests', self.dir, 'tif3D.tif'),
                                      dtype=np.float32)
        assert image.dtype == 'float32', \
            'Error while loading .tif. Incorrect dtype!'
        self.check_size3D(image)

    def test_mrc2D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'mrc2D.mrc'))
        self.check_size2D(image)
        self.check_pixel_size(pixel_size)

    def test_mrc3D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'mrc3D.mrc'))
        self.check_size3D(image)
        self.check_pixel_size(pixel_size)

    def test_rec2D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'rec2D.rec'))
        self.check_size2D(image)
        self.check_pixel_size(pixel_size)

    def test_rec3D(self):
        image, pixel_size = loader.import_mrc(join('tests',
                                                   self.dir,
                                                   'rec3D.rec'))
        self.check_size3D(image)
        self.check_pixel_size(pixel_size)

    def test_am2D(self):
        image, pixel_size = loader.import_am(join('tests',
                                                  self.dir,
                                                  'am2D.am'))
        self.check_size2D(image)
        self.check_pixel_size(pixel_size)


    def test_am3D(self):
        image, pixel_size = loader.import_am(join('tests',
                                                  self.dir,
                                                  'am3D.am'))
        assert image.shape == (8, 256, 256), \
            'Error while loading file. Incorrect image shape or orientation!'
        assert pixel_size == 92.8, \
            f'Pixel size was not detected correctly. Given {pixel_size}, expected !'

    