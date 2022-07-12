from os.path import join

import numpy as np
import tardis.slcpy.utils.load_data as loader


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
        image, pixel_size, _ = loader.import_am(join('tests',
                                                     self.dir,
                                                     'am2D.am'))
        self.check_size2D(image)
        self.check_pixel_size(pixel_size)

    def test_am3D(self):
        image, pixel_size, _ = loader.import_am(join('tests',
                                                     self.dir,
                                                     'am3D.am'))
        assert image.shape == (8, 256, 256), \
            'Error while loading file. Incorrect image shape or orientation!'
        assert pixel_size == 92.8, \
            f'Pixel size was not detected correctly. Given {pixel_size}, expected !'

    def test_amira_spatialgraph(self):
        amira_src = join('tests', self.dir, 'am3D.CorrelationLines.am')

        AmiraImporter = loader.ImportDataFromAmira(src_am=amira_src,
                                                   src_img=None)
        segments = AmiraImporter.get_segments()
        points = AmiraImporter.get_points()
        _ = AmiraImporter.get_segmented_points()

        assert len(segments) == 3, \
            f'Wrong nubmer of imported segments. Given {len(segments)}, expected 3!'
        assert len(points) == 10, \
            f'Wrong nubmer of imported points. Given {len(points)}, expected 10!'

    def test_amira_spatialgraph_binary(self):
        amira_src = join('tests', self.dir, 'am3D.CorrelationLines.am')
        amira_binary = join('tests', self.dir, 'am3D.am')

        AmiraImporter = loader.ImportDataFromAmira(src_am=amira_src,
                                                   src_img=amira_binary)
        segments = AmiraImporter.get_segments()
        points = AmiraImporter.get_points()
        _ = AmiraImporter.get_segmented_points()

        # Check general data structure
        assert len(segments) == 3, \
            f'Wrong nubmer of imported segments. Given {len(segments)}, expected 3!'
        assert np.sum(segments) == 10, \
            f'Incorrect number of segments. Given {np.sum(segments)}, expected 10!'
        assert points.shape == (10, 3), 'Array of points imported impropriety!'

        # Test data transformation
        assert np.max(points[:, :2]) <= 256, \
            'Point transformation is incorrect!'

        # Test detected pixel size
        assert AmiraImporter.pixel_size == 92.8, \
            'Wrong pixel size detected!'
