from os import mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import tardis.slcpy_data_processing.utils.stitch as stich
import tardis.slcpy_data_processing.utils.trim as trim
import tifffile.tifffile as tif


class TestTrimming:
    trim_xy = 64
    trim_z = 64
    temp_dir = join('tests', 'temp')

    def test_trim_image_even2D(self):
        image = np.zeros((512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        for _ in range(10):
            y = np.random.choice(8, size=1)[0]
            x = np.random.choice(8, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{0}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_odd2D(self):
        image = np.zeros((525, 582)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        for _ in range(10):
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{0}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_multi2D(self):
        image = np.zeros((525, 582, 3)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        for _ in range(10):
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{0}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy, 3), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_even3D(self):
        image = np.zeros((128, 512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        for _ in range(10):
            z = np.random.choice(2, size=1)[0]
            y = np.random.choice(8, size=1)[0]
            x = np.random.choice(8, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{z}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_odd3D(self):
        image = np.zeros((72, 525, 582)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        for _ in range(10):
            z = np.random.choice(2, size=1)[0]
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{z}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_multi3D(self):
        image = np.zeros((72, 525, 582, 3)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=True,
                        prefix='_test')

        for _ in range(10):
            z = np.random.choice(2, size=1)[0]
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{z}_{y}_{x}_0_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy, 3), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_stride2D(self):
        image = np.zeros((525, 582)) + 1
        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              clean_empty=False,
                              prefix='_test')

        for _ in range(10):
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{0}_{y}_{x}_25_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy), \
                'Wrong output size!'
        rmtree(self.temp_dir)

    def test_trim_stride3D(self):
        image = np.zeros((72, 525, 582)) + 1
        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              prefix='_test')

        for _ in range(10):
            z = np.random.choice(2, size=1)[0]
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{z}_{y}_{x}_25_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy), \
                'Wrong output size!'
        rmtree(self.temp_dir)

    def test_trim_stride_multi2D(self):
        image = np.zeros((525, 582, 3)) + 1
        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              prefix='_test')

        for _ in range(10):
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{0}_{y}_{x}_25_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy, 3), \
                'Wrong output size!'
        rmtree(self.temp_dir)

    def test_trim_stride_multi3D(self):
        image = np.zeros((72, 525, 582, 3)) + 1
        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              prefix='_test')

        for _ in range(10):
            z = np.random.choice(2, size=1)[0]
            y = np.random.choice(9, size=1)[0]
            x = np.random.choice(10, size=1)[0]

            image = tif.imread(
                join(self.temp_dir, f'{0}_{z}_{y}_{x}_25_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy, 3), \
                'Wrong output size!'
        rmtree(self.temp_dir)

    def check_stitch(self,
                     dir: str):
        mkdir(join(self.temp_dir, 'output'))

        stitcher = stich.StitchImages()

        stitcher(image_dir=dir,
                 output=join(self.temp_dir, 'output'),
                 mask=False,
                 prefix='_test',
                 dtype=np.int8)

    def test_stitch2D(self):
        image = np.zeros((512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        self.check_stitch(dir=self.temp_dir)
        rmtree(self.temp_dir)

    def test_stitch3D(self):
        image = np.zeros((128, 512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_image(image=image,
                        trim_size_xy=self.trim_xy,
                        trim_size_z=self.trim_z,
                        image_counter=0,
                        output=self.temp_dir,
                        clean_empty=False,
                        prefix='_test')

        self.check_stitch(dir=self.temp_dir)
        rmtree(self.temp_dir)

    def test_stitch_stride2D(self):
        image = np.zeros((512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              prefix='_test')

        self.check_stitch(dir=self.temp_dir)
        rmtree(self.temp_dir)

    def test_stitch_stride3D(self):
        image = np.zeros((128, 512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        trim.trim_with_stride(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              output=self.temp_dir,
                              image_counter=0,
                              stride=25,
                              prefix='_test')

        self.check_stitch(dir=self.temp_dir)
        rmtree(self.temp_dir)
