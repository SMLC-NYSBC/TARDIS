import numpy as np
import tardis.slcpy_data_processing.utilis.trim as trim
from os.path import join, isdir
from os import mkdir
from shutil import rmtree
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

        idx = trim.trim_image(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              image_counter=0,
                              output=self.temp_dir,
                              clean_empty=False,
                              prefix='_test')
        assert idx - 1 == 64, \
            f'Wrong number of outputted images given {idx - 1} expected 64!'

        for _ in range(10):
            rand_idx = np.random.choice(idx, size=1)[0]
            if rand_idx == 0:
                rand_idx = 1

            image = tif.imread(
                join(self.temp_dir, str(rand_idx) + '_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_odd2D(self):
        image = np.zeros((525, 582)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        idx = trim.trim_image(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              image_counter=0,
                              output=self.temp_dir,
                              clean_empty=False,
                              prefix='_test')
        assert idx - 1 == 90, \
            f'Wrong number of outputted images given {idx - 1} expected 64!'

        for _ in range(10):
            rand_idx = np.random.choice(idx, size=1)[0]
            if rand_idx == 0:
                rand_idx = 1

            image = tif.imread(
                join(self.temp_dir, str(rand_idx) + '_test.tif'))
            assert image.shape == (self.trim_xy, self.trim_xy), \
                f'Wrong output size given {image.shape}, expected [64, 64]!'

        rmtree(self.temp_dir)

    def test_trim_image_even3D(self):
        image = np.zeros((128, 512, 512)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        idx = trim.trim_image(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              image_counter=0,
                              output=self.temp_dir,
                              clean_empty=False,
                              prefix='_test')
        assert idx - 1 == 128, \
            f'Wrong number of outputted images given {idx - 1} expected 128!'

        for _ in range(10):
            rand_idx = np.random.choice(idx, size=1)[0]
            if rand_idx == 0:
                rand_idx = 1

            image = tif.imread(
                join(self.temp_dir, str(rand_idx) + '_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)

    def test_trim_image_odd3D(self):
        image = np.zeros((72, 525, 582)) + 1

        if isdir(self.temp_dir):
            rmtree(self.temp_dir)
        mkdir(self.temp_dir)

        idx = trim.trim_image(image=image,
                              trim_size_xy=self.trim_xy,
                              trim_size_z=self.trim_z,
                              image_counter=0,
                              output=self.temp_dir,
                              clean_empty=False,
                              prefix='_test')
        assert idx - 1 == 180, \
            f'Wrong number of outputted images given {idx - 1} expected 64!'

        for _ in range(10):
            rand_idx = np.random.choice(idx, size=1)[0]
            image = tif.imread(
                join(self.temp_dir, str(rand_idx) + '_test.tif'))
            assert image.shape == (self.trim_z, self.trim_xy, self.trim_xy), \
                'Wrong output size!'

        rmtree(self.temp_dir)
