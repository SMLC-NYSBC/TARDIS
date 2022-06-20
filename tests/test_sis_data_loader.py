from os.path import join
from tardis.dist_pytorch.utils.dataloader import GraphDataset
from tardis.dist_pytorch.utils.augmentation import preprocess_data

import numpy as np


class TestDataLoader:
    dir = join('tests', 'test_data', 'data_loader')
    coord_dir = join(dir, 'am3D.CorrelationLines.am')
    img_dir = join(dir, 'am3D.am')

    def test_preprocess_3Dimg(self):
        coord, img = preprocess_data(coord=self.coord_dir,
                                     image=self.img_dir,
                                     include_label=True,
                                     size=64,
                                     pixel_size=None,
                                     normalization='rescale',
                                     memory_save=False)
        assert coord.ndim == 2
        assert coord.shape == (10, 4), \
            f'Coord of wrong shape {coord.shape}'
        assert img.shape == (10, 262144), \
            f'img of wrong shape {img.shape}'

    def test_preprocess_3Dseg(self):
        coord, img = preprocess_data(coord=self.coord_dir,
                                     image=None,
                                     include_label=True,
                                     size=64,
                                     pixel_size=None,
                                     normalization='rescale',
                                     memory_save=False)
        assert coord.ndim == 2
        assert coord.shape == (10, 4), \
            f'Coord of wrong shape {coord.shape}'
        assert np.all(img == 0), 'Image type not zeros'
        assert img.shape == (64, 64, 64), \
            f'img of wrong shape {img.shape}'

    def test_preprocess3D(self):
        coord, img, graph = preprocess_data(coord=self.coord_dir,
                                            image=None,
                                            include_label=False,
                                            size=64,
                                            pixel_size=None,
                                            normalization='rescale',
                                            memory_save=False)
        assert coord.ndim == 2, 'Incorrect no. of dimension'
        assert coord.shape == (10, 3), \
            f'Coord of wrong shape {coord.shape}'
        assert graph.shape == (10, 10), 'Graph of wrong shape!'

    def test_training_DL_w_img(self):
        train_DL = GraphDataset(coord_dir=self.dir,
                                coord_format=[".CorrelationLines.am", '.am'],
                                img_dir=self.dir,
                                prefix=None,
                                size=12,
                                # voxal_size=500,
                                downsampling_if=500,
                                # drop_rate=1,
                                downsampling_rate=None,
                                normalize="rescale",
                                memory_save=False)

        coords_v, imgs_v, graph_v, output_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert imgs_v[0].shape == (10, 1728)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10, )

    def test_training_DL_no_img(self):
        train_DL = GraphDataset(coord_dir=join('tests', 'test_data', 'data_loader'),
                                coord_format=[".CorrelationLines.am"],
                                img_dir=None,
                                prefix=None,
                                size=None,
                                # voxal_size=500,
                                downsampling_if=500,
                                # drop_rate=1,
                                downsampling_rate=None,
                                normalize="rescale",
                                memory_save=False)

        coords_v, imgs_v, graph_v, output_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert imgs_v[0].shape == (1, 1)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10, )
