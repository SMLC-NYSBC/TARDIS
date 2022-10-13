import shutil
from os.path import join

import numpy as np
from tardis.dist_pytorch.utils.augmentation import preprocess_data
from tardis.dist_pytorch.utils.dataloader import (FilamentDataset,
                                                  PartnetDataset,
                                                  ScannetColorDataset,
                                                  ScannetDataset)


class TestDataLoader:
    dir = join('tests', 'test_data', 'data_loader')
    coord_dir = join(dir, 'filament_mt', 'train', 'masks',
                     'am3D.CorrelationLines.am')
    img_dir = join('tests', 'test_data', 'data_loader')

    def test_preprocess_3Dimg(self):
        coord, img = preprocess_data(coord=self.coord_dir,
                                     image=join(self.img_dir, 'filament_mt',
                                                'train', 'imgs', 'am3D.am'),
                                     include_label=True,
                                     size=64,
                                     normalization='rescale')
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
                                     normalization='rescale')
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
                                            normalization='rescale')
        assert coord.ndim == 2, 'Incorrect no. of dimension'
        assert coord.shape == (10, 3), \
            f'Coord of wrong shape {coord.shape}'
        assert graph.shape == (10, 10), 'Graph of wrong shape!'

    # !!! DEPRECIATED !!!
    # def test_training_DL_w_img(self):
    #     train_DL = GraphDataset(coord_dir=self.dir,
    #                             coord_format=(".CorrelationLines.am", '.csv'),
    #                             img_format='.am',
    #                             img_dir=self.dir,
    #                             prefix=None,
    #                             size=12,
    #                             downsampling_if=500,
    #                             downsampling_rate=None,
    #                             normalize="rescale",
    #                             memory_save=False)

    #     coords_v, imgs_v, graph_v, output_idx, _ = train_DL.__getitem__(0)

    #     assert len(coords_v) == 1
    #     assert coords_v[0].shape == (10, 3)
    #     assert imgs_v[0].shape == (10, 1728)
    #     assert graph_v[0].shape == (10, 10)
    #     assert output_idx[0].shape == (10, )

    #     shutil.rmtree('./temp_train')

    # !!! DEPRECIATED !!!
    # def test_training_DL_no_img(self):
    #     train_DL = GraphDataset(coord_dir=join('tests', 'test_data', 'data_loader'),
    #                             coord_format=(".CorrelationLines.am", '.csv'),
    #                             img_dir=None,
    #                             prefix=None,
    #                             size=None,
    #                             downsampling_if=500,
    #                             downsampling_rate=None,
    #                             normalize="rescale",
    #                             memory_save=False)

    #     coords_v, imgs_v, graph_v, output_idx, _ = train_DL.__getitem__(0)

    #     assert len(coords_v) == 1
    #     assert coords_v[0].shape == (10, 3)
    #     assert imgs_v[0].shape == (1, 1)
    #     assert graph_v[0].shape == (10, 10)
    #     assert output_idx[0].shape == (10, )

    #     coords_v, imgs_v, graph_v, output_idx, _ = train_DL.__getitem__(1)

    #     assert len(coords_v) == 1
    #     assert coords_v[0].shape == (122, 2)
    #     assert imgs_v[0].shape == (1, 1)
    #     assert graph_v[0].shape == (122, 122)
    #     assert output_idx[0].shape == (122, )

    #     shutil.rmtree(join(self.dir, 'temp_train'))

    def test_filament_mt_dataloader(self):
        train_DL = FilamentDataset(coord_dir=join(self.dir, 'filament_mt',
                                                  'train', 'masks'),
                                   coord_format=(".CorrelationLines.am"),
                                   patch_if=500)

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10, )

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10, )

        shutil.rmtree('./temp_train')

    def test_filament_mem_dataloader(self):
        train_DL = FilamentDataset(coord_dir=join(self.dir, 'filament_mem',
                                                  'train', 'masks'),
                                   coord_format=(".csv"),
                                   patch_if=500)

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (122, 2)
        assert graph_v[0].shape == (122, 122)
        assert output_idx[0].shape == (122, )

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (122, 2)
        assert graph_v[0].shape == (122, 122)
        assert output_idx[0].shape == (122, )

        shutil.rmtree('./temp_train')

    def test_scannet_dataloader(self):
        train_DL = ScannetDataset(coord_dir=join(self.dir, 'scannet',
                                                 'train', 'masks'),
                                  coord_format=(".ply"),
                                  patch_if=500)

        # Build first time
        coords_v, _, graph_v, output_idx, clx_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (483, 3)
        assert graph_v[0].shape == (483, 483)
        assert output_idx[0].shape == (483, )
        assert clx_idx[0].shape[0] == 483

        # Load from memory
        coords_v, _, graph_v, output_idx, clx_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (483, 3)
        assert graph_v[0].shape == (483, 483)
        assert output_idx[0].shape == (483, )
        assert clx_idx[0].shape[0] == 483

        shutil.rmtree('./temp_train')

    def test_scannet_color_dataloader(self):
        train_DL = ScannetColorDataset(coord_dir=join(self.dir, 'scannet',
                                                      'train', 'masks'),
                                       coord_format=(".ply"),
                                       patch_if=500)

        # Build first time
        coords_v, rgb_idx, graph_v, output_idx, clx_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (483, 3)
        assert graph_v[0].shape == (483, 483)
        assert output_idx[0].shape == (483, )
        assert clx_idx[0].shape[0] == 483
        assert rgb_idx[0].shape[0] == 483

        # Load from memory
        coords_v, rgb_idx, graph_v, output_idx, clx_idx = train_DL.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (483, 3)
        assert graph_v[0].shape == (483, 483)
        assert output_idx[0].shape == (483, )
        assert clx_idx[0].shape[0] == 483
        assert rgb_idx[0].shape[0] == 483

        shutil.rmtree('./temp_train')

    def test_partnet_dataloader(self):
        train_DL = PartnetDataset(coord_dir=join(self.dir, 'partnet',
                                                 'train', 'masks'),
                                  coord_format=(".ply"),
                                  patch_if=500)

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 16
        assert coords_v[0].shape == (338, 3)
        assert graph_v[0].shape == (338, 338)
        assert output_idx[0].shape == (338, )

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_DL.__getitem__(0)

        assert len(coords_v) == 16
        assert coords_v[0].shape == (338, 3)
        assert graph_v[0].shape == (338, 338)
        assert output_idx[0].shape == (338, )

        shutil.rmtree('./temp_train')
