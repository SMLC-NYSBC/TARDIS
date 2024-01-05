#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import shutil
from os.path import join

import numpy as np

from tardis_em.dist_pytorch.datasets.augmentation import preprocess_data
from tardis_em.dist_pytorch.datasets.dataloader import (
    FilamentDataset,
    PartnetDataset,
    ScannetColorDataset,
    ScannetDataset,
)


class TestDataLoader:
    dir_ = join("tests", "test_data", "data_loader")
    coord_dir = join(dir_, "filament_mt", "train", "masks", "am3D.CorrelationLines.am")
    img_dir = join("tests", "test_data", "data_loader")

    def test_preprocess_3d_img(self):
        coord, img = preprocess_data(
            coord=self.coord_dir,
            image=join(self.img_dir, "filament_mt", "train", "imgs", "am3D.am"),
            size=64,
        )
        assert coord.ndim == 2
        assert coord.shape == (10, 4), f"Coord of wrong shape {coord.shape}"
        assert img.shape == (10, 262144), f"img of wrong shape {img.shape}"

    def test_preprocess_3d_seg(self):
        coord, img = preprocess_data(
            coord=self.coord_dir, size=64, normalization="rescale"
        )
        assert coord.ndim == 2
        assert coord.shape == (10, 4), f"Coord of wrong shape {coord.shape}"
        assert np.all(img == 0), "Image type not zeros"
        assert img.shape == (64, 64, 64), f"img of wrong shape {img.shape}"

    def test_preprocess3d(self):
        coord, _, graph = preprocess_data(
            coord=self.coord_dir, include_label=False, size=64, normalization="rescale"
        )
        assert coord.ndim == 2, "Incorrect no. of dimension"
        assert coord.shape == (10, 3), f"Coord of wrong shape {coord.shape}"
        assert graph.shape == (10, 10), "Graph of wrong shape!"

    def test_filament_mt_dataloader(self):
        train_dl = FilamentDataset(
            coord_dir=join(self.dir_, "filament_mt", "train", "masks"),
            coord_format=".CorrelationLines.am",
            patch_if=500,
        )

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10,)

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (10, 3)
        assert graph_v[0].shape == (10, 10)
        assert output_idx[0].shape == (10,)

        shutil.rmtree("./temp_train")

    def test_filament_mem_dataloader(self):
        train_dl = FilamentDataset(
            coord_dir=join(self.dir_, "filament_mem", "train", "masks"),
            coord_format=".csv",
            patch_if=500,
        )

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (122, 2)
        assert graph_v[0].shape == (122, 122)
        assert output_idx[0].shape == (122,)

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 1
        assert coords_v[0].shape == (122, 2)
        assert graph_v[0].shape == (122, 122)
        assert output_idx[0].shape == (122,)

        shutil.rmtree("./temp_train")

    def test_scannet_dataloader(self):
        train_dl = ScannetDataset(
            coord_dir=join(self.dir_, "scannet", "train", "masks"),
            coord_format=".ply",
            patch_if=500,
        )

        # Build first time
        coords_v, _, graph_v, output_idx, clx_idx = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)
        assert clx_idx[0].shape[0] == 1

        # Load from memory
        coords_v, _, graph_v, output_idx, clx_idx = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)
        assert clx_idx[0].shape[0] == 1

        shutil.rmtree("./temp_train")

    def test_scannet_color_dataloader(self):
        train_dl = ScannetColorDataset(
            coord_dir=join(self.dir_, "scannet", "train", "masks"),
            coord_format=".ply",
            patch_if=500,
        )

        # Build first time
        coords_v, rgb_idx, graph_v, output_idx, clx_idx = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)
        assert clx_idx[0].shape[0] == s
        assert rgb_idx[0].shape[0] == s

        # Load from memory
        coords_v, rgb_idx, graph_v, output_idx, clx_idx = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)
        assert clx_idx[0].shape[0] == s
        assert rgb_idx[0].shape[0] == s

        shutil.rmtree("./temp_train")

    def test_partnet_dataloader(self):
        train_dl = PartnetDataset(
            coord_dir=join(self.dir_, "partnet", "train", "masks"),
            coord_format=".ply",
            patch_if=500,
        )

        # Build first time
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)

        # Load from memory
        coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)

        assert len(coords_v) == 4
        s = coords_v[0].shape[0]
        assert coords_v[0].shape == (s, 3)
        assert graph_v[0].shape == (s, s)
        assert output_idx[0].shape == (s,)

        shutil.rmtree("./temp_train")

    # def test_s3dis_dataloader(self):
    #     train_dl = Stanford3DDataset(
    #         coord_dir=join(self.dir_, "s3dis", "train", "masks"),
    #         coord_format=".ply",
    #         patch_if=500,
    #     )
    #
    #     # Build first time
    #     coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)
    #
    #     assert int(len(coords_v)) == 10
    #     s = coords_v[0].shape[0]
    #     assert coords_v[0].shape == (s, 3)
    #     assert graph_v[0].shape == (s, s)
    #     assert output_idx[0].shape == (s,)
    #
    #     # Load from memory
    #     coords_v, _, graph_v, output_idx, _ = train_dl.__getitem__(0)
    #
    #     assert int(len(coords_v)) == 10
    #     s = coords_v[0].shape[0]
    #     assert coords_v[0].shape == (s, 3)
    #     assert graph_v[0].shape == (s, s)
    #     assert output_idx[0].shape == (s,)
    #
    #     shutil.rmtree("./temp_train")
