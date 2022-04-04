from os.path import join

import numpy as np
import tifffile.tifffile as tif
from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.image_to_point_cloud import BuildPointCloud


class TestPointCloudBuilder:
    point_cloud = tif.imread(join('tests', 'test_data', 'pointcloud.tif'))
    downsampling = 5
    builder = ImageToPointCloud(tqdm=True)

    def test_builder(self):
        # Test loading 2D
        image = np.zeros((512, 512))
        image[250:275, 250:275] = 1
        BuildPointCloud(tqdm=True).check_data(image)

        # Test loading 3D
        image = np.zeros((64, 512, 512))
        image[15:20, 250:275, 250:275] = 1

        BuildPointCloud(tqdm=True).check_data(image)

    def check_pointcloud(self,
                         pointcloud):
        assert pointcloud.shape[1] == 3, \
            f'Wrong number of dimension. Given {pointcloud.shape[1]}, expected 3 [XYZ]!'

    def test_pointcloud2D(self):
        pc = self.builder(image=self.point_cloud[5, :],
                          euclidean_transform=False,
                          label_size=250,
                          down_sampling_voxal_size=None)

        self.check_pointcloud(pc)

    def test_pointcloud3D(self):
        pc = self.builder(image=self.point_cloud,
                          euclidean_transform=False,
                          label_size=250,
                          down_sampling_voxal_size=None)

        self.check_pointcloud(pc)

    def test_pointcloud_downsample2D(self):
        pc, pc_ld = self.builder(image=self.point_cloud[5, :],
                                 euclidean_transform=False,
                                 label_size=250,
                                 down_sampling_voxal_size=50)

        self.check_pointcloud(pc)
        self.check_pointcloud(pc_ld)

    def test_pointcloud_downsample3D(self):
        pc, pc_ld = self.builder(image=self.point_cloud,
                                 euclidean_transform=False,
                                 label_size=250,
                                 down_sampling_voxal_size=5)

        self.check_pointcloud(pc)
        self.check_pointcloud(pc_ld)
