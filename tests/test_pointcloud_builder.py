from os.path import join

import numpy as np
import tifffile.tifffile as tif
from tardis.slcpy_data_processing.utils.image_to_point_cloud import BuildPointCloud


class TestPointCloudBuilder:
    point_cloud = tif.imread(join('tests', 'test_data', 'pointcloud.tif'))
    downsampling = 5
    
    def test_class(self):
        # Test loading 2D
        image = np.zeros((512, 512))
        image[250:275, 250:275] = 1
        BuildPointCloud(image=image,
                        tqdm=True)

        # Test loading 3D
        image = np.zeros((64, 512, 512))
        image[15:20, 250:275, 250:275] = 1
        
        BuildPointCloud(image=image,
                        tqdm=True)

    def check_pointcloud(self,
                         pointcloud):
        assert pointcloud.shape[1] == 3, \
            f'Wrong number of dimension. Given {pointcloud.shape[1]}, expected 3 [XYZ]!'
        
    def test_pointcloud2D(self):
        builder = BuildPointCloud(image=self.point_cloud[5, :],
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=None)

        self.check_pointcloud(pc)
        
    def test_pointcloud3D(self):
        builder = BuildPointCloud(image=self.point_cloud,
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=None)

        self.check_pointcloud(pc)

    def test_pointcloud_downsample2D(self):
        builder = BuildPointCloud(image=self.point_cloud[5, :],
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=5)

        self.check_pointcloud(pc)
        
    def test_pointcloud_downsample3D(self):
        builder = BuildPointCloud(image=self.point_cloud,
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=5)

        self.check_pointcloud(pc)