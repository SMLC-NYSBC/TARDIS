from os.path import join

import numpy as np
import tifffile.tifffile as tif
from tardis.slcpy_data_processing.utils.image_to_point_cloud import BuildPointCloud


class TestPointCloudBuilder:
    point_cloud = tif.imread(join('tests', 'test_data', 'pointcloud.tif'))
    downsampling = 5
    
    def test_class(self):
        # Test loading 2D
        BuildPointCloud(image=np.zeros((512, 512)),
                        tqdm=True)

        # Test loading 3D
        BuildPointCloud(image=np.zeros((64, 512, 512)),
                        tqdm=True)

    def test_pointcloud2D(self):
        builder = BuildPointCloud(image=self.point_cloud[5, :],
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=None)
        
    def test_pointcloud3D(self):
        builder = BuildPointCloud(image=self.point_cloud,
                                  tqdm=True)

        pc = builder.build_point_cloud(edt=False,
                                       label_size=250,
                                       down_sampling=None)
