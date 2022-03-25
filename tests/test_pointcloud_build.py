from tardis.slcpy_data_processing.utilis.image_to_point_cloud import BuildPointCloud
import numpy as np


class TestPointCloudBilder:
    downsampling = 5
    
    def test_class(self):
        BuildPointCloud(image=np.zeros((512, 512)),
                        tqdm=True)
        BuildPointCloud(image=np.zeros((64, 512, 512)),
                        tqdm=True)
        
    def test_pointcloud2D(self):
    
    def test_pointcloud3D(self):
    
    def test_pointcloud_downsamlping2D(self):
    
    def test_pointcloud_downsampling3D(self):
        