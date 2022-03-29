import torch
from tardis.spindletorch.utils.augment import preprocess
import numpy as np


class TestDataSetBuilder3D:
    mask3D = np.zeros((64, 64, 64))
    mask3D[:32, 32, :32] = 1 # draw a line
    mask2D = np.zeros((64, 64))
    mask2D[32, :32] = 1 # draw a line
    
    def test_data_augmentation3D(self):
        augmented_data = preprocess(image=np.random.rand(64, 64, 64),
                                    mask=self.mask3D,
                                    normalization='simple',
                                    transformation=True,
                                    size=(64,64,64),
                                    output_dim_mask=1)
        
    def test_data_augmentation2D(self):
        augmented_data = preprocess(image=np.random.rand(64, 64),
                                    mask=self.mask2D,
                                    normalization='simple',
                                    transformation=True,
                                    size=(64,64),
                                    output_dim_mask=1)

