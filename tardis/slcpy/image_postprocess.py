from typing import Optional

import numpy as np
from tardis.slcpy.utils.image_to_point_cloud import BuildPointCloud


class ImageToPointCloud:
    def __init__(self,
                 tqdm: bool):
        self.postprocess = BuildPointCloud(tqdm=tqdm)

    def __call__(self,
                 image: Optional[str] = np.ndarray,
                 euclidean_transform=True,
                 label_size=2.5,
                 down_sampling_voxal_size=None):

        return self.postprocess.build_point_cloud(image=image,
                                                  EDT=euclidean_transform,
                                                  label_size=label_size,
                                                  down_sampling=down_sampling_voxal_size)
