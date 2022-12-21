import gc
from typing import Optional

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize_3d, skeletonize


class ImageToPointCloud:
    def __init__(self):
        self.postprocess = BuildPointCloud()

    def __call__(self,
                 image: Optional[str] = np.ndarray,
                 euclidean_transform=True,
                 label_size=2.5,
                 down_sampling_voxal_size=None,
                 as_2d=False):

        return self.postprocess.build_point_cloud(image=image,
                                                  EDT=euclidean_transform,
                                                  label_size=label_size,
                                                  down_sampling=down_sampling_voxal_size,
                                                  as_2d=as_2d)


class BuildPointCloud:
    """
    MAIN MODULE FOR SEMANTIC MASK IMAGE DATA TRANSFORMATION INTO POINT CLOUD

    Module build point cloud from semantic mask based on skeletonization in 3D.
    Optionally, user can trigger point cloud correction with euclidean distance
    transformation to correct for skeletonization artefact. It will not fix
    issue with heavily overlapping objects.

    The workflow follows: optional(edt_2d -> edt_thresholds -> edt_binary) ->
        skeletonization_3d -> output point cloud -> (downsampling)

    Args:
        image: source of the 2D/3D .tif file in [Z x Y x X] or 2D/3D array
        filter_small_object: Filter size to remove small object .
        clean_close_point: If True, close point will be removed.
    """

    def __init__(self):
        pass

    def check_data(self,
                   image):
        try:
            if isinstance(image, str):
                from tardis.utils.load_data import import_tiff

                image, _ = import_tiff(img=image,
                                       dtype=np.int8)
        except RuntimeWarning:
            raise Warning("Directory/input .tiff file/array is not correct...")

        assert image.ndim in [2, 3], 'File must be 2D or 3D array!'

        """Check for binary"""
        unique_val = pd.unique(image.flatten())  # Use panda for speed
        assert len(unique_val) == 2, 'Not binary image'

        """Check for int8 vs uint8"""
        if np.any(unique_val > 254):  # Fix uint8 formatting
            image = image / 255

        """Any other exertions"""
        assert unique_val[1] == 1, \
            'Array or file directory loaded properly but image is not semantic mask...'

        return image

    def build_point_cloud(self,
                          image: Optional[str] = np.ndarray,
                          EDT=False,
                          label_size=2.5,
                          down_sampling: Optional[float] = None,
                          as_2d=False):
        image = self.check_data(image)

        if EDT:
            import edt

            """Calculate EDT and apply threshold based on predefine mask size"""
            if image.ndim == 2:
                image_edt = edt.edt(image)
                image_edt = np.array(np.where(image_edt > label_size, 1, 0),
                                     dtype=np.int8)
            else:
                image_edt = np.zeros(image.shape, dtype=np.int8)

                for i in range(image_edt.shape[0]):
                    image_edt[i, :] = np.where(edt.edt(image[i, :]) > label_size, 1, 0)

            """Skeletonization"""
            if image_edt.flatten().shape[0] > 1000000000 or as_2d:
                image_point = np.zeros(image_edt.shape, dtype=np.int8)
                start = 0

                if as_2d:
                    for i in range(image_edt.shape[0]):
                        image_point[i, :] = skeletonize(image_edt[i, :])
                else:
                    for i in range(10, image_edt.shape[0], 10):
                        if (image_edt.shape[0] - i) // 10 == 0:
                            i = image_edt.shape[0]
                        image_point[start:i, :] += skeletonize_3d(image_edt[start:i, :])
                        start = i

                image_point = np.where(image_point > 0)
            else:
                image_point = np.where(skeletonize_3d(image_edt) > 0)
        else:
            """Skeletonization"""
            if image.flatten().shape[0] > 1000000000 or as_2d:
                image_point = np.zeros(image.shape, dtype=np.int8)
                start = 0

                if as_2d:
                    for i in range(image.shape[0]):
                        image_point[i, :] = skeletonize(image[i, :])
                else:
                    for i in range(10, image.shape[0], 10):
                        if (image.shape[0] - i) // 10 == 0:
                            i = image.shape[0]

                        image_point[start:i, :] += skeletonize_3d(image[start:i, :])
                        start = i
                image_point = np.where(image_point > 0)
            else:
                image_point = np.where(skeletonize_3d(image_edt) > 0)

        """CleanUp to avoid memory loss"""
        image = None
        self.image_edt = None
        del image, self.image_edt

        """Output point cloud [X x Y x Z]"""
        if len(image_point) == 2:
            """If 2D bring artificially Z dim == 0"""
            coordinates_HD = np.stack((image_point[1],
                                       image_point[0],
                                       np.zeros(image_point[0].shape))).T
        elif len(image_point) == 3:
            coordinates_HD = np.stack((image_point[2],
                                       image_point[1],
                                       image_point[0])).T

        """CleanUp to avoid memory loss"""
        image_point = None
        del image_point
        gc.collect()

        """ Down-sampling point cloud by removing closest point """
        if down_sampling is not None:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coordinates_HD)
            coordinates_LD = np.asarray(
                pcd.voxel_down_sample(voxel_size=down_sampling).points)

            return coordinates_HD, coordinates_LD

        return coordinates_HD
