import gc
from typing import Optional

import numpy as np
from skimage.morphology import skeletonize_3d


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

    def __init__(self,
                 tqdm):
        self.tqdm = tqdm

    def check_data(self,
                   image):
        try:
            if isinstance(image, str):
                from tardis.slcpy.utils.load_data import import_tiff

                image = import_tiff(img=image,
                                    dtype=np.int8)
            else:
                image = image
        except RuntimeWarning:
            raise Warning("Directory or input .tiff file is not correct...")

        if np.any(np.unique(image) > 1):  # Fix uint8 formatting
            image = image / 255

        assert np.unique(image)[1] == 1, \
            'Array or file directory loaded properly but image is not semantic mask...'

        assert image.ndim in [2, 3], 'File must be 2D or 3D array!'

        return image

    def build_point_cloud(self,
                          image: Optional[str] = np.ndarray,
                          EDT=False,
                          label_size=2.5,
                          down_sampling: Optional[float] = None):
        if self.tqdm:
            from tqdm import tqdm

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

                if self.tqdm:
                    edt_iter = tqdm(range(image_edt.shape[0]),
                                    'Calculating EDT map:',
                                    leave=True)
                else:
                    edt_iter = range(image_edt.shape[0])

                for i in edt_iter:
                    image_edt[i, :] = np.where(edt.edt(image[i, :]) > label_size, 1, 0)

            """Skeletonization"""
            image_point = np.where(skeletonize_3d(image_edt) > 0)
        else:
            """Skeletonization"""
            image_point = np.where(skeletonize_3d(image) > 0)

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
