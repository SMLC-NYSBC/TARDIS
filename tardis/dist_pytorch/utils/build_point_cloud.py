#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import gc
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from skimage.morphology import skeletonize, skeletonize_3d

from tardis.utils.errors import TardisError


class BuildPointCloud:
    """
    MAIN MODULE FOR SEMANTIC MASK IMAGE DATA TRANSFORMATION INTO POINT CLOUD

    Module build point cloud from semantic mask based on skeletonization in 3D.
    Optionally, user can trigger point cloud correction with euclidean distance
    transformation to correct for skeletonization artefact. It will not fix
    issue with heavily overlapping objects.

    The workflow follows: optional(edt_2d -> edt_thresholds -> edt_binary) ->
        skeletonization_3d -> output point cloud -> (down-sampling)
    """

    @staticmethod
    def check_data(image: Optional[str] = np.ndarray) -> np.ndarray:
        """
        Check image data and correct it if needed to uint8 type.

        Args:
            image (np.ndarray, str): 2D/3D image array

        Returns:
            np.ndarray: Check and converted to uint8 data
        """
        try:
            if isinstance(image, str):
                from tardis.utils.load_data import import_tiff

                image, _ = import_tiff(tiff=image)
        except RuntimeWarning:
            TardisError('121',
                        'tardis/dist/utils',
                        "Directory/input .tiff file/array is not correct...")

        if image.ndim not in [2, 3]:
            TardisError('113',
                        'tardis/dist/utils',
                        f'Image dim expected to be 2 or 3 bu got {image.ndim}')

        """Check for binary"""
        unique_val = np.sort(pd.unique(image.flatten()))  # Use panda for speed
        if len(unique_val) != 2:
            TardisError('115',
                        'tardis/dist/utils',
                        f'Not binary image. Expected 0-1 value but got: {unique_val}')

        """Check for int8 vs uint8"""
        if np.any(unique_val > 254):  # Fix uint8 formatting
            image = image / 255

        """Any other exertions"""
        if unique_val[1] != 1:
            TardisError('115',
                        'tardis/dist/utils',
                        'Array or file directory loaded properly but image '
                        'is not semantic mask... '
                        f'Expected 0-1 value but got: {unique_val}')

        return image

    def build_point_cloud(self,
                          image: Optional[str] = np.ndarray,
                          EDT=False,
                          down_sampling: Optional[float] = None,
                          as_2d=False) -> Union[Tuple[ndarray, ndarray], np.ndarray]:
        """
        Build point cloud data from semantic mask.

        Args:
            image (np.ndarray): Predicted semantic mask.
            EDT (bool): If True, compute EDT to extract line centers.
            down_sampling (float, None): If not None, down-sample point cloud with open3d.
            as_2d: Treat data as 2D not 3D.

        Returns:
            Union[Tuple[ndarray, ndarray], np.ndarray]: Point cloud of 2D/3D
            semantic objects.
        """
        image = self.check_data(image)

        if EDT:
            import edt

            """Calculate EDT and apply threshold based on predefine mask size"""
            if image.ndim == 2:
                image_edt = edt.edt(image)
                image_edt = np.where(image_edt > (image_edt.max() / 2), 1, 0)
            else:
                image_edt = np.zeros(image.shape, dtype=np.uint8)

                if as_2d:
                    for i in range(image_edt.shape[0]):
                        df_edt = edt.edt(image[i, :])
                        edt_factor = df_edt.max()

                        if edt_factor > 3:
                            image_edt[i, :] = np.where(df_edt > (edt_factor / 3),
                                                       df_edt, 0)
                elif image_edt.flatten().shape[0] > 1000000000:
                    start = 0

                    for i in range(10, image_edt.shape[0], 10):
                        if (image_edt.shape[0] - i) // 10 == 0:
                            i = image_edt.shape[0]

                        df_edt = edt.edt(image[start:i, :])
                        edt_factor = df_edt.max()

                        if edt_factor > 3:
                            image_edt[start:i, :] = np.where(df_edt > (edt_factor / 3),
                                                             df_edt, 0)
                        else:
                            image_edt[start:i, :] = image[start:i, :]
                        start = i
                else:
                    image_edt = edt.edt(image)
                    edt_factor = image_edt.max()

                    if edt_factor > 3:
                        image_edt = np.where(image_edt > (edt_factor / 3),
                                             image_edt, 0)
                    else:
                        image_edt = image
                image_edt = np.where(image_edt > 0, 1, 0)

            image_edt = image_edt.astype(np.uint8)

            """Skeletonization"""
            if image_edt.flatten().shape[0] > 1000000000 or as_2d:
                image_point = np.zeros(image_edt.shape, dtype=np.int8)
                start = 0

                if image.ndim == 2:
                    image_point = skeletonize(image_edt)
                elif as_2d:
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

            """CleanUp to avoid memory loss"""
            del image, image_edt
        else:
            """Skeletonization"""
            if image.flatten().shape[0] > 1000000000 or as_2d:
                image_point = np.zeros(image.shape, dtype=np.int8)
                start = 0

                if image.ndim == 2:
                    image_point = skeletonize(image)
                elif as_2d:
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
                image_point = np.where(skeletonize_3d(image) > 0)

            """CleanUp to avoid memory loss"""
            del image

        """Output point cloud [X x Y x Z]"""
        if len(image_point) == 2:
            """If 2D bring artificially Z dim == 0"""
            coordinates_HD = np.stack((image_point[1],
                                       image_point[0],
                                       np.zeros(image_point[0].shape))).T
        else:
            coordinates_HD = np.stack((image_point[2],
                                       image_point[1],
                                       image_point[0])).T

        """CleanUp to avoid memory loss"""
        del image_point
        gc.collect()

        """ Down-sampling point cloud by removing closest point """
        if down_sampling is not None:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coordinates_HD)
            coordinates_LD = np.asarray(
                pcd.voxel_down_sample(voxel_size=down_sampling).points
            )

            return coordinates_HD, coordinates_LD
        return coordinates_HD
