import gc
from typing import Optional
import numpy as np
from skimage.morphology import skeletonize_3d


class BuildPointCloud:
    """
    MAIN MODULE FOR SEMANTIC MASK IMAGE DATA TRANSFORMATION INTO POINT CLOUD

    Module build point cloud from semantic mask based on skeletonization in 3D. 
    Optionally, user can trigger point cloud correction with euclidean distance 
    transformation to correct for skeletonization artefact. It will not fix issue
    with heavily overlaping objects.

    The workflow follows: (edt_2d -> edt_thresholding -> edt_bianary) ->
        skeletonization_3d -> output point cloud -> (downsampling)

    Args:
        image: source of the 2D/3D .tif file in [Z x Y x X] or 2D/3D array
        filter_small_object: Filter size to remove small object .
        clean_close_point: If True, close point will be removed.
    """

    def __init__(self,
                 image: Optional[str] = np.ndarray,
                 tqdm=True):
        self.tqdm = tqdm

        try:
            if isinstance(image, str):
                from tardis.slcpy_data_processing.utilis.load_data import import_tiff

                self.image = import_tiff(img=image,
                                         dtype=np.int8)
            else:
                self.image = image
        except RuntimeWarning:
            raise Warning("Directory or input .tiff file is not correct...")

        if np.unique(self.image) in [0, 255]:  # Fix uint8 formating
            self.image = self.image / 255
        assert np.unique(self.image) not in [1] and len(np.unique(self.image)) != 2, \
            'Array or file directory loaded properly but image is not semanti mask...'

    def build_point_cloud(self,
                          edt=False,
                          label_size=250,
                          down_sampling: Optional[float] = None):
        x, y, z = [], [], []
        if self.tqdm:
            from tqdm import tqdm

        if edt:
            import edt

            label_size = label_size / 100

            """Calculate EDT and apply threshold based on predefine mask size"""
            if image_edt.ndim == 2:
                image_edt = edt.edt(self.image)
                image_edt = np.array(np.where(image_edt > label_size, 1, 0),
                                     dtype=np.int8)
            else:
                image_edt_df = np.zeros(self.image.shape, dtype=np.float16)
                image_edt = np.zeros(self.image.shape, dtype=np.int8)

                if self.tqdm:
                    edt_iter = tqdm(range(image_edt_df.shape[0]),
                                    'Calculating EDT map:',
                                    total=len(range(image_edt_df.shape[0])))
                else:
                    edt_iter = range(image_edt_df.shape[0])

                for i in edt_iter:
                    image_edt[i, :] = np.array(np.where(edt.edt(self.image[i, :]) > label_size, 1, 0),
                                               dtype=np.int8)

                del(image_edt_df)

            """CleanUp to avoid memory loss"""
            del(self.image)
            gc.collect()

            """Skeletonization"""
            image_edt = np.where(skeletonize_3d(image_edt) > 0, 1, 0)

            """Output point cloud"""
            coordinates = []
        else:
            """Skeletonization"""
            image_sk = np.where(skeletonize_3d(self.image) > 0, 1, 0)

            """Output point cloud"""
            coordinates = []
            
        """CleanUp to avoid memory loss"""
        del image_edt
        gc.collect()

        """ Down-sampling point cloud by removing closest point """
        if down_sampling is not None:
            import open3d as o3d
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coordinates)
            coordinates_ds = np.asarray(pcd.voxel_down_sample(voxel_size=down_sampling).points)

        return coordinates
