#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from os import listdir
from os.path import isfile, join
from typing import Optional

import numpy as np
import tifffile.tifffile as tif


class StitchImages:
    """
    MAIN MODULE TO STITCH IMAGE FROM IMAGE PATCHES

    Class object to stitch cut date into one big image. Object recognize images
    with naming 0_1_1_1_25_pf where:
    - 0 indicate id of image
    - 1 indicate xyz position
    - 25 indicate stride value for overlap
    - pf indicate optional prefix in file name
    """

    def __init__(self):
        self.idx = 0  # Variable storing number of stitched images
        self.nx, self.ny, self.nz = 0, 0, 0  # Variable used to store xyz image dimension
        self.x, self.y, self.z = 0, 0, 0  # Variable to store number of patches in xyz
        self.stride = 0  # Variable to store step size

    def _find_xyz(self,
                  file_list: list,
                  idx: int):
        """
        Find index from for stitching image patches into one file.

        Args:
            file_list (list): List of files.
            idx: Find file index number.

        Returns:
            Update global class values.
        """
        self.z = max(list(map(int,
                              [str.split(f[:-4], "_")[1] for f in file_list
                               if f.startswith(f'{idx}')]))) + 1
        self.y = max(list(map(int,
                              [str.split(f[:-4], "_")[2] for f in file_list
                               if f.startswith(f'{idx}')]))) + 1
        self.x = max(list(map(int,
                              [str.split(f[:-4], "_")[3] for f in file_list
                               if f.startswith(f'{idx}')]))) + 1
        self.stride = max(list(map(int,
                                   [str.split(f[:-4], "_")[4] for f in file_list
                                    if f.startswith(f'{idx}')])))

    def _calculate_dim(self,
                       image: np.ndarray):
        """
        Find and update image patch size from array.

        Args:
            image (np.ndarray): Image array.

        Returns:
            Update global class values.
        """
        if image.ndim == 3:
            self.nz, self.ny, self.nx = image.shape
        else:
            self.ny, self.nx = image.shape
            self.nz = 0

    def __call__(self,
                 image_dir: str,
                 mask: bool,
                 output: Optional[str] = None,
                 prefix='',
                 dtype=np.uint8) -> np.ndarray:
        """
        STITCH IMAGE FROM IMAGE PATCHES

        Args:
            image_dir (str): Directory where all images are stored.
            mask (np.ndarray): If True treat image as binary mask and sum-up overlay
                zones, else do replacement.
            output (str, Optional): Optional, output directory.
            dtype (np.dtype): Numpy dtype for output
            prefix (str): Prefix name if available.

        Returns:
            np.ndarray: If indicated output, image is saved in output directory
            else stitch images is return as array.
        """
        """Extract information about images in dir_path"""
        file_list = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
        file_list = [f for f in file_list if f.endswith('.tif')]

        self.idx = max(list(map(int, [str.split(f[:-4], "_")[0] for f in file_list]))) + 1

        for idx in range(self.idx):
            self._find_xyz(file_list, idx)
            self._calculate_dim(tif.imread(join(image_dir,
                                                f'{idx}_0_0_0_{self.stride}{prefix}.tif')))

            x_dim = self.nx + ((self.nx - self.stride) * (self.x - 1))
            y_dim = self.ny + ((self.ny - self.stride) * (self.y - 1))
            if self.nz == 0:
                z_dim = 0
                stitched_image = np.zeros((y_dim, x_dim), dtype=dtype)
            else:
                z_dim = self.nz + ((self.nz - self.stride) * (self.z - 1))
                stitched_image = np.zeros((z_dim, y_dim, x_dim), dtype=dtype)

            z_start, z_stop = 0 - (self.nz - self.stride), 0

            if self.z == 0:
                self.z = 1

            batch_iter_z = range(self.z)

            batch_iter_y = range(self.y)

            for i in batch_iter_z:
                z_start = z_start + self.nz - self.stride
                z_stop = z_start + self.nz
                y_start, y_stop = 0 - (self.ny - self.stride), 0

                for j in batch_iter_y:
                    y_start = y_start + self.ny - self.stride
                    y_stop = y_start + self.ny
                    x_start, x_stop = 0 - (self.nx - self.stride), 0

                    for k in range(self.x):
                        x_start = x_start + self.nx - self.stride
                        x_stop = x_start + self.nx

                        img_name = str(join(image_dir,
                                            f"{idx}_{i}_{j}_{k}_{self.stride}{prefix}.tif"))

                        img = tif.imread(img_name)

                        if self.nz == 0:
                            assert img.shape == (self.ny, self.nx)
                        else:
                            assert img.shape == (self.nz, self.ny, self.nx)

                        if mask and self.nz == 0:
                            stitched_image[y_start:y_stop,
                                           x_start:x_stop] += img
                        elif mask and self.nz > 0:
                            stitched_image[z_start:z_stop,
                                           y_start:y_stop,
                                           x_start:x_stop] += img
                        elif not mask and self.nz == 0:
                            stitched_image[y_start:y_stop,
                                           x_start:x_stop] = img
                        else:
                            stitched_image[z_start:z_stop,
                                           y_start:y_stop,
                                           x_start:x_stop] = img

            if mask:
                stitched_image = np.where(stitched_image > 0, 1, 0).astype(np.uint8)

            if output is None:
                return np.array(stitched_image, dtype=dtype)
            else:
                tif.imwrite(join(output, f'Stitched_Image_idx_{idx}.tif'),
                            np.array(stitched_image, dtype=dtype))
