#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import sys
from os import listdir
from os.path import isfile, join
from typing import Optional

import numpy as np
import tifffile.tifffile as tif

from tardis_em.utils.errors import TardisError


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
        self.nx, self.ny, self.nz = (
            0,
            0,
            0,
        )  # Variable used to store xyz image dimension
        self.x, self.y, self.z = 0, 0, 0  # Variable to store number of patches in xyz
        self.stride = 0  # Variable to store step size

    def _find_xyz(self, file_list: list, idx: int):
        """
        Find index from for stitching image patches into one file.

        Args:
            file_list (list): List of files.
            idx: Find file index number.

        Returns:
            Update global class values.
        """
        self.z = (
            max(
                list(
                    map(
                        int,
                        [
                            str.split(f[:-4], "_")[1]
                            for f in file_list
                            if f.startswith(f"{idx}")
                        ],
                    )
                )
            )
            + 1
        )
        self.y = (
            max(
                list(
                    map(
                        int,
                        [
                            str.split(f[:-4], "_")[2]
                            for f in file_list
                            if f.startswith(f"{idx}")
                        ],
                    )
                )
            )
            + 1
        )
        self.x = (
            max(
                list(
                    map(
                        int,
                        [
                            str.split(f[:-4], "_")[3]
                            for f in file_list
                            if f.startswith(f"{idx}")
                        ],
                    )
                )
            )
            + 1
        )
        self.stride = max(
            list(
                map(
                    int,
                    [
                        str.split(f[:-4], "_")[4]
                        for f in file_list
                        if f.startswith(f"{idx}")
                    ],
                )
            )
        )

    def _calculate_dim(self, image: np.ndarray):
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

    def __call__(
        self,
        image_dir: str,
        mask: bool,
        output: Optional[str] = None,
        prefix="",
        dtype=np.uint8,
    ) -> np.ndarray:
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
        file_list = [f for f in file_list if f.endswith(".tif")]

        # Number of images to stitch
        self.idx = (
            max(list(map(int, [str.split(f[:-4], "_")[0] for f in file_list]))) + 1
        )

        for idx in range(self.idx):
            self._find_xyz(file_list, idx)
            self._calculate_dim(
                tif.imread(join(image_dir, f"{idx}_0_0_0_{self.stride}{prefix}.tif"))
            )

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

                        img_name = str(
                            join(
                                image_dir,
                                f"{idx}_{i}_{j}_{k}_{self.stride}{prefix}.tif",
                            )
                        )

                        img = tif.imread(img_name)

                        if self.nz == 0:
                            if img.shape != (self.ny, self.nx):
                                TardisError(
                                    id_="",
                                    py="tardis_em/cnn/data_processing.md/stitch.py",
                                    desc=f"Stitch image size does not match. {img.shape} "
                                    f"doesn't match ({self.ny}, {self.nx})",
                                )
                                sys.exit()
                        else:
                            if img.shape != (self.nz, self.ny, self.nx):
                                TardisError(
                                    id_="",
                                    py="tardis_em/cnn/data_processing.md/stitch.py",
                                    desc=f"Stitch image size does not match. {img.shape} "
                                    f"doesn't match ({self.nz}, {self.ny}, {self.nx})",
                                )
                                sys.exit()

                        if mask and self.nz == 0:
                            stitched_image[y_start:y_stop, x_start:x_stop] += img
                        elif mask and self.nz > 0:
                            stitched_image[
                                z_start:z_stop, y_start:y_stop, x_start:x_stop
                            ] += img
                        elif not mask and self.nz == 0:
                            stitched_image[y_start:y_stop, x_start:x_stop] += img
                        else:
                            stitched_image[
                                z_start:z_stop, y_start:y_stop, x_start:x_stop
                            ] += img

            # Reduce overlapping areas by averaging
            if mask:
                stitched_image = np.where(stitched_image > 0, 1, 0).astype(np.uint8)
            else:
                if self.nz == 0:
                    grid_y, grid_x = generate_grid(
                        stitched_image.shape,
                        [self.ny, self.nx],
                        [self.y, self.x],
                        self.stride,
                    )
                    if len(grid_y) > 0:
                        stitched_image[grid_y[:,], :] = (
                            stitched_image[grid_y[:,], :] / 2
                        )
                    if len(grid_x) > 0:
                        stitched_image[:, grid_x[:,]] = (
                            stitched_image[:, grid_x[:,]] / 2
                        )
                else:
                    grid_y, grid_x, grid_z = generate_grid(
                        stitched_image.shape,
                        [self.nz, self.ny, self.nx],
                        [self.z, self.y, self.x],
                        self.stride,
                    )

                    if len(grid_y) > 0:
                        stitched_image[grid_y[:, 0], grid_y[:, 1], :] = (
                            stitched_image[grid_y[:, 0], grid_y[:, 1], :] / 2
                        )
                    if len(grid_x) > 0:
                        stitched_image[grid_x[:, 0], :, grid_x[:, 1]] = (
                            stitched_image[grid_x[:, 0], :, grid_x[:, 1]] / 2
                        )
                    if len(grid_z) > 0:
                        stitched_image[grid_z[:,], ...] = (
                            stitched_image[grid_z[:,], ...] / 2
                        )

            if output is None:
                return np.array(stitched_image, dtype=dtype)
            else:
                tif.imwrite(
                    join(output, f"Stitched_Image_idx_{idx}.tif"),
                    np.array(stitched_image, dtype=dtype),
                )


def generate_grid(image_size: tuple, patch_size: list, grid_size: list, stride: int):
    """
    Generates grid coordinates for either 2D or 3D images.

    Args:
        image_size (list): The dimensions of the image.
            For 3D, it should be [z, y, x], and for 2D, [y, x].
        patch_size (list): The dimensions of each patch.
            For 3D, [nz, ny, nx], and for 2D, [ny, nx].
        grid_size (list): The grid size.
            For 3D, [gz, gy, gx], and for 2D, [gy, gx].
        stride (int): The stride for grid generation.

    Returns:
        Tuple[list]: A tuple of numpy arrays with coordinates of the grid.
    """
    # Determine if the image is 2D based on the length of image_size
    D2 = len(image_size) == 2

    # Unpack the dimensions based on whether the image is 2D or 3D
    if D2:
        z_dim, nz, gz = 0, 0, 0
        y_dim, x_dim = image_size
        ny, nx = patch_size
        gy, gx = grid_size
    else:
        z_dim, y_dim, x_dim = image_size
        nz, ny, nx = patch_size
        gz, gy, gx = grid_size

    # Generate y and x coordinates
    y_coords = [
        np.arange((ny * g - stride * g), ny * g - (stride * (g - 1)))
        for g in range(1, gy)
    ]
    if len(y_coords) > 0:
        y_coords = np.concatenate(
            [
                np.arange((ny * g - stride * g), ny * g - (stride * (g - 1)))
                for g in range(1, gy)
            ]
        )

    x_coords = [
        np.arange((nx * g - stride * g), nx * g - (stride * (g - 1)))
        for g in range(1, gx)
    ]
    if len(x_coords) > 0:
        x_coords = np.concatenate(
            [
                np.arange((nx * g - stride * g), nx * g - (stride * (g - 1)))
                for g in range(1, gx)
            ]
        )

    # Generate z coordinates and meshgrids if the image is 3D
    if not D2:
        z_coords = np.arange(z_dim)
        zz, yy = np.meshgrid(z_coords, y_coords, indexing="ij")
        coordinates_yy = np.vstack([zz.ravel(), yy.ravel()]).T

        zz, xx = np.meshgrid(z_coords, x_coords, indexing="ij")
        coordinates_xx = np.vstack([zz.ravel(), xx.ravel()]).T

        z_coords = [
            np.arange((nz * g - stride * g), nz * g - (stride * (g - 1)))
            for g in range(1, gz)
        ]
        if len(z_coords) > 0:
            z_coords = np.concatenate(z_coords)

        return coordinates_yy, coordinates_xx, z_coords
    else:
        return y_coords, x_coords
