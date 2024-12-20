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
    A class for stitching multiple image patches into a single image.

    This class is utilized to process and combine multiple small image patches
    into a large cohesive image. The images are expected to follow a specific
    naming convention that includes indices and dimensions, which the class
    uses for computation. This functionality is useful in domains such as
    image processing and machine learning, where use of segmented image parts
    during preprocessing and their subsequent recombination is common.
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
        Find and set the maximum values of `x`, `y`, `z`, and `stride` attributes from the
        provided `file_list` based on the specified index `idx`. The attributes are determined
        by extracting components of file names that match the given index and calculating
        the maximum of their respective numeric values.

        :param file_list: A list of strings representing file names, where each file name
                          contains underscore-separated values encoding attributes such as
                          `z`, `y`, `x`, and `stride` in specific positions.
        :param idx: An integer representing the index used to filter files starting with this
                    value in their names.

        :return: None. This is a setter method that updates the attributes `x`, `y`, `z`,
                 and `stride` in the class instance based on the processed `file_list` and `idx`.
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
        Calculates and assigns the dimensions of a given image based on its array shape.
        This method distinguishes between 2D and 3D arrays and updates the class attributes
        `nz`, `ny`, and `nx` accordingly. For 3D arrays, all three attributes are derived
        from the array's shape, while for 2D arrays, `nx` and `ny` are derived, and `nz` is
        set to 0.

        :param image: The input image represented as a NumPy array. It could be either a
            2D or 3D array.
        :type image: numpy.ndarray
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
        Calls the object as a function to process and stitch images from the specified
        directory, applying optional masking, and saving or returning the stitched output.

        This method reads and processes `.tif` image files from the provided directory.
        It stitches them together based on defined dimensions, overlaps, strides, and
        optional masking behavior. If specified, the stitched image is saved to the
        output directory; otherwise, it returns the result as a NumPy array.

        :param image_dir:
            The path to the directory containing the `.tif` image files to be stitched.
        :param mask:
            A boolean flag to enable masking. If True, overlapping regions of the stitched
            image are handled with binary mask operations.
        :param output:
            Optional path to save the stitched image. If None, the method returns the
            resulting stitched image as a NumPy array.
        :param prefix:
            A string prefix used to identify `.tif` image filenames for processing.
        :param dtype:
            The data type of the resulting stitched image. Defaults to `np.uint8`.

        :return:
            A NumPy array representing the stitched image(s) if `output` is None. The
            array can have 2D (if `nz` equals 0) or 3D (if `nz` is greater than 0) shape,
            and its dimensions depend on the stitcher's configuration and the images in
            the input directory.

        """
        """Extract information about images in dir_path"""
        file_list = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
        file_list = [f for f in file_list if f.endswith(".tif")]

        # Number of images to stitch
        self.idx = (
            max(list(map(int, [str.split(f[:-4], "_")[0] for f in file_list]))) + 1
        )

        if self.idx > 1:
            image_stack = []

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

            if self.idx > 1:
                image_stack.append(stitched_image)

        if output is None:
            return np.array(
                stitched_image if self.idx == 1 else image_stack, dtype=dtype
            )
        else:
            tif.imwrite(
                join(output, f"Stitched_Image_idx_{idx}.tif"),
                np.array(stitched_image if self.idx == 1 else image_stack, dtype=dtype),
            )


def generate_grid(image_size: tuple, patch_size: list, grid_size: list, stride: int):
    """
    Generates grid coordinates for a given image, with support for both 2D and 3D images. The
    method computes overlapping patch regions for an image based on the specified patch size, grid
    size, and stride. For 2D images, the function generates grid points for rows and columns,
    and for 3D images, it extends the functionality by incorporating depth information. Output
    grid coordinates serve as indexing patterns for segmenting images into smaller patches or
    for analyzing specific grid areas.

    :param image_size: Tuple representing dimensions of the image. For a 2D image,
        it takes the form (height, width). For a 3D image, it takes the form
        (depth, height, width).
    :param patch_size: List representing the size of the patches for either a 2D or 3D image.
        For a 2D image, it should be [patch_height, patch_width]. For a 3D image,
        [patch_depth, patch_height, patch_width].
    :param grid_size: List defining the number of grid steps along each dimension.
        For a 2D image, it is [grid_rows, grid_columns]. For a 3D image, it is
        [grid_depth, grid_rows, grid_columns].
    :param stride: Integer specifying the step size or overlap between the patch regions.

    :return: If the input image is 2D, returns two arrays containing the y-coordinates
        and x-coordinates of the grid. If the input image is 3D, returns three arrays:
        (1) a 2D array of coordinates for depth and rows, (2) a 2D array of coordinates
        for depth and columns, and (3) a 1D array of z-coordinates grid.
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
