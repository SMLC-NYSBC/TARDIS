#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from math import ceil
from os import mkdir
from os.path import isdir, join
from typing import Optional, Tuple

import numpy as np
from tifffile import tifffile as tif

from tardis_em.cnn.data_processing.scaling import scale_image
from tardis_em.utils.device import get_device
from tardis_em.utils.export_data import to_mrc
from tardis_em.utils.errors import TardisError


def trim_with_stride(
    image: np.ndarray,
    trim_size_xy: int,
    trim_size_z: int,
    output: str,
    image_counter: int,
    scale: list,
    clean_empty=True,
    keep_if=0.01,
    stride=25,
    mask: Optional[np.ndarray] = None,
    log=True,
    pixel_size=None,
    device=get_device("cpu"),
):
    """
    Trim a 2D/3D image with optional masking into smaller patches with a specific stride
    and size. This function is designed for preprocessing image data for machine learning
    tasks, ensuring that the image dimensions meet specified requirements or adjusting
    them when necessary.

    :param image:
        The input image array, which can be 2D or 3D, and include grayscale or RGB
        channels. The data type and shape may vary depending on the input.
    :param trim_size_xy:
        Desired size for trimming in the X and Y dimensions. The input value determines
        the width and height of each patch. If not provided, default values will be
        calculated.
    :param trim_size_z:
        Desired size for trimming in the Z dimension. The input value determines
        the depth of each patch in case of 3D data. If not provided, default values
        will be calculated.
    :param output:
        Directory path where the trimmed patches will be saved. Output images and masks
        are stored in subdirectories.
    :param image_counter:
        Counter used for naming the output files. This ensures unique filenames during
        the trimming process.
    :param scale:
        A scaling factor list that adjusts the image size before trimming is performed.
    :param clean_empty:
        When set to True, automatic filtering is applied to exclude patches considered
        empty based on a specific condition.
    :param keep_if:
        Minimum threshold for keeping patches. The condition ensures that patches
        with non-empty values above this percentage are retained.
    :param stride:
        The stride size for trimming patches. Stride determines the overlap between
        adjacent patches when slicing the image.
    :param mask:
        Optional mask array corresponding to the input image. Used for segmentation
        or region-specific analyses. The function scales and trims the mask alongside
        the image.
    :param log:
        Flag to enable or disable logging of the process; logs errors and important
        actions during execution when set to True.
    :param pixel_size:
        Indicates whether the output should be saved in '.mrc' (specific file format)
        or '.tif' (common image format). Leave None for default behavior.
    :param device:
        Specifies the computational device (CPU, GPU, etc.) to be used if scaling
        operations employ device-specific optimizations.

    :return:
        Returns None. Side effect: Saves the image patches and corresponding mask
        patches (when provided) to the specified output directory. Each patch is
        named according to its positional indices to maintain unique identification.
    """
    img_dtype = np.float32

    if not isdir(join(output, "imgs")):
        mkdir(join(output, "imgs"))
    if not isdir(join(output, "masks")) and mask is not None:
        mkdir(join(output, "masks"))

    if mask is not None:
        mask_dtype = np.uint8
        image, mask, dim = scale_image(
            image=image,
            mask=mask,
            scale=tuple(scale),
            device=device,
        )
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if image.shape != mask.shape:
            TardisError(
                "111",
                "tardis_em/cnn/data_processing.md/trim.py",
                f"Image {image.shape} has different shape from mask {mask.shape}",
            )
    else:
        image, dim = scale_image(image=image, scale=scale, device=device)

    if img_dtype != image.dtype:
        TardisError(
            "111",
            "tardis_em/cnn/data_processing.md/trim.py",
            f"Image {img_dtype} has different dtype after interpolation {image.dtype}",
        )

    nz, ny, nx, nc = 0, 0, 0, None
    if image.ndim == 4 and dim == 3:  # 3D RGB
        nz, ny, nx, nc = image.shape
    elif image.ndim == 3 and dim == 3:  # 2D RGB
        ny, nx, nc = image.shape
        nz = 0  # 2D
    elif image.ndim == 3 and dim == 1:  # 3D gray
        nz, ny, nx = image.shape
        nc = None  # Gray
    elif image.ndim == 2:  # 2D gray
        ny, nx = image.shape
        nc = None  # 2D
        nz = 0  # Gray

    if stride is None:
        TardisError(
            "112",
            "tardis_em/cnn/data_processing.md",
            "Trim sizes or stride has to be indicated!",
        )

    """Calculate number of patches and stride for xyz"""
    x, y, z = ceil(nx / trim_size_xy), ceil(ny / trim_size_xy), ceil(nz / trim_size_z)

    if nz > 0:
        x_pad, y_pad, z_pad = (
            (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx,
            (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny,
            (trim_size_z + (trim_size_z - stride) * (z - 1)) - nz,
        )
    else:
        x_pad, y_pad, z_pad = (
            (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx,
            (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny,
            0,
        )

    """Adapt number of patches or patch size for trimming"""
    if trim_size_xy is not None or trim_size_z is not None:
        while x_pad < 0:
            x += 1
            x_pad += trim_size_xy - stride
        while y_pad < 0:
            y += 1
            y_pad += trim_size_xy - stride
        if nz > 0:
            while z_pad < 0:
                z += 1
                z_pad += trim_size_z - stride
    else:
        while x_pad <= 0 or y_pad <= 0:
            trim_size_xy += 1
            x_pad = (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx
            y_pad = (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny

        if nz > 0:
            while z_pad < 0:
                trim_size_z += 1
                z_pad = (trim_size_z + ((trim_size_z - stride) * (z - 1))) - nz
        else:
            z_pad = 0

    """Expand image of a patch"""
    if nz > 0:
        if nc is not None:
            image_padded = np.pad(image, [(0, z_pad), (0, y_pad), (0, x_pad), (0, 0)])
        else:
            image_padded = np.pad(image, [(0, z_pad), (0, y_pad), (0, x_pad)])

        if mask is not None:
            mask_padded = np.pad(mask, [(0, z_pad), (0, y_pad), (0, x_pad)])
    else:
        if nc is not None:
            image_padded = np.pad(image, [(0, y_pad), (0, x_pad), (0, 0)])
        else:
            image_padded = np.pad(image, [(0, y_pad), (0, x_pad)])

            if mask is not None:
                mask_padded = np.pad(mask, [(0, y_pad), (0, x_pad)])

    """Trim image and mask with stride"""
    z_start, z_stop = 0 - (trim_size_z - stride), 0
    count = len(range(x)) * len(range(y))
    if len(range(z)) > 0:
        count = count * len(range(z))
    print(f"Found {count} patches...")

    count_save = 0
    if z == 0:
        z = 1

    for i in range(z):
        z_start = z_start + trim_size_z - stride
        z_stop = z_start + trim_size_z
        y_start, y_stop = 0 - (trim_size_xy - stride), 0

        for j in range(y):
            y_start = y_start + trim_size_xy - stride
            y_stop = y_start + trim_size_xy
            x_start, x_stop = 0 - (trim_size_xy - stride), 0

            for k in range(x):
                x_start = x_start + trim_size_xy - stride
                x_stop = x_start + trim_size_xy

                if pixel_size is None:
                    img_name = str(f"{image_counter}_{i}_{j}_{k}_{stride}.tif")
                    mask_name = str(f"{image_counter}_{i}_{j}_{k}_{stride}_mask.tif")
                else:
                    img_name = str(f"{image_counter}_{i}_{j}_{k}_{stride}.mrc")
                    mask_name = str(f"{image_counter}_{i}_{j}_{k}_{stride}_mask.mrc")

                if nc is None:
                    if nz > 0:
                        trim_img = image_padded[
                            z_start:z_stop, y_start:y_stop, x_start:x_stop
                        ]
                        if mask is not None:
                            trim_mask = mask_padded[
                                z_start:z_stop, y_start:y_stop, x_start:x_stop
                            ]
                    else:
                        trim_img = image_padded[y_start:y_stop, x_start:x_stop]
                        if mask is not None:
                            trim_mask = mask_padded[y_start:y_stop, x_start:x_stop]
                else:
                    if nz > 0:
                        trim_img = image_padded[
                            z_start:z_stop, y_start:y_stop, x_start:x_stop, :
                        ]
                        if mask is not None:
                            trim_mask = mask_padded[
                                z_start:z_stop, y_start:y_stop, x_start:x_stop
                            ]
                    else:
                        trim_img = image_padded[y_start:y_stop, x_start:x_stop, :]
                        if mask is not None:
                            trim_mask = mask_padded[
                                z_start:z_stop, y_start:y_stop, x_start:x_stop
                            ]

                trim_img = np.array(trim_img, dtype=img_dtype)

                if clean_empty and mask is not None:
                    if np.sum(trim_mask) > keep_if:
                        count_save += 1

                        if pixel_size is None:
                            tif.imwrite(
                                join(output, "imgs", img_name),
                                trim_img,
                                shape=trim_img.shape,
                            )
                            tif.imwrite(
                                join(output, "masks", mask_name),
                                np.array(trim_mask, dtype=mask_dtype),
                                shape=trim_mask.shape,
                            )
                        else:
                            to_mrc(trim_img, pixel_size, join(output, "imgs", img_name))
                            to_mrc(
                                np.array(trim_mask, mask_dtype),
                                pixel_size,
                                join(output, "masks", mask_name),
                            )
                else:
                    count_save += 1
                    if mask is None:
                        if pixel_size is None:
                            tif.imwrite(
                                join(output, "imgs", img_name),
                                trim_img,
                                shape=trim_img.shape,
                            )
                        else:
                            to_mrc(trim_img, pixel_size, join(output, "imgs", img_name))

                    else:
                        if pixel_size is None:
                            tif.imwrite(
                                join(output, "imgs", img_name),
                                trim_img,
                                shape=trim_img.shape,
                            )
                            tif.imwrite(
                                join(output, "masks", mask_name),
                                np.array(trim_mask, dtype=mask_dtype),
                                shape=trim_mask.shape,
                            )
                        else:
                            to_mrc(trim_img, pixel_size, join(output, "imgs", img_name))
                            to_mrc(
                                np.array(trim_mask, mask_dtype),
                                pixel_size,
                                join(output, "masks", mask_name),
                            )

    if log:
        return [count, count_save]


def trim_label_mask(
    points: np.ndarray, image: np.ndarray, label_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trim the label mask, image, and adjust the point coordinates based on the minimum and
    maximum boundaries in the provided points. The function calculates the bounding box
    using the points array, crops the image and label mask within this bounding box,
    and shifts the points coordinates accordingly.

    :param points: A numpy array of shape (N, 3) representing the coordinates of points.
    :param image: A 3D numpy array representing the volumetric image.
    :param label_mask: A 3D numpy array representing the volumetric label mask.

    :return: A tuple containing the trimmed image array, the trimmed label_mask array,
        and the adjusted points array.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    max_x, min_x = max(points[:, 0]), min(points[:, 0])
    max_y, min_y = max(points[:, 1]), min(points[:, 1])
    max_z, min_z = max(points[:, 2]), min(points[:, 2])

    if min_z < 0:
        min_z = 0

    image_trim = image[
        int(min_z) : int(max_z), int(min_y) : int(max_y), int(min_x) : int(max_x)
    ]

    label_mask_trim = label_mask[
        int(min_z) : int(max_z), int(min_y) : int(max_y), int(min_x) : int(max_x)
    ]

    points[:, 0] = points[:, 0] - min_x
    points[:, 1] = points[:, 1] - min_y
    points[:, 2] = points[:, 2] - min_z

    return image_trim, label_mask_trim, points
