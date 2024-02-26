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

from tardis_em.cnn.utils.utils import scale_image
from tardis_em.utils.device import get_device
from tardis_em.utils.export_data import to_mrc
from tardis_em.utils.errors import TardisError
from tardis_em.utils.normalization import MeanStdNormalize, RescaleNormalize


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
    Function to patch image and mask to specified patch size with overlying area

    Output images are saved as tiff with naming shame 1_1_1_25. Where
    number indicate grid position in xyz. Last number indicate stride.

    Args:
        image (np.ndarray): Corresponding image for the labels.
        trim_size_xy (int): Size of trimming in xy dimension.
        trim_size_z (int): Size of trimming in z dimension.
        output (str): Name of the output directory for saving.
        image_counter (int): Number id of image.
        scale (tuple): Up- DownScale image and mask to the given shape or factor.
        clean_empty (bool): Remove empty patches.
        keep_if (float): If float, keep only patches that have mask.
            Evaluated based on % of pixels with mask
        stride (int): Trimming step size.
        mask (np.ndarray, None): Label mask.
        log (bool): If True, output trimming log information.
        pixel_size (None, float): If not None, save mask as mrc with pixel size information.
        device (torch.device): Optional device.
    """
    # Normalize histogram
    normalize = RescaleNormalize(clip_range=(1, 99))
    meanstd = MeanStdNormalize()

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
            scale=scale,
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

    # Rescale image intensity
    image = normalize(meanstd(image)).astype(np.float32)
    if not image.min() >= -1 or not image.max() <= 1:  # Image not between in -1 and 1
        if image.min() >= 0 and image.max() <= 1:
            image = (image - 0.5) * 2
        elif image.min() >= 0 and image.max() <= 255:
            image = image / 255  # move to 0 - 1
            image = (image - 0.5) * 2

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

    if trim_size_xy is not None or trim_size_z is not None:
        if not nx >= trim_size_xy:
            TardisError(
                "112",
                "tardis_em/cnn/data_processing.md",
                "trim_size_xy should be equal or greater then X dimension!",
            )
        if not ny >= trim_size_xy:
            TardisError(
                "112",
                "tardis_em/cnn/data_processing.md",
                "trim_size_xy should be equal or greater then Y dimension!",
            )
    else:
        if stride is None:
            TardisError(
                "112",
                "tardis_em/cnn/data_processing.md",
                "Trim sizes or stride has to be indicated!",
            )
        trim_size_xy = 64
        trim_size_z = 64

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
    count = len(range(x)) * len(range(y)) * len(range(z))
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
        return f"{str(count)}|{str(count_save)}"


def trim_label_mask(
    points: np.ndarray, image: np.ndarray, label_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ! DEPRECIATED ! Module to trim image and mask to boundary box of point cloud.

    Args:
        points (np.ndarray): 3D coordinates of pitons.
        image (np.ndarray): corresponding image for the labels.
        label_mask (np.ndarray): empty label mask.
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
