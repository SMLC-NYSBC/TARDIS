from math import ceil
from os.path import join
from typing import Optional

import numpy as np
from tifffile import tifffile as tif

from tardis.spindletorch.data_processing.semantic_mask import fill_gaps_in_semantic
from tardis.spindletorch.utils.utils import scale_image
from tardis.utils.errors import TardisError


def trim_with_stride(image: np.ndarray,
                     trim_size_xy: int,
                     trim_size_z: int,
                     output: str,
                     image_counter: int,
                     prefix='',
                     clean_empty=True,
                     stride=25,
                     scale: Optional[tuple] = 1.0,
                     mask: Optional[np.ndarray] = None):
    """
    FUNCTION TO TRIMMED IMAGE AND MASKS TO SPECIFIED SIZE

    Output images are saved as tiff with naming shame 1_1_1_25. Where
    number indicate grid position in xyz. Last number indicate stride.

    Args:
        image: Corresponding image for the labels
        trim_size_xy: Size of trimming in xy dimension
        trim_size_z: Size of trimming in z dimension
        output: Name of the output directory for saving
        image_counter: Number id of image
        prefix: Prefix name at the end of the file
        clean_empty: Remove empty patches
        stride: Trimming step size
        mask: Label mask
    """
    img_dtype = np.float32

    if mask is not None:
        mask_dtype = np.uint8
        assert image.shape == mask.shape, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        f'Image {image.shape} has different shape from mask {mask.shape}')

        image, mask, dim = scale_image(image=image,
                                       mask=mask,
                                       scale=scale)

        mask = fill_gaps_in_semantic(mask)
        mask = mask.astype(np.uint8)
    else:
        image, dim = scale_image(image=image,
                                 scale=scale)

    assert img_dtype == image.dtype, f'{img_dtype} != {image.dtype}'

    if image.ndim == 4 and dim == 3:  # 3D RGB
        nz, ny, nx, nc = image.shape
        min_px_count = trim_size_xy * trim_size_xy * trim_size_z
    elif image.ndim == 3 and dim == 3:  # 2D RGB
        ny, nx, nc = image.shape
        nz = 0  # 2D
        min_px_count = trim_size_xy * trim_size_xy
    elif image.ndim == 3 and dim == 1:  # 3D gray
        nz, ny, nx = image.shape

        nc = None  # Gray
        min_px_count = trim_size_xy * trim_size_xy * trim_size_z
    elif image.ndim == 2:  # 2D gray
        ny, nx = image.shape
        nc = None  # 2D
        nz = 0  # Gray
        min_px_count = trim_size_xy * trim_size_xy
    min_px_count = min_px_count * 0.005  # 0.001% of pixels must be occupied

    if trim_size_xy is not None or trim_size_z is not None:
        assert nx >= trim_size_xy, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        "trim_size_xy should be equal or greater then X dimension!")
        assert ny >= trim_size_xy, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        "trim_size_xy should be equal or greater then Y dimension!")
    else:
        assert stride is not None, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        "Trim sizes or stride has to be indicated!")
        trim_size_xy = 64
        trim_size_z = 64

    """Calculate number of patches and stride for xyz"""
    x, y, z = ceil(nx / trim_size_xy), ceil(ny / trim_size_xy), ceil(nz / trim_size_z)

    if nz > 0:
        x_pad, y_pad, z_pad = (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx, \
            (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny, \
            (trim_size_z + (trim_size_z - stride) * (z - 1)) - nz
    else:
        x_pad, y_pad, z_pad = (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx, \
            (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny, \
            0

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
            image_padded = np.pad(image,
                                  [(0, z_pad), (0, y_pad), (0, x_pad), (0, 0)],
                                  mode='constant')
        else:
            image_padded = np.pad(image,
                                  [(0, z_pad), (0, y_pad), (0, x_pad)],
                                  mode='constant')

        if mask is not None:
            mask_padded = np.pad(mask,
                                 [(0, z_pad), (0, y_pad), (0, x_pad)],
                                 mode='constant')
    else:
        if nc is not None:
            image_padded = np.pad(image,
                                  [(0, y_pad), (0, x_pad), (0, 0)],
                                  mode='constant')
        else:
            image_padded = np.pad(image,
                                  [(0, y_pad), (0, x_pad)],
                                  mode='constant')

            if mask is not None:
                mask_padded = np.pad(mask,
                                     [(0, y_pad), (0, x_pad)],
                                     mode='constant')

    """Trim image and mask with stride"""
    z_start, z_stop = 0 - (trim_size_z - stride), 0
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

                img_name = str(
                    f'{image_counter}_{i}_{j}_{k}_{stride}{prefix}.tif')

                if nc is None:
                    if nz > 0:
                        trim_img = image_padded[z_start:z_stop,
                                                y_start:y_stop,
                                                x_start:x_stop]
                        if mask is not None:
                            trim_mask = mask_padded[z_start:z_stop,
                                                    y_start:y_stop,
                                                    x_start:x_stop]
                    else:
                        trim_img = image_padded[y_start:y_stop,
                                                x_start:x_stop]
                        if mask is not None:
                            trim_mask = mask_padded[y_start:y_stop,
                                                    x_start:x_stop]
                else:
                    if nz > 0:
                        trim_img = image_padded[z_start:z_stop,
                                                y_start:y_stop,
                                                x_start:x_stop,
                                                :]
                        if mask is not None:
                            trim_mask = mask_padded[z_start:z_stop,
                                                    y_start:y_stop,
                                                    x_start:x_stop]
                    else:
                        trim_img = image_padded[y_start:y_stop,
                                                x_start:x_stop,
                                                :]
                        if mask is not None:
                            trim_mask = mask_padded[z_start:z_stop,
                                                    y_start:y_stop,
                                                    x_start:x_stop]

                trim_img = np.array(trim_img, dtype=img_dtype)

                if clean_empty and mask is not None:
                    if np.sum(trim_mask) > min_px_count:
                        tif.imwrite(join(output, 'imgs', img_name),
                                    trim_img,
                                    shape=trim_img.shape)
                        tif.imwrite(join(output, 'masks', f'{img_name[:-4]}_mask.tif'),
                                    np.array(trim_mask, dtype=mask_dtype),
                                    shape=trim_mask.shape)
                else:
                    if mask is None:
                        tif.imwrite(join(output, img_name),
                                    trim_img,
                                    shape=trim_img.shape)

                    else:
                        tif.imwrite(join(output, 'imgs', dtype=img_name),
                                    trim_img,
                                    shape=trim_img.shape)
                        tif.imwrite(join(output, 'masks', f'{img_name[:-4]}_mask.tif'),
                                    np.array(trim_mask, dtype=mask_dtype),
                                    shape=trim_mask.shape)


def trim_label_mask(points: np.ndarray,
                    image: np.ndarray,
                    label_mask: np.ndarray):
    """
    MODULE TO TRIM CREATED IMAGES AND MASK TO BOUNDARY BOX OF POINT CLOUD

    Args:
        points: 3D coordinates of pitons
        image: corresponding image for the labels
        label_mask: empty label mask
    """
    max_x, min_x = max(points[:, 0]), min(points[:, 0])
    max_y, min_y = max(points[:, 1]), min(points[:, 1])
    max_z, min_z = max(points[:, 2]), min(points[:, 2])

    if min_z < 0:
        min_z = 0

    image_trim = image[int(min_z):int(max_z),
                       int(min_y):int(max_y),
                       int(min_x):int(max_x)]

    label_mask_trim = label_mask[int(min_z):int(max_z),
                                 int(min_y):int(max_y),
                                 int(min_x):int(max_x)]

    points[:, 0] = points[:, 0] - min_x
    points[:, 1] = points[:, 1] - min_y
    points[:, 2] = points[:, 2] - min_z

    return image_trim, label_mask_trim, points
