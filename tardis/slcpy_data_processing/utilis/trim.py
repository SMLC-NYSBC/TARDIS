import math
from os.path import join
from typing import Optional

import numpy as np
from tifffile import tifffile


def trim_image(image: np.ndarray,
               trim_size_xy: int,
               output: str,
               image_counter: int,
               clean_empty=False,
               trim_size_z=0,
               prefix=''):
    """
    MODULE FOR TRIMMING DATA TO SPECIFIED SIZES

    Args:
        image: Image for triming of a shape [Z,Y,X,C]/[Z,Y,X]/[Y,X,C]/[Y,X],
            where C==3 indicate number of channels (RGB)
        label_mask: Empty label mask or None if image mask is not included
        trim_size_xy: Size of trimming in xy dimension
        trim_size_z: Size of trimming in z dimension
        output: Name of the output directory for saving
        image_counter: Number id of image
        clean_empty: Omit saving images with all values at 0 aka empty
        prefix: Prefix name added at the end of each trimmed file
    """
    # Initial set-up
    idx = image_counter + 1

    if image.ndim == 4:  # 3D with RGB
        nz, ny, nx, nc = image.shape
        dim = 3
    elif image.ndim == 3 and image.shape[2] == 3:  # 2D with RGB
        ny, nx, nc = image.shape
        nz = 0
        dim = 2
    elif image.ndim == 3 and image.shape[2] != 3:  # 3D with Gray
        nz, ny, nx = image.shape
        nc = None
        dim = 3
    else:  # 2D with Gray
        ny, nx = image.shape
        nc = None
        nz = 0
        dim = 2

    # Count number of trimming in z, y and x with padding
    x_axis, y_axis, z_axis = nx // trim_size_xy, ny // trim_size_xy, nz // trim_size_z
    if x_axis < nx / trim_size_xy:  # Calculate padding
        x_axis += 1
    if y_axis < ny / trim_size_xy:  # Calculate padding
        y_axis += 1
    if nz == 0:  # Hardfix for 2D data when nz == 0
        z_axis = 1
    if z_axis < nz / trim_size_z:  # Calculate padding
        z_axis += 1

    # Zero-out Y axis counter
    ny_start, ny_end = -trim_size_xy, 0
    print(x_axis, y_axis, z_axis)
    # Trimming throw Y axis
    for _ in range(y_axis):
        ny_start += trim_size_xy
        ny_end += trim_size_xy
        nx_start, nx_end = -trim_size_xy, 0  # Zero-out X axis counter

        # Trimming throw X axis
        for _ in range(x_axis):
            nx_start += trim_size_xy
            nx_end += trim_size_xy
            nz_start, nz_end = -trim_size_z, 0  # Zero-out Z axis counter

            # Trimming throw Z axis
            for _ in range(z_axis):
                nz_start += trim_size_z
                nz_end += trim_size_z

                if dim == 3:
                    if nc is not None:
                        trim_image = empty_mask((trim_size_z,
                                                 trim_size_xy,
                                                 trim_size_xy,
                                                 3))
                        trim_df = image[nz_start:nz_end,
                                        ny_start:ny_end,
                                        nx_start:nx_end,
                                        :]

                        trim_image[0:trim_df.shape[0],
                                   0:trim_df.shape[1],
                                   0:trim_df.shape[2],
                                   :] = trim_df
                    else:
                        trim_image = empty_mask((trim_size_z,
                                                trim_size_xy,
                                                trim_size_xy))
                        trim_df = image[nz_start:nz_end,
                                        ny_start:ny_end,
                                        nx_start:nx_end]

                        trim_image[0:trim_df.shape[0],
                                   0:trim_df.shape[1],
                                   0:trim_df.shape[2]] = trim_df
                elif dim == 2:
                    if nc is not None:
                        trim_image = empty_mask((trim_size_xy,
                                                trim_size_xy,
                                                3))
                        trim_df = image[ny_start:ny_end,
                                        nx_start:nx_end,
                                        :]
                        
                        trim_image[0:trim_df.shape[0],
                                   0:trim_df.shape[1],
                                   :] = trim_df
                    else:
                        trim_image = empty_mask((trim_size_xy,
                                                 trim_size_xy))
                        trim_df = image[ny_start:ny_end,
                                           nx_start:nx_end]
                        
                        trim_image[0:trim_df.shape[0],
                                   0:trim_df.shape[1]] = trim_df

                img_name = str(idx) + prefix + '.tif'

                if not clean_empty and not np.all(trim_image == 0):
                    # Hard transform between int8 and uint8
                    if np.min(trim_image) < 0:
                        trim_image = trim_image + 128

                    tifffile.imwrite(join(output, img_name),
                                     np.array(trim_image, 'int8'))
                    idx += 1

    return idx


def empty_mask(size: tuple):
    return np.zeros((size))


def trim_to_patches(image: np.ndarray,
                    trim_size_xy: int,
                    trim_size_z: int,
                    multi_layer: bool,
                    output: str,
                    image_counter: int,
                    label_mask: Optional[np.ndarray] = None,
                    stride=25):
    """
    FUNCTION TO TRIMMED IMAGE AND MASKS TO SPECIFIED SIZE

    Output images are saved as tiff with naming shame 1_1_1_25. Where
    number indicate grid position in xyz. Last number indicate stride.

    Args:
        image: Corresponding image for the labels
        label_mask: Empty label mask
        trim_size_xy: Size of trimming in xy dimension
        trim_size_z: Size of trimming in z dimension
        multi_layer: Single, or unique value for each lines
        output: Name of the output directory for saving
        image_counter:
        stride: Trimming step size

    Returns:
        Saved trimmed images as tiff in specified folder
    """
    idx = image_counter

    if multi_layer:
        nz, ny, nx, nc = label_mask.shape
    elif multi_layer is False and label_mask is not None:
        nz, ny, nx = label_mask.shape
        nc = None
    else:
        nz, ny, nx = image.shape
        nc = None

    if trim_size_xy is not None or trim_size_z is not None:
        assert nx >= trim_size_xy, \
            "trim_size_xy should be equal or greater then X dimension!"
        assert ny >= trim_size_xy, \
            "trim_size_xy should be equal or greater then Y dimension!"
    else:
        assert stride is not None, \
            "Trim sizes or stride has to be indicated!"
        trim_size_xy = 64
        trim_size_z = 64

    # Calculate number of patches, patch sizes, and stride for xyz
    x, y, z = math.ceil(nx / trim_size_xy), \
        math.ceil(ny / trim_size_xy), \
        math.ceil(nz / trim_size_z)

    x_padding, y_padding, z_padding = (trim_size_xy + ((trim_size_xy - stride) * (x - 1))) - nx, \
                                      (trim_size_xy + ((trim_size_xy - stride) * (y - 1))) - ny, \
                                      (trim_size_z +
                                       ((trim_size_z - stride) * (z - 1))) - nz

    # Adapt number of patches for trimming
    if trim_size_xy is not None or trim_size_z is not None:
        while x_padding < 0:
            x += 1
            x_padding += trim_size_xy - stride
        while y_padding < 0:
            y += 1
            y_padding += trim_size_xy - stride
        while z_padding < 0:
            z += 1
            z_padding += trim_size_z - stride

    # Adapt patch size for trimming
    else:
        while x_padding <= 0 or y_padding <= 0:
            trim_size_xy += 1
            x_padding = (trim_size_xy +
                         ((trim_size_xy - stride) * (x - 1))) - nx
            y_padding = (trim_size_xy +
                         ((trim_size_xy - stride) * (y - 1))) - ny

        while z_padding < 0:
            trim_size_z += 1
            z_padding = (trim_size_z + ((trim_size_z - stride) * (z - 1))) - nz

    # Expand image of a patch
    image_padded = np.pad(image,
                          [(0, z_padding), (0, y_padding), (0, x_padding)],
                          mode='constant')

    if label_mask is not None:
        if nc is None:
            mask_padded = np.pad(label_mask,
                                 [(0, z_padding), (0, y_padding), (0, x_padding)],
                                 mode='constant')
        else:
            mask_padded = np.pad(label_mask,
                                 [(0, z_padding), (0, y_padding),
                                  (0, x_padding), (0, 0)],
                                 mode='constant')

    # Trim image and mask with stride
    z_start, z_stop = 0 - (trim_size_z - stride), 0

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

                img_name = str("{}_{}_{}_{}_{}.tif".format(
                    idx, k, j, i, stride))
                mask_name = str(
                    "{}_{}_{}_{}_{}_mask.tif".format(idx, k, j, i, stride))

                trim_img = image_padded[z_start:z_stop,
                                        y_start:y_stop,
                                        x_start:x_stop]

                if label_mask is not None:
                    if nc is None:
                        trim_mk = mask_padded[z_start:z_stop,
                                              y_start:y_stop,
                                              x_start:x_stop]
                    else:
                        trim_mk = mask_padded[z_start:z_stop,
                                              y_start:y_stop,
                                              x_start:x_stop,
                                              :]
                    tifffile.imwrite(join(output, 'mask', mask_name),
                                     np.array(trim_mk, 'int8'))

                if label_mask is not None:
                    tifffile.imwrite(join(output, 'imgs', img_name),
                                     np.array(trim_img, 'int8'))
                else:
                    tifffile.imwrite(join(output, img_name),
                                     np.array(trim_img, 'int8'))
    idx += 1
    return idx


def trim_label_mask(points: np.ndarray,
                    image: np.ndarray,
                    label_mask: np.ndarray):
    """
    MODULE TO TRIM CREATED IMAGES AND MASK

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
