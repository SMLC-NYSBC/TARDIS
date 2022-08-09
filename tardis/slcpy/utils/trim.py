from math import ceil
from os.path import join
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tifffile import tifffile as tif


def trim_image(image: np.ndarray,
               trim_size_xy: int,
               output: str,
               image_counter: int,
               clean_empty=False,
               trim_size_z=0,
               prefix=''):
    """
    MODULE FOR TRIMMING DATA TO SPECIFIED SIZES

    Output images are saved as tiff with naming shame 1_1_1_25. Referring to:
        Number of image, z, y, x and stride

    Args:
        image: Image for trimming of a shape [Z,Y,X,C]/[Z,Y,X]/[Y,X,C]/[Y,X],
            where C==3 indicate number of channels (RGB)
        label_mask: Empty label mask or None if image mask is not included
        trim_size_xy: Size of trimming in xy dimension
        trim_size_z: Size of trimming in z dimension
        output: Name of the output directory for saving
        image_counter: Number id of image
        clean_empty: Omit saving images with all values at 0 aka empty
        prefix: Prefix name added at the end of each trimmed file
    """
    if image.ndim == 4 and image.shape[3] == 3:  # 3D with RGB
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

    """Count number of trimming in z, y and x with padding"""
    x_axis, y_axis, z_axis = ceil(nx / trim_size_xy), ceil(ny / trim_size_xy), \
        ceil(nz / trim_size_z)
    if nz == 0:  # Hard-fix for 2D data when nz == 0
        z_axis = 1

    """Zero-out Z axis counter"""
    nz_start, nz_end = -trim_size_z, 0

    """Trimming throw Z axis"""
    for z in range(z_axis):
        nz_start += trim_size_z
        nz_end += trim_size_z
        ny_start, ny_end = -trim_size_xy, 0  # Zero-out Y axis counter

        """Trimming throw Y axis"""
        for y in range(y_axis):
            ny_start += trim_size_xy
            ny_end += trim_size_xy
            nx_start, nx_end = -trim_size_xy, 0  # Zero-out X axis counter

            """Trimming throw Z axis"""
            for x in range(x_axis):
                nx_start += trim_size_xy
                nx_end += trim_size_xy

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

                """0 refers to stride == 0"""
                img_name = str(f'{image_counter}_{z}_{y}_{x}_0{prefix}.tif')

                if clean_empty:
                    if not np.all(trim_image == 0):
                        """Hard transform between int8 and uint8"""
                        if np.min(trim_image) < 0:
                            trim_image = trim_image + 128

                        tif.imwrite(join(output, img_name),
                                    np.array(trim_image, 'uint8'),
                                    shape=trim_image.shape)
                else:
                    tif.imwrite(join(output, img_name),
                                np.array(trim_image, 'uint8'),
                                shape=trim_image.shape)


def empty_mask(size: tuple):
    return np.zeros((size))


def scale_image(image: np.ndarray,
                scale: float,
                mask: Optional[np.ndarray] = None):
    """
    Scale image module using torch GPU interpolation

    Args:
        image: image data
        mask: Optional binary mask image data
        scale: scale value for image
    """
    if scale == 1:
        if image.ndim == 4:  # 3D with RGB
            dim = 3
        elif image.ndim == 3 and image.shape[2] == 3:  # 2D with RGB
            dim = 3
        elif image.ndim == 3 and image.shape[2] != 3:  # 3D with Gray
            dim = 1
        else:  # 2D with Gray
            dim = 1

        if mask is not None:
            return image, mask, dim
        else:
            return image, dim

    if image.ndim == 4:  # 3D with RGB
        dim = 3

        image = np.transpose(F.interpolate(torch.Tensor(np.transpose(image, (3, 0, 1, 2)))[None, :],
                             scale_factor=scale,
                             mode='trilinear',
                             align_corners=False).cpu().detach().numpy()[0, :],
                             (1, 2, 3, 0))
        if mask is not None:
            mask = np.transpose(F.interpolate(torch.Tensor(np.transpose(mask, (3, 0, 1, 2)))[None, :],
                                scale_factor=scale,
                                mode='trilinear',
                                align_corners=False).cpu().detach().numpy()[0, :],
                                (1, 2, 3, 0))
    elif image.ndim == 3 and image.shape[2] == 3:  # 2D with RGB
        dim = 3

        image = np.transpose(F.interpolate(torch.Tensor(np.transpose(image, (2, 0, 1)))[None, :],
                             scale_factor=scale,
                             mode='bicubic',
                             align_corners=False).cpu().detach().numpy()[0, :],
                             (1, 2, 0))
        if mask is not None:
            mask = np.transpose(F.interpolate(torch.Tensor(np.transpose(mask, (2, 0, 1)))[None, :],
                                scale_factor=scale,
                                mode='bicubic',
                                align_corners=False).cpu().detach().numpy()[0, :],
                                (1, 2, 0))
    elif image.ndim == 3 and image.shape[2] != 3:  # 3D with Gray
        dim = 1

        image = F.interpolate(torch.Tensor(image)[None, None, :],
                              scale_factor=scale,
                              mode='trilinear',
                              align_corners=False).cpu().detach().numpy()[0, 0, :]
        if mask is not None:
            mask = F.interpolate(torch.Tensor(mask)[None, None, :],
                                 scale_factor=scale,
                                 mode='trilinear',
                                 align_corners=False).cpu().detach().numpy()[0, 0, :]
    else:  # 2D with Gray
        dim = 1

        image = F.interpolate(torch.Tensor(image)[None, None, :],
                              scale_factor=scale,
                              mode='bicubic',
                              align_corners=False).cpu().detach().numpy()[0, 0, :]
        if mask is not None:
            mask = F.interpolate(torch.Tensor(mask)[None, None, :],
                                 scale_factor=scale,
                                 mode='bicubic',
                                 align_corners=False).cpu().detach().numpy()[0, 0, :]

    if mask is not None:
        return image, np.where(mask > 0.5, 1, 0), dim
    else:
        return image, dim


def trim_with_stride(image: np.ndarray,
                     trim_size_xy: int,
                     trim_size_z: int,
                     output: str,
                     image_counter: int,
                     prefix='',
                     clean_empty=True,
                     stride=25,
                     scale: Optional[float] = 1.0,
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
    if mask is not None:
        assert image.shape == mask.shape, \
            f'Image {image.shape} has different shape from mask {mask.shape}'
        image, mask, dim = scale_image(image=image,
                                       mask=mask,
                                       scale=scale)
    else:
        image, dim = scale_image(image=image,
                                 scale=scale)

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
    min_px_count = min_px_count * 0.0005  # 0.1% of mask must be occupied 

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

                if clean_empty and mask is not None:
                    if not np.sum(trim_mask) > min_px_count:
                        """Hard transform between int8 and uint8"""
                        if np.min(trim_img) < 0:
                            trim_img = trim_img + 128

                        tif.imwrite(join(output, 'imgs', img_name),
                                    np.array(trim_img, 'uint8'),
                                    shape=trim_img.shape)
                        tif.imwrite(join(output, 'masks', f'{img_name[:-4]}_mask.tif'),
                                    np.array(trim_mask, 'uint8'),
                                    shape=trim_mask.shape)
                else:
                    if mask is None:
                        tif.imwrite(join(output, img_name),
                                    np.array(trim_img, 'uint8'),
                                    shape=trim_img.shape)

                    else:
                        tif.imwrite(join(output, 'imgs', img_name),
                                    np.array(trim_img, 'uint8'),
                                    shape=trim_img.shape)
                        tif.imwrite(join(output, 'masks', f'{img_name[:-4]}_mask.tif'),
                                    np.array(trim_mask, 'uint8'),
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
