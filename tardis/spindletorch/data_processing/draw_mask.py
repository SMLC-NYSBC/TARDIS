"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> SpindleTorch - Data_Processing - draw_mask

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2023
"""
from math import pow, sqrt
from typing import Tuple

import numpy as np
from skimage import draw

from tardis.utils.errors import TardisError


def draw_circle(r: int,
                c: np.ndarray,
                label_mask: np.ndarray,
                segment_color: int) -> np.ndarray:
    """
    Module draw_label to construct circle shape of a label

    Args:
        r (int): radius of a circle in Angstrom.
        c (np.ndarray): point in 3D indicating center of a circle.
        label_mask (np.ndarray): array of a mask on which circle is drawn.
        segment_color (int): Single digit color value as int.

    Returns:
        np.ndarray: Binary mask.
    """
    assert label_mask.ndim in [2, 3], \
        TardisError('113',
                    f'Unsupported dimensions given {label_mask.ndim} expected 2!')

    if label_mask.ndim == 3:
        _, ny, nx = label_mask.shape
    else:
        ny, nx = label_mask.shape
    x = int(c[0])
    y = int(c[1])
    if label_mask.ndim == 3:
        z = int(c[2])

    cy, cx = draw.disk((y, x), r, shape=(ny, nx))

    if label_mask.ndim == 3:
        label_mask[z, cy, cx] = segment_color
    else:
        label_mask[cy, cx] = segment_color

    return label_mask


def draw_3d(r: int,
            c: np.ndarray,
            label_mask: np.ndarray,
            segment_color: int) -> np.ndarray:
    """
    Module draw_label to construct sphere shape of a label

    Args:
        r (int): radius of a circle in Angstrom.
        c (np.ndarray): point in 3D indicating center of a circle.
        label_mask (np.ndarray): array of a mask on which circle is drawn.
        segment_color (int): Single digit color value as int.

    Returns:
        np.ndarray: Binary mask.
    """
    assert label_mask.ndim == 3, \
        TardisError('113',
                    'tardis/spindletorch/data_processing/draw_mask.py'
                    f'Unsupported dimensions given {label_mask.ndim} expected 2!')

    nz, ny, nx = label_mask.shape
    x = int(c[0])
    y = int(c[1])
    z = int(c[2])

    cz, cy, cx = draw_sphere(r=r,
                             c=(z, y, x),
                             shape=label_mask.shape)

    label_mask[cz, cy, cx] = segment_color

    return label_mask


def draw_sphere(r: int,
                c: tuple,
                shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        r (int): radius of a circle in Angstrom.
        c (tuple): point in 3D indicating center of a circle [Z, Y, X].
        shape (tuple): Shape of mask to eliminated ofe-flowed pixel
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of array's with zyx coordinates
    """
    r_dim = round(r * 3)
    sphere_frame = np.zeros((r_dim, r_dim, r_dim), dtype=np.int8)

    c_frame = round(r)

    z, y, x = sphere_frame.shape
    for z_dim in range(z):
        for y_dim in range(y):
            for x_dim in range(x):
                dist_zyx_to_c = sqrt(pow(abs(z_dim - c_frame), 2) +
                                     pow(abs(y_dim - c_frame), 2) +
                                     pow(abs(x_dim - c_frame), 2))
                if dist_zyx_to_c > 5:
                    sphere_frame[z_dim, y_dim, x_dim] = False
                else:
                    sphere_frame[z_dim, y_dim, x_dim] = True
    z, y, x = np.where(sphere_frame)

    c = ((c[0] - c_frame), (c[1] - c_frame), (c[2] - c_frame))
    z, y, x = z + c[0], y + c[1], x + c[2]

    # Remove pixel out of frame
    zyx = np.array((z, y, x)).T
    del_id = []
    for id, i in enumerate(zyx):
        if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
            del_id.append(id)

    zyx = np.delete(zyx, del_id, 0)

    return zyx[:, 0], zyx[:, 1], zyx[:, 2]
