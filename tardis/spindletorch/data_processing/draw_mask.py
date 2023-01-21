#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from math import pow, sqrt
from typing import Tuple, Union

import numpy as np
from skimage import draw

from tardis.utils.errors import TardisError


def draw_mask(r: int,
              c: np.ndarray,
              label_mask: np.ndarray,
              segment_shape: str) -> np.ndarray:
    """
    Module draw_label to construct sphere shape of a label

    Args:
        r (int): radius of a circle in Angstrom.
        c (np.ndarray): point in 3D indicating center of a circle [X x Y x Z].
        label_mask (np.ndarray): array of a mask on which circle is drawn.
        segment_shape (str): Type of shape to draw. Expect ['s', 'c'].

    Returns:
        np.ndarray: Binary mask.
    """
    assert label_mask.ndim == 3, \
        TardisError('113',
                    'tardis/spindletorch/data_processing/draw_mask.py'
                    f'Unsupported dimensions given {label_mask.ndim} expected 2!')

    x = int(c[0])
    y = int(c[1])
    if len(c) == 3:
        z = int(c[2])
    else:
        z = None

    assert segment_shape in ['s', 'c'], \
        TardisError('123',
                    'tardis/spindletorch/data_processing/draw_mask.py'
                    f'Unsupported shape type to draw given {segment_shape} but '
                    'expected "c" - circle or "s" - sphere')
    if segment_shape == 's':
        cz, cy, cx = draw_sphere(r=r,
                                 c=(z, y, x),
                                 shape=label_mask.shape)
        label_mask[cz, cy, cx] = 1
    else:
        if z is not None:
            cz, cy, cx = draw_circle(r=r,
                                     c=(z, y, x),
                                     shape=label_mask.shape)
            label_mask[cz, cy, cx] = 1
        else:
            cy, cx = draw_circle(r=r,
                                 c=(z, y, x),
                                 shape=label_mask.shape)
            label_mask[cy, cx] = 1

    return label_mask


def draw_circle(r: int,
                c: tuple,
                shape: tuple) -> Union[Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Draw a circle and shit coordinate to c position.

    Args:
        r (int): radius of a circle in Angstrom.
        c (tuple): point in 3D indicating center of a circle [(Z), Y, X].
        shape (tuple): Shape of mask to eliminated ofe-flowed pixel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of array's with zyx coordinates
    """
    r_dim = round(r * 3)
    c_frame = round(r)

    ny, nx = draw.disk((r, r), r, shape=(r_dim, r_dim))
    if len(c) == 3:
        nz = np.repeat(c[0], len(nx))

        c = ((c[1] - c_frame), (c[2] - c_frame))
        z, y, x = nz, ny + c[0], nx + c[1]

        # Remove pixel out of frame
        zyx = np.array((z, y, x)).T
        del_id = []
        for id, i in enumerate(zyx):
            if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
                del_id.append(id)

        zyx = np.delete(zyx, del_id, 0)

        return zyx[:, 0], zyx[:, 1], zyx[:, 2]
    else:
        c = ((c[0] - c_frame), (c[1] - c_frame))
        y, x = ny + c[0], nx + c[1]

        # Remove pixel out of frame
        yx = np.array((y, x)).T
        del_id = []
        for id, i in enumerate(yx):
            if i[0] >= shape[0] or i[1] >= shape[1]:
                del_id.append(id)

        yx = np.delete(yx, del_id, 0)

        return yx[:, 0], yx[:, 1]


def draw_sphere(r: int,
                c: tuple,
                shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw a sphere and shit coordinate to c position.

    Args:
        r (int): radius of a sphere in Angstrom.
        c (tuple): point in 3D indicating center of a sphere [Z, Y, X].
        shape (tuple): Shape of mask to eliminated ofe-flowed pixel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of array's with zyx coordinates
    """
    r_dim = round(r * 2)
    sphere_frame = np.zeros((r_dim, r_dim, r_dim), dtype=np.int8)
    trim = round(r_dim / 6)
    c_frame = round(r)

    z, y, x = sphere_frame.shape
    for z_dim in range(z):
        for y_dim in range(y):
            for x_dim in range(x):
                dist_zyx_to_c = sqrt(pow(abs(z_dim - c_frame), 2) +
                                     pow(abs(y_dim - c_frame), 2) +
                                     pow(abs(x_dim - c_frame), 2))
                if dist_zyx_to_c > c_frame:
                    sphere_frame[z_dim, y_dim, x_dim] = False
                else:
                    sphere_frame[z_dim, y_dim, x_dim] = True
    sphere_frame[:trim, :] = False  # Trim bottom of the sphere
    sphere_frame[-trim:, :] = False  # Trim top of the sphere
    z, y, x = np.where(sphere_frame)

    c = ((c[0] - c_frame), (c[1] - c_frame), (c[2] - c_frame))
    z, y, x = z + c[0], y + c[1], x + c[2]

    # Remove pixel out of frame
    zyx = np.array((z, y, x)).T
    del_id = []
    for id, i in enumerate(zyx):
        if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
            del_id.append(id)
        if i[0] < 0:  # Remove negative value
            del_id.append(id)

    zyx = np.delete(zyx, del_id, 0)

    return zyx[:, 0], zyx[:, 1], zyx[:, 2]
