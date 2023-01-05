"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> SpindleTorch - Data_Processing - interpolation

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
from typing import Iterable

import numpy as np

from tardis.utils.errors import TardisError


def interpolate_generator(points: np.ndarray) -> Iterable:
    """

    Args:
        points: Expect array of 2 point in 3D as [X x Y x Z] of [2, 3] shape

    Returns:
        Iterable: Iterable object to generate 3D list of points between given 2
        points
    """
    assert points.shape in ((2, 3), (2, 2)), \
        TardisError('134',
                    'tardis/spindletorch/data_processing/interpolation.py',
                    'Interpolation supports only 2D or 3D for 2 points at a time; '
                    f'But {points.shape} was given!')

    is_3d = False
    if points.ndim == 2:
        is_3d = True

    x0, x1 = points[0, 0], points[1, 0]
    y0, y1 = points[0, 1], points[1, 1]
    if is_3d:  # 3D only
        z0, z1 = points[0, 2], points[1, 2]

    delta_x = x1 - x0
    if delta_x == 0:
        dx_sign = 0
    else:
        dx_sign = int(abs(delta_x) / delta_x)

    delta_y = y1 - y0
    if delta_y == 0:
        dy_sign = 0
    else:
        dy_sign = int(abs(delta_y) / delta_y)

    delta_err_yx = abs(delta_y / delta_x)
    error_xy = 0
    y = y0
    z = z0

    if is_3d:  # 3D only
        delta_z = z1 - z0

        if delta_z == 0:
            dz_sign = 0
        else:
            dz_sign = int(abs(delta_z) / delta_z)

        delta_err_zx = abs(delta_z / delta_x)
        delta_err_zy = abs(delta_z / delta_y)
        delta_err_z = np.min((delta_err_zx, delta_err_zy))
        error_z = 0

    for x in range(x0, x1, dx_sign):
        yield x, y, z

        error_xy = error_xy + delta_err_yx
        while error_xy >= 0.5:
            y += dy_sign
            error_xy -= 1

        if is_3d:  # 3D only
            error_z = error_z + delta_err_z
            while error_z >= 0.5:
                z += dz_sign
                error_z -= 1
            yield x1, y1, z1
        else:
            yield x1, y1


def interpolation(points: np.ndarray) -> np.ndarray:
    """
    3D INTERPOLATION FOR BUILDING SEMANTIC MASK

    Args:
        points (np.ndarray): numpy array with points belonging to individual segments
            given by x, y, (z) coordinates.

    Returns:
        np.ndarray: Interpolated 2 or 3D array
    """
    new_coord = []
    for i in range(0, len(points) - 1):
        """3D interpolation for XYZ dimension"""
        new_coord.append(list(interpolate_generator(points[i:i + 2, :]))[:-1])

    return np.concatenate(new_coord)
