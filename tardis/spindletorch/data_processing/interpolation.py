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
        points as [X x Y x (Z)]
    """
    assert points.shape == (2, 3), \
        TardisError('134',
                    'tardis/spindletorch/data_processing/interpolation.py',
                    'Interpolation supports only 3D for 2 points at a time; '
                    f'But {points.shape} was given!')

    if points.dtype not in [np.uint8, np.int8, np.uint16, np.int16]:
        points = np.round(points).astype(np.int16)

    is_3d = False
    if points.ndim == 2:
        is_3d = True

    # Collect first and last point in array for XYZ
    x0, x1 = points[0, 0], points[1, 0]
    y0, y1 = points[0, 1], points[1, 1]
    z0, z1 = points[0, 2], points[1, 2]

    # Delta between first and last point to interpolate
    delta_x = x1 - x0
    delta_y = y1 - y0
    delta_z = z1 - z0

    # Calculate axis to iterate throw
    max_delta = np.where((abs(delta_x), abs(delta_y), abs(delta_z)) ==
                         np.max((abs(delta_x), abs(delta_y), abs(delta_z))))[0][0]

    # Calculate scaling step for XYZ
    if delta_x == 0:
        dx_sign = 0
    else:
        dx_sign = int(abs(delta_x) / delta_x)

    if delta_y == 0:
        dy_sign = 0
    else:
        dy_sign = int(abs(delta_y) / delta_y)

    if delta_z == 0:
        dz_sign = 0
    else:
        dz_sign = int(abs(delta_z) / delta_z)

    x = x0
    y = y0
    z = z0

    # Calculating scaling threshold
    if delta_x == 0:  # Threshold for X axis
        delta_err_x = 0.0
        if delta_z != 0:
            delta_err_y = abs(delta_y / delta_z)
        else:
            delta_err_y = 0.5
    elif delta_y == 0:  # Threshold for Y axis
        delta_err_y = 0.0
        if delta_z != 0:
            delta_err_x = abs(delta_x / delta_z)
        else:
            delta_err_x = 0.5
    else:
        delta_err_x = abs(delta_x / delta_y)
        delta_err_y = abs(delta_y / delta_x)

    if delta_z == 0:  # Threshold for Z axis
        delta_err_z = 0.0
    else:
        if delta_x != 0 and delta_y != 0:
            delta_err_z = np.min((abs(delta_z / delta_x), abs(delta_z / delta_y)))
        else:
            if delta_x == 0:
                if delta_y != 0:
                    delta_err_z = (abs(delta_z / delta_y))
                else:
                    delta_err_z = 0.0
            else:
                if delta_x != 0:
                    delta_err_z = (abs(delta_z / delta_x))
                else:
                    delta_err_z = 0.0

    # Zero out threshold
    error_x = 0
    error_y = 0
    error_z = 0

    if max_delta == 0:  # Scale XYZ by iterating throw X axis
        for x in range(x0, x1, dx_sign):
            yield x, y, z

            # Iteratively add and scale Y axis
            error_y = error_y + delta_err_y
            while error_y >= 0.5:
                y += dy_sign
                error_y -= 1

            # Iteratively add and scale Z axis
            error_z = error_z + delta_err_z
            while error_z >= 0.5:
                z += dz_sign
                error_z -= 1
    if max_delta == 1:  # Scale XYZ by iterating throw Y axis
        for y in range(y0, y1, dy_sign):
            yield x, y, z

            # Iteratively add and scale X axis
            error_x = error_x + delta_err_x
            while error_x >= 0.5:
                x += dx_sign
                error_x -= 1

            # Iteratively add and scale Z axis
            error_z = error_z + delta_err_z
            while error_z >= 0.5:
                z += dz_sign
                error_z -= 1
    if max_delta == 2:  # Scale XYZ by iterating throw Z axis
        for z in range(z0, z1, dz_sign):
            yield x, y, z


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

        new_coord.append(list(interpolate_generator(points[i:i + 2, :])))

    return np.concatenate(new_coord)
