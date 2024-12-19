#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Iterable

import numpy as np

from tardis_em.utils.errors import TardisError


def interpolate_generator(points: np.ndarray) -> Iterable:
    """
    Generates interpolated points between two given 2D or 3D points. The function
    determines the interpolation path in a pixel-grid-like manner between the start
    and end points, based on Bresenham-like algorithms. Interpolation supports
    only 2D and 3D coordinate systems, and exactly two points must be provided for
    interpolation.

    :param points: An array of shape (2, 2) for 2D points or (2, 3) for 3D points.
                   Defines the two points between which interpolation is performed.
    :return: An iterable generator that yields tuples representing (x, y) in 2D
             space and (x, y, z) in 3D space.
    """
    if points.shape not in [(2, 3), (2, 2)]:
        TardisError(
            "134",
            "tardis_em/cnn/data_processing.md/interpolation.py",
            "Interpolation supports only 2D/3D for 2 points at a time; "
            f"But {points.shape} was given!",
        )

    points = np.round(points).astype(np.int32)
    if points.shape == (2, 2):
        dim_ = 2
    else:
        dim_ = 3

    # Collect first and last point in array for XYZ
    x0, x1 = points[0, 0], points[1, 0]
    y0, y1 = points[0, 1], points[1, 1]
    if dim_ == 2:
        z0, z1 = 0, 0
    else:
        z0, z1 = points[0, 2], points[1, 2]

    # Delta between first and last point to interpolate
    delta_x, delta_y, delta_z = x1 - x0, y1 - y0, z1 - z0

    # Calculate axis to iterate throw
    max_delta = np.where(
        (abs(delta_x), abs(delta_y), abs(delta_z))
        == np.max((abs(delta_x), abs(delta_y), abs(delta_z)))
    )[0][0]
    if delta_x == 0 and delta_y == 0 and delta_z == 0:
        max_delta = 3

    # Calculate scaling direction + or - or None
    dx_sign, dy_sign, dz_sign = np.sign(delta_x), np.sign(delta_y), np.sign(delta_z)

    # Calculating scaling threshold
    delta_err_x = (
        0.0
        if delta_x == 0
        else (
            abs(delta_x / delta_y)
            if delta_y != 0
            else abs(delta_x / delta_z) if delta_z != 0 else 0.0
        )
    )
    delta_err_y = (
        0.0
        if delta_y == 0
        else (
            abs(delta_y / delta_x)
            if delta_x != 0
            else abs(delta_y / delta_z) if delta_z != 0 else 0.0
        )
    )

    if dim_ != 2:
        delta_err_z = (
            0.0
            if delta_z == 0
            else (
                np.minimum(abs(delta_z / delta_x), abs(delta_z / delta_y))
                if delta_x != 0 and delta_y != 0
                else (
                    abs(delta_z / delta_y)
                    if delta_x == 0 and delta_y != 0
                    else abs(delta_z / delta_x) if delta_x != 0 else 0.0
                )
            )
        )

    # Zero out threshold
    error_x, error_y, error_z = 0, 0, 0
    x, y, z = x0, y0, z0

    if max_delta == 0:  # Scale XYZ by iterating throw X axis
        for x in range(x0, x1, dx_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y

            # Iteratively add and scale Y axis
            error_y = error_y + delta_err_y
            while error_y >= 0.5:
                y += dy_sign
                error_y -= 1

            if dim_ != 2:
                # Iteratively add and scale Z axis
                error_z = error_z + delta_err_z
                while error_z >= 0.5:
                    z += dz_sign
                    error_z -= 1
    if max_delta == 1:  # Scale XYZ by iterating throw Y axis
        for y in range(y0, y1, dy_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y

            # Iteratively add and scale X axis
            error_x = error_x + delta_err_x
            while error_x >= 0.5:
                x += dx_sign
                error_x -= 1

            if dim_ != 2:
                # Iteratively add and scale Z axis
                error_z = error_z + delta_err_z
                while error_z >= 0.5:
                    z += dz_sign
                    error_z -= 1
    if max_delta == 2:  # Scale XYZ by iterating throw Z axis
        for z in range(z0, z1, dz_sign):
            if dim_ != 2:
                yield x, y, z
            else:
                yield x, y
    if max_delta == 3:  # Nothing to do
        if dim_ != 2:
            yield x, y, z
        else:
            yield x, y


def interpolation(points: np.ndarray) -> np.ndarray:
    """
    Performs 3D interpolation for XYZ coordinates using a given interpolation
    generator over sequential point pairs, and appends the last point to the
    output in integer format. The function creates a continuous series of
    interpolated points connecting all input coordinates.

    :param points: An array of shape (N, 3) representing a sequence of 3D points.

    :return: An array of shape (M, 3) containing the interpolated 3D coordinates,
             where M >= N due to newly generated intermediate points.
    """
    new_coord = []
    for i in range(0, len(points) - 1):
        """3D interpolation for XYZ dimension"""
        new_coord.append(list(interpolate_generator(points[i : i + 2, :])))

    # Append last point
    new_coord.append(list(np.round(points[-1, :]).astype(np.int32)))

    return np.vstack(new_coord)
