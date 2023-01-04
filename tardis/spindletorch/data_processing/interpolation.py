import numpy as np


def interpolation_1d(start: int,
                     stop: int,
                     max_len: int) -> np.ndarray:
    """
    1D INTERPOLATION FOR BUILDING SEMANTIC MASK

    Args:
        start (int): 1D single coordinate to start interpolation.
        stop (int): 1D single coordinate to stop interpolation.
        max_len (int): 1D axis length.

    Returns:
        np.ndarray: Interpolated 1D array
    """
    points_seq = np.linspace(start=int(start),
                             stop=int(stop),
                             num=max_len).round()

    return points_seq


def interpolation(points: np.ndarray) -> np.ndarray:
    """
    3D INTERPOLATION FOR BUILDING SEMANTIC MASK

    Args:
        points (np.ndarray): numpy array with points belonging to individual segments
            given by x, y, (z) coordinates.

    Returns:
        np.ndarray: Interpolated 2 or 3D array
    """
    coord = points

    for i in range(0, len(points) - 1):
        """1D interpolation for X dimension"""
        x = points[i:i + 2, 0]
        x_len = abs(x[0] - x[1])

        """1D interpolation for Y dimension"""
        y = points[i:i + 2, 1]
        y_len = abs(y[0] - y[1])

        """1D interpolation for optional Z dimension"""
        if points.shape[1] == 3:
            z = points[i:i + 2, 2]
            z_len = abs(z[0] - z[1])

            max_len = int(max([x_len, y_len, z_len]) + 1)
        else:
            z = None
            max_len = int(max([x_len, y_len]) + 1)

        new_coord = np.zeros((max_len, 3))

        x_new = interpolation_1d(start=x[0],
                                 stop=x[1],
                                 max_len=max_len)
        y_new = interpolation_1d(start=y[0],
                                 stop=y[1],
                                 max_len=max_len)
        z_new = interpolation_1d(start=z[0],
                                 stop=z[1],
                                 max_len=max_len)

        new_coord[0:max_len, 0] = list(map(int, x_new))
        new_coord[0:max_len, 1] = list(map(int, y_new))

        if z is not None:
            new_coord[0:max_len, 2] = list(map(int, z_new))

    return np.append(arr=coord, values=new_coord, axis=0)
