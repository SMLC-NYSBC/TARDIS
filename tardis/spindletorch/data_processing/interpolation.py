import numpy as np


def interpolation_1D(start: int,
                     stop: int,
                     max_len: int):
    """
    1D INTERPOLATION FOR BUILDING SEMANTIC MASK

    Args:
        start: 1D single coordinate to start interpolation
        stop: 1D single coordinate to stop interpolation
        max_len: 1D axis length
    """
    points_seq = np.linspace(start=int(start),
                             stop=int(stop),
                             num=max_len).round()

    return points_seq


def interpolation_3D(points: np.ndarray):
    """
    3D INTERPOLATION FOR BUILDING SEMANTIC MASK

    Args:
        points: numpy array with points belonging to individual segments given
        by x, y, z coordinates
    """
    interpolated = points

    for i in range(0, len(points) - 1):
        x = points[i:i + 2, 0]
        y = points[i:i + 2, 1]
        z = points[i:i + 2, 2]

        x_len = abs(x[0] - x[1])
        y_len = abs(y[0] - y[1])
        z_len = abs(z[0] - z[1])
        max_len = int(max([x_len, y_len, z_len]) + 1)

        """1D interpolation for each dimension"""
        x_new = interpolation_1D(start=x[0],
                                 stop=x[1],
                                 max_len=max_len)

        y_new = interpolation_1D(start=y[0],
                                 stop=y[1],
                                 max_len=max_len)

        z_new = interpolation_1D(start=z[0],
                                 stop=z[1],
                                 max_len=max_len)

        df = np.zeros((max_len, 3))

        df[0:max_len, 0] = list(map(int, x_new))
        df[0:max_len, 1] = list(map(int, y_new))
        df[0:max_len, 2] = list(map(int, z_new))

        interpolated = np.append(arr=interpolated,
                                 values=df,
                                 axis=0)

    return interpolated
