#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Union

import numpy as np
from scipy.interpolate import splev, splprep


def curvature(coord: np.ndarray, tortuosity_=False) -> Union[float, tuple]:
    """
    Calculate spline curvature.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.
        tortuosity_ (bool): Optional, if True calculates the tortuosity.

    Returns:
        float: Spline tortuosity measured with tortuosity.
    """

    tck, u = splprep(coord.T, s=0)

    # Calculate the first, second, and third derivatives
    der1 = splev(u, tck, der=1)  # First derivative (velocity)
    der2 = splev(u, tck, der=2)  # Second derivative (acceleration)

    # Stack the derivatives for easier manipulation
    r1 = np.vstack(der1).T  # First derivative (velocity)
    r2 = np.vstack(der2).T  # Second derivative (acceleration)

    # Calculate curvature
    curvature = (
        np.linalg.norm(np.cross(r1, r2), axis=1) / np.linalg.norm(r1, axis=1) ** 3
    )

    if tortuosity_:
        # Calculate the curve length (arc length) using the fine points
        arc_length = np.sum(np.linalg.norm(np.diff(coord, axis=0), axis=1))

        # Calculate the straight-line distance between the start and end points
        start_point = coord[0]
        end_point = coord[-1]
        straight_line_distance = np.linalg.norm(end_point - start_point)

        # Calculate tortuosity
        tortuosity = arc_length / straight_line_distance

        return curvature, tortuosity
    return curvature


def curvature_list(
    coord: np.ndarray, tortuosity_=False, mean_=False
) -> Union[list, tuple]:
    """
    Calculate the curvature of all splines and return it as a list.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.
        tortuosity_ (bool): Optional, if True calculates the tortuosity.
        mean_ (bool): If true, return an average curvature.

    Returns:
        list: Spline curvature list.
    """

    spline_curvature_list, spline_tortuosity_list = [], []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]

        if len(points) > 5:
            if tortuosity_:
                c, t = curvature(points, tortuosity_=True)

                if mean_:
                    c = np.mean(c).item(0)
                spline_curvature_list.append(c)
                spline_tortuosity_list.append(t)
            else:
                spline_curvature_list.append(curvature(points))
        else:
            spline_curvature_list.append(0.0)
            spline_tortuosity_list.append(1.0)

    if tortuosity_:
        return spline_curvature_list, [float(x) for x in spline_tortuosity_list]
    else:
        return spline_curvature_list


def tortuosity(coord: np.ndarray) -> float:
    """
    Calculate spline tortuosity.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        float: Spline tortuosity measured with tortuosity.
    """
    if len(coord) <= 1:
        return 1.0

    length = total_length(coord) + 1e-16
    start_point = coord[0]
    end_point = coord[-1]
    straight_line_distance = np.linalg.norm(end_point - start_point)

    return length / straight_line_distance


def tortuosity_list(coord: np.ndarray) -> list:
    """
    Calculate the tortuosity of all splines and return it as a list.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        list: Spline tortuosity list.
    """

    spline_tortuosity_list = []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        spline_tortuosity_list.append(tortuosity(points))

    return [float(x) for x in spline_tortuosity_list]


def total_length(coord: np.ndarray) -> float:
    """
    Calculate the total length of the spline.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        float: Spline length.
    """
    length = np.sum(np.linalg.norm(np.diff(coord, axis=0), axis=1))

    return length


def length_list(coord: np.ndarray) -> list:
    """
    Calculate the total length of all splines and return it as a list.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.

    Returns:
        list: Spline length list.
    """
    spline_length_list = []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        spline_length_list.append(total_length(points))

    return [float(x) for x in spline_length_list]


def angle_between_vectors(v1, v2):
    """
    Calculate the angle in degrees between two vectors.

    This function uses the dot product and the magnitudes of the vectors
    to calculate the angle between them according to the formula:

        cos(theta) = (A . B) / (||A|| ||B||)

    Args:
        v1(np.ndarray): First input vector.
        v2(np.ndarray): Second input vector.

    Returns:
        float The angle in degrees between vector 'v1' and 'v2'.
    """
    # Calculate the dot product of vectors v1 and v2
    dot_product = np.dot(v1, v2)

    # Calculate the magnitude (norm) of vectors
    magnitude_v1 = np.linalg.norm(v1) + 1e-16
    magnitude_v2 = np.linalg.norm(v2) + 1e-16

    # Calculate angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return np.degrees(angle)


def intensity(data: np.ndarray, image: np.ndarray, thickness=1.0):
    tck, u = splprep(data.T, s=0)

    data_fine = np.linspace(0, 1, int(total_length(data)))
    data_fine = np.array(splev(data_fine, tck)).T

    pixel_coords = np.rint(data_fine).astype(int)

    # Extract the pixel values along the spline
    spline_intensity = pixel_intensity(pixel_coords, image)

    # Determined avg. background level
    spline_up = np.copy(pixel_coords)
    spline_up[:, 1] = spline_up[:, 1] + 10

    spline_down = np.copy(pixel_coords)
    spline_down[:, 1] = spline_down[:, 1] - 10

    # Extract the pixel values along the spline
    intensity_up = pixel_intensity(spline_up, image)
    intensity_down = pixel_intensity(spline_down, image)

    if intensity_up is None and intensity_down is not None:
        spline_background = intensity_down
    elif intensity_down is None and intensity_up is not None:
        spline_background = intensity_up
    elif intensity_up is not None and intensity_down is not None:
        if np.mean(intensity_up) > np.mean(intensity_down):
            spline_background = intensity_down
        else:
            spline_background = intensity_up
    else:
        return np.zeros_like(spline_intensity)
    return np.array(spline_intensity) - np.array(spline_background)


def pixel_intensity(coord, image):
    extracted_pixels = []

    if image.ndim == 2:
        for x, y, _ in coord:
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                extracted_pixels.append(image[y, x])
            else:
                extracted_pixels.append(None)
    else:
        for x, y, z in coord:
            if (
                0 <= z < image.shape[0]
                and 0 <= y < image.shape[1]
                and 0 <= x < image.shape[2]
            ):
                extracted_pixels.append(image[y, x])
            else:
                extracted_pixels.append(None)

    if None in extracted_pixels:
        try:
            min_value = min(item for item in extracted_pixels if item is not None)
            extracted_pixels = [
                min_value if item is None else item for item in extracted_pixels
            ]
        except ValueError:
            extracted_pixels = None
    return extracted_pixels


def intensity_list(coord: np.ndarray, image: np.ndarray):
    """
    Calculate the total length of all splines and return it as a list.

    Args:
        coord (np.ndarray): Coordinates for each unsorted point idx.
        image (np.ndarray): Associated image.

    Returns:
        list: Spline length list.
    """
    spline_intensity_list = []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        if len(points) > 5:
            spline_intensity_list.append(intensity(points, image))
        else:
            spline_intensity_list.append(0.0)

    return spline_intensity_list
