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
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

from tardis_em.utils.normalization import MinMaxNormalize


def curvature(coord: np.ndarray, tortuosity_b=False) -> Union[float, tuple]:
    """
    Computes the curvature of a curve specified by input coordinates. Optionally, it can also compute
    the tortuosity of the curve if the `tortuosity_` parameter is set to True. The curvature is calculated
    as the norm of the cross product of the first and second derivatives divided by the norm of the first
    derivative raised to the power of three. Tortuosity is defined as the ratio of the curve's arc length
    to the straight-line distance between its endpoints.

    :param coord: A NumPy array representing the coordinates of the curve where curvature calculations
        are performed.
    :param tortuosity_b: A boolean flag indicating whether to calculate the tortuosity value in addition
        to the curvature. If True, tortuosity is calculated as the ratio of the curve length to the
        straight-line distance between the endpoints.

    :return: If `tortuosity_b` is False, a single float is returned representing the average curvature;
        if `tortuosity_b` is True, a tuple is returned containing the average curvature as the first
        element and the tortuosity value as the second element.
    """
    # try:
    tck, u = splprep(coord.T, s=1)

    # Calculate the first, second, and third derivatives
    der1 = splev(u, tck, der=1)  # First derivative (velocity)
    der2 = splev(u, tck, der=2)  # Second derivative (acceleration)

    # Stack the derivatives for easier manipulation
    r1 = np.vstack(der1).T  # First derivative (velocity)
    r2 = np.vstack(der2).T  # Second derivative (acceleration)

    # Calculate curvature
    curvature_value = (
        np.linalg.norm(np.cross(r1, r2), axis=1) / np.linalg.norm(r1, axis=1) ** 3
    )
    # except:
    #     curvature_value = [0.0]

    if tortuosity_b:
        # Calculate the curve length (arc length) using the fine points
        arc_length = np.sum(np.linalg.norm(np.diff(coord, axis=0), axis=1))

        # Calculate the straight-line distance between the start and end points
        start_point = coord[0]
        end_point = coord[-1]
        straight_line_distance = np.linalg.norm(end_point - start_point)

        # Calculate tortuosity
        tortuosity_value = arc_length / straight_line_distance

        return curvature_value, tortuosity_value
    return curvature_value


def curvature_list(
    coord: np.ndarray, tortuosity_b=False, mean_b=False
) -> Union[list, tuple]:
    """
    Computes curvature and optionally tortuosity of splines corresponding to unique
    coordinate groups in a given array. The input consists of a 2D array where the
    first column represents group identifiers and the subsequent columns represent
    the coordinates of points in each group. Curvature computation is based on groups
    with at least six points.

    :param coord: A 2D numpy array where the first column indicates grouping identifiers
        and the remaining columns represent coordinates of points.
    :param tortuosity_b: A boolean indicating whether to compute and return the tortuosity
        for each group. Default is False.
    :param mean_b: A boolean flag. When set to True, the mean curvature value for each
        group is computed instead of returning individual curvatures. Default is False.

    :return: If tortuosity_b is False, returns a list of curvatures for each unique group.
        If tortuosity_b is True, returns a tuple containing two elements: a list of curvatures
        and a list of tortuosity values (as floats) for each group.
    """

    def mean_without_outliers_iqr(data):
        q1 = np.percentile(data, 5)
        q3 = np.percentile(data, 95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = data[(data >= lower_bound) & (data <= upper_bound)]
        return np.mean(filtered)

    spline_curvature_list, spline_tortuosity_list = [], []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]

        if len(points) > 4:
            if tortuosity_b:
                c, t = curvature(points, tortuosity_b=True)

                if mean_b:
                    c = np.mean(c).item(0)
                spline_curvature_list.append(c)
                spline_tortuosity_list.append(t)
            else:
                c = curvature(points)
                if mean_b:
                    spline_curvature_list.append(mean_without_outliers_iqr(c).item(0))
                else:
                    if c == [0]:
                        c = np.repeat(0.0, len(points))
                    spline_curvature_list.append(c)
        else:
            spline_curvature_list.append(0.0)
            spline_tortuosity_list.append(1.0)

    if tortuosity_b:
        return spline_curvature_list, [float(x) for x in spline_tortuosity_list]
    else:
        return spline_curvature_list


def tortuosity(coord: np.ndarray) -> float:
    """
    Calculates the tortuosity of a curve.

    This function computes the ratio of the total length of a curve defined by a
    series of coordinates to the straight-line distance between the first and
    last points of the curve. If there is only one coordinate or none, it
    returns a default value of 1.0 as the tortuosity cannot be defined.

    :param coord: A numpy array containing the coordinates of the curve, where
                  each coordinate is a point in n-dimensional space.

    :return: A float representing the tortuosity, which is computed as the ratio
             of the total length of the curve to the straight-line distance
             between the start and end points.
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
    Calculates the tortuosity for each unique coordinate id in the provided array.
    The function groups the input coordinates by their unique first column values,
    computes the tortuosity for each group, and returns the results as a list of
    floats.

    :param coord: A 2-dimensional numpy array where the first column represents
        unique ids and the subsequent columns represent coordinates.
    :type coord: numpy.ndarray

    :return: A list of tortuosity values as floats, each corresponding to a
        unique id from the first column of input.
    :rtype: list
    """

    spline_tortuosity_list = []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        spline_tortuosity_list.append(tortuosity(points))

    return [float(x) for x in spline_tortuosity_list]


def total_length(coord: np.ndarray) -> float:
    """
    Computes the total length of a path defined by a sequence of coordinates.

    This function calculates the total distance between consecutive points
    in a path represented by an array of coordinates. The distance is computed
    using the Euclidean norm.

    :param coord: A numpy array where each row represents a point's coordinates
                  in the path.

    :return: Sum of the Euclidean distances between consecutive points
             in the path.
    :rtype: float
    """
    length = np.sum(np.linalg.norm(np.diff(coord, axis=0), axis=1))

    return length


def length_list(coord: np.ndarray) -> list:
    """
    Calculate the total lengths of grouped coordinates and return them as a list.

    This function processes a numpy array of coordinates, groups them based on the
    unique values in the first column, calculates the lengths of each group of coordinates
    using an external `total_length` function, and returns the lengths of these
    groups as a list of floats.

    :param coord: A numpy array of shape (N, M) where the first column describes a
        grouping attribute, and the remaining columns define the coordinates for
        which the lengths are calculated.

    :return: A list of float values representing the calculated lengths for each
        group of coordinates based on the unique grouping from the first column
        of the input array.
    """
    spline_length_list = []

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        spline_length_list.append(total_length(points))

    return [float(x) for x in spline_length_list]


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angle between two vectors in degrees. This function computes the
    angle based on the dot product and magnitudes of the input vectors. The arc cosine
    of the cosine similarity is calculated to derive the angle between the vectors.
    A small value is added to the magnitudes to prevent division by zero.

    cos(theta) = (A . B) / (||A|| ||B||)

    :param v1: The first vector represented as a numpy array.
    :type v1: np.ndarray
    :param v2: The second vector represented as a numpy array.
    :type v2: np.ndarray

    :return: The angle between `v1` and `v2` in degrees.
    :rtype: float
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


def intensity(data: np.ndarray, image: np.ndarray, thickness=1) -> tuple:
    """
    Computes the intensity values along a parametric spline through a given dataset and subtracts
    the background intensity estimated from parallel shifted spline paths. This function is designed
    specifically for tasks involving intensity extraction in image processing.

    :param data: 2D array containing the coordinates corresponding to the data points the spline
                 is fitted to.
    :type data: np.ndarray
    :param image: 2D image array from which pixel intensity values will be extracted.
    :type image: np.ndarray
    :param thickness: Integer denoting the thickness of the line for which pixel intensities will
                      be aggregated. Default value is 1.
    :type thickness: int, optional

    :return: 1D array containing the adjusted intensity values along the spline after background
             subtraction.
    :rtype: np.ndarray
    """
    tck, u = splprep(data.T, s=0)

    data_fine = np.linspace(0, 1, 2 * int(total_length(data)))
    data_fine = np.array(splev(data_fine, tck)).T

    pixel_coords = np.rint(data_fine).astype(int)
    pixel_coords = np.unique(pixel_coords, axis=0)

    if thickness[1] > 1:
        pixel_coords_bg = thicken_line_coordinates(pixel_coords, thickness[1])
    else:
        pixel_coords_bg = np.copy(pixel_coords)

    if thickness[0] > 1:
        pixel_coords = thicken_line_coordinates(pixel_coords, thickness[0])

    # Extract the pixel values along the spline
    spline_intensity = pixel_intensity(pixel_coords, image)

    # Determined avg. background level
    spline_up = np.copy(pixel_coords_bg)
    spline_up[:, 1] = spline_up[:, 1] + thickness[1]

    spline_down = np.copy(pixel_coords_bg)
    spline_down[:, 1] = spline_down[:, 1] - thickness[1]

    # Extract the pixel values along the spline
    intensity_up = pixel_intensity(spline_up, image)
    intensity_down = pixel_intensity(spline_down, image)

    if intensity_up is None and intensity_down is not None:
        spline_background = intensity_down
    elif intensity_down is None and intensity_up is not None:
        spline_background = intensity_up
    elif intensity_up is not None and intensity_down is not None:
        if np.mean(intensity_up) > np.mean(intensity_down):
            spline_background = np.mean(intensity_down)
        else:
            spline_background = np.mean(intensity_up)
    else:
        return 0.0, 0.0

    m = np.mean(np.array(spline_intensity) - spline_background)
    s = np.sum(np.array(spline_intensity) - spline_background)

    return m, s


def pixel_intensity(coord: np.ndarray, image: np.ndarray) -> Union[list, None]:
    """
    Extracts pixel intensities from an image at specified coordinates. The function
    supports both 2D and 3D images and handles out-of-bound coordinates with `None`.
    If any extracted pixel value is `None`, it replaces them with the minimum valid
    pixel value present in the list. If no valid pixel values are found, it returns
    `None`.

    :param coord: A numpy array of shape (N, 3) for 3D or (N, 2) for 2D, containing
        the coordinates where pixel intensities are to be extracted.
    :param image: A numpy array representing the image from which pixel intensities
        are to be extracted. Can be either 2D or 3D.

    :return: A list of extracted pixel intensities. If any coordinate is invalid,
        its corresponding intensity is initially `None` but replaced with the
        minimum valid pixel value in the list. Returns `None` if no valid pixel
        values exist.
    """
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
                extracted_pixels.append(image[z, y, x])
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


def thicken_line_coordinates(coords: np.ndarray, thickness: int):
    """
    Expands the thickness of a set of 3D coordinates to represent a line or shape with a specified
    thickness. This is achieved by creating a grid of points around each coordinate based on the
    specified thickness.

    This function is useful for visualizing or processing lines/shapes in three-dimensional space
    with a specific thickness, often needed in environments like 3D geometry processing or
    voxel-based computations.

    :param coords: A NumPy array containing 3D coordinates to be thickened. Each coordinate is
        represented as a tuple (x, y, z).
    :param thickness: An integer that defines the extent of expansion around the input coordinates.
        The thickness is centered on each coordinate, producing a local grid.

    :return: A NumPy array containing the updated set of 3D coordinates, including all points
        within the specified thickness range.
    """
    thickened_coords = set()

    # Radius for thickness (e.g., thickness=3 will add a 3x3 grid around each point)
    radius = thickness // 2

    for x, y, z in coords:
        # Create a local grid around each coordinate
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    thickened_coords.add((x + dx, y + dy, z + dz))

    return np.array(list(thickened_coords))


def intensity_list(coord: np.ndarray, image: np.ndarray, thickness=1):
    """
    Extracts intensity values along spline coordinates from the given image.

    This function computes the intensity values along a set of spline coordinates
    within a given image, using the specified thickness for sampling. It iterates
    through unique spline identifiers in the coordinate array, processes points
    associated with each identifier, and computes the intensity or returns a zero
    value if there are insufficient points.

    :param coord: A 2D array containing the spline coordinates. The first column
        represents unique identifiers for each spline, and the remaining columns
        represent the spatial coordinates of each point.
    :param image: A 2D numpy array representing the image from which intensity
        values are to be extracted.
    :param thickness: An optional parameter specifying the thickness used for
        sampling intensity values. Defaults to 1.

    :return: A list of intensity values computed for each unique spline identifier.
        The list contains a computed intensity value for identifiers with sufficient
        points and 0.0 for those with insufficient points.
    """
    norm = MinMaxNormalize()
    spline_intensity_list_m, spline_intensity_list_s = [], []
    image = norm(image)

    for i in np.unique(coord[:, 0]):
        points = coord[np.where(coord[:, 0] == i)[0], 1:]
        if len(points) > 5:
            m, s = intensity(points, image, thickness)
            spline_intensity_list_m.append(m)
            spline_intensity_list_s.append(s)
        else:
            spline_intensity_list_m.append(0.0)
            spline_intensity_list_s.append(0.0)

    return spline_intensity_list_m, spline_intensity_list_s


def calculate_spline_correlations(image_stack, spline_coords, frame_id, thickness=[1, 1]):
    norm = MinMaxNormalize()
    frame_ref = norm(image_stack[frame_id])

    N, Y, X = image_stack.shape
    num_splines = np.unique(spline_coords[:, 0])

    correlations = np.zeros((len(num_splines), N))
    correlations_px = {}
    for spline_id in num_splines:
        points = spline_coords[np.where(spline_coords[:, 0] == spline_id)[0], 1:]
        tck, u = splprep(points.T, s=0)

        data_fine = np.linspace(0, 1, 2 * int(total_length(points)))
        data_fine = np.array(splev(data_fine, tck)).T

        pixel_coords = np.rint(data_fine).astype(int)
        pixel_coords = np.unique(pixel_coords, axis=0)

        if thickness[1] > 1:
            pixel_coords_bg = thicken_line_coordinates(pixel_coords, thickness[1])
        else:
            pixel_coords_bg = np.copy(pixel_coords)

        if thickness[0] > 1:
            pixel_coords = thicken_line_coordinates(pixel_coords, thickness[0])

        # Determined avg. background level
        spline_up = np.copy(pixel_coords_bg)
        spline_up[:, 1] = spline_up[:, 1] + thickness[1]

        spline_down = np.copy(pixel_coords_bg)
        spline_down[:, 1] = spline_down[:, 1] - thickness[1]

        # Extract the pixel values along the spline
        intensity_up = pixel_intensity(spline_up, frame_ref)
        intensity_down = pixel_intensity(spline_down, frame_ref)

        if intensity_up is None and intensity_down is not None:
            spline_background = intensity_down
        elif intensity_down is None and intensity_up is not None:
            spline_background = intensity_up
        elif intensity_up is not None and intensity_down is not None:
            if np.mean(intensity_up) > np.mean(intensity_down):
                spline_background = np.mean(intensity_down)
            else:
                spline_background = np.mean(intensity_up)
        else:
            spline_background = 0.0

        # Get reference sequence at frame_id
        ref_intensity = pixel_intensity(pixel_coords, frame_ref) - spline_background

        correlations_px[int(spline_id)] = {}
        correlations_px[int(spline_id)]["reference"] = [float(f) for f in ref_intensity]
        correlations_px[int(spline_id)]["MT"] = {}
        for i in range(N):
            spline_intensity = pixel_intensity(pixel_coords, norm(image_stack[i])) - spline_background

            if i == frame_id:
                correlations_px[int(spline_id)]["MT"][i] = [0.0 for f in spline_intensity]
                correlations[int(spline_id), i] = 0.0
            else:
                correlations[int(spline_id), i] = np.corrcoef(ref_intensity, spline_intensity)[0, 1]
                correlations_px[int(spline_id)]["MT"][i] = [float(f) for f in spline_intensity]

    return correlations, correlations_px


def group_points_by_distance(points: np.ndarray, eps: float = 'auto', min_samples: int = 1) -> list[list[int]]:
    """
    Groups 3D points based on spatial proximity using DBSCAN.

    :param points: np.ndarray of shape (n, 4) where the first column is ID and the rest are XYZ.
    :param eps: Maximum distance between two samples for them to be considered in the same neighborhood.
    :param min_samples: Minimum number of samples to form a dense region.
    :return: List of lists of point IDs grouped by spatial proximity.
    """
    ids = points[:, 0].astype(int)
    coords = points[:, 1:]  # 3D coordinates

    # Auto-threshold calculation
    if eps == 'auto':
        pairwise_distances = pdist(coords)
        median_distance = np.median(pairwise_distances)
        eps = median_distance / 2

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    unique_labels = np.unique(labels)  # remove noise if any
    groups = []

    for label in unique_labels:
        group_ids = ids[labels == label].tolist()
        groups.append(group_ids)

    return groups
