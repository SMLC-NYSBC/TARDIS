#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import gc
from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from skimage.morphology import skeletonize

from tardis_em.dist_pytorch.utils.utils import VoxelDownSampling
from tardis_em.utils.errors import TardisError


class BuildPointCloud:
    """
    A utility class for handling and processing images to build point cloud data, typically from semantic masks.
    This class includes functions to validate and adjust input image data, as well as convert binary semantic
    masks into point cloud data. The generated point cloud data can be represented in 2D or 3D space.
    """

    @staticmethod
    def check_data(image: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
        """
        Checks and processes a binary input image to ensure it meets the required conditions
        for a semantic mask. The function supports input as a file path to a .tiff image or a
        numpy array, and it returns the processed image if all conditions are satisfied.

        The function performs the following checks and adjustments:
        - Reads the image from a file path if a string is provided as input.
        - Validates that the image has either 2 or 3 dimensions.
        - Ensures the image is binary (contains only 0 and 1 values).
        - Converts the binary image values appropriately to ensure consistency.
        - Normalizes the image if necessary.

        :param image: Input image that can be either a string representing a file path to a
                      .tiff image or a numpy array. The input file or array is checked and
                      formatted as required.
        :type image: Union[str, np.ndarray]

        :return: Processed numpy array representing the validated binary image, or None if the
                 image fails the binary or formatting checks.
        :rtype: Union[np.ndarray, None]

        :raises RuntimeWarning: If the input directory or .tiff image file is incorrect.
        :raises TardisError: If the input image fails any validation checks, such as having
                             incorrect dimensions or not being binary.
        """
        try:
            if isinstance(image, str):
                from tardis_em.utils.load_data import load_tiff

                image, _ = load_tiff(tiff=image)
        except RuntimeWarning:
            TardisError(
                "121",
                "tardis_em/dist/utils",
                "Directory/input .tiff file/array is not correct...",
            )

        if image.ndim not in [2, 3]:
            TardisError(
                "113",
                "tardis_em/dist/utils",
                f"Image dim expected to be 2 or 3 bu got {image.ndim}",
            )

        """Check for binary"""
        unique_val = np.sort(pd.unique(image.flatten()))  # Use panda for speed
        if len(unique_val) == 1:
            TardisError(
                "115",
                "tardis_em/dist/utils",
                f"Not binary image. Expected 0-1 value but got: {unique_val}",
            )
            return None

        if len(unique_val) != 2:
            TardisError(
                "115",
                "tardis_em/dist/utils",
                f"Not binary image. Expected 0-1 value but got: {unique_val}",
            )

        """Check for int8 vs uint8"""
        if np.any(unique_val > 254):  # Fix uint8 formatting
            image = image / 255

        """Any other exertions"""
        if unique_val[1] != 1:
            if len(unique_val) == 2:
                image = np.where(image > 0, 1, 0)
            else:
                TardisError(
                    "115",
                    "tardis_em/dist/utils",
                    "Array or file directory loaded properly but image "
                    "is not semantic mask... "
                    f"Expected 0-1 value but got: {unique_val}",
                )

        return image

    def build_point_cloud(
        self,
        image: Union[str, np.ndarray],
        down_sampling: Union[float, None] = None,
        as_2d=False,
    ) -> Union[Tuple[ndarray, ndarray], np.ndarray]:
        """
        Generates a point cloud from an input image through skeletonization, with optional
        down-sampling and control of dimensionality reduction to 2D.

        The function processes the input image by skeletonizing it to extract
        the pixel-wise skeleton. Depending on the image dimensions or user-provided
        parameters, a 2D or 3D spatial representation of the skeleton is then
        constructed. The point cloud data is returned in a structured format
        after optional down-sampling.

        :param image: The input data to generate the point cloud from. It can be
                      a string representing the file path to an image or a numpy
                      array of the input data.
        :type image: Union[str, np.ndarray]
        :param down_sampling: Optional down-sampling factor to reduce the point cloud
                              density. If provided, it defines the voxel size for
                              filtering points in 3D space.
        :type down_sampling: Union[float, None]
        :param as_2d: Boolean flag for forcing the representation of the point cloud
                      in two dimensions irrespective of the dimensionality of the
                      input data.
        :type as_2d: bool
        :return: Returns a tuple containing (1) the generated point cloud as a numpy
                 array with X, Y (and optionally Z) coordinates, and (2) the reduced
                 point cloud if down-sampling is applied. If no down-sampling is applied,
                 only the point cloud is returned without being wrapped in a tuple.
        :rtype: Union[Tuple[ndarray, ndarray], np.ndarray]
        """
        image = self.check_data(image)
        if image is None:
            e = np.empty(0, dtype=np.int8)

            return e, e

        """Skeletonization"""
        if image.ndim == 2:
            image_point = skeletonize(image)
        elif as_2d:
            image_point = np.zeros(image.shape, dtype=np.uint8)

            for i in range(image_point.shape[0]):
                image_point[i, :] = np.where(skeletonize(image[i, :]), 1, 0)
        else:
            image_point = skeletonize(image)
        image_point = np.where(image_point > 0)

        """CleanUp to avoid memory loss"""
        del image

        """Output point cloud [X x Y x Z]"""
        if len(image_point) == 2:
            """If 2D bring artificially Z dim == 0"""
            coordinates_HD = np.stack(
                (image_point[1], image_point[0], np.zeros(image_point[0].shape))
            ).T
        else:
            coordinates_HD = np.stack(
                (image_point[2], image_point[1], image_point[0])
            ).T

            """CleanUp to avoid memory loss"""
            del image_point
            gc.collect()

        """ Down-sampling point cloud by removing closest point """
        if down_sampling is not None:
            down_sampling = VoxelDownSampling(
                voxel=down_sampling, labels=False, KNN=True
            )
            return coordinates_HD, down_sampling(coord=coordinates_HD)
        return coordinates_HD


def draw_line(p0: np.ndarray, p1: np.ndarray, line_id: int) -> np.ndarray:
    """
    Generates a sequence of 3D coordinates between two specified points using an incremental
    line drawing algorithm. The method determines the dominant axis of movement and computes
    intermediate points accordingly while maintaining resolution and alignment in 3D space.

    This is a computationally efficient implementation of a generalized Bresenham's algorithm
    adapted for 3D space. Each generated coordinate is stored along with its corresponding
    line ID.

    :param p0: Starting point in the form of a NumPy array with [z, y, x] coordinates.
    :type p0: np.ndarray
    :param p1: Ending point in the form of a NumPy array with [z, y, x] coordinates.
    :type p1: np.ndarray
    :param line_id: Identifier associated with the generated line.
    :type line_id: int
    :return: The generated line as a NumPy 2D array where each row consists of [line_id, x, y, z].
    :rtype: np.ndarray
    """
    xyz = []
    z0, y0, x0 = p0
    z1, y1, x1 = p1
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1

    if dx >= dy and dx >= dz:
        err_1 = 2 * dy - dx
        err_2 = 2 * dz - dx

        while x0 != x1:
            xyz.append([line_id, x0, y0, z0])

            if err_1 > 0:
                y0 += sy
                err_1 -= 2 * dx
            if err_2 > 0:
                z0 += sz
                err_2 -= 2 * dx
            err_1 += 2 * dy
            err_2 += 2 * dz
            x0 += sx

    elif dy >= dx and dy >= dz:
        err_1 = 2 * dx - dy
        err_2 = 2 * dz - dy

        while y0 != y1:
            xyz.append([line_id, x0, y0, z0])
            if err_1 > 0:
                x0 += sx
                err_1 -= 2 * dy
            if err_2 > 0:
                z0 += sz
                err_2 -= 2 * dy
            err_1 += 2 * dx
            err_2 += 2 * dz
            y0 += sy

    else:
        err_1 = 2 * dy - dz
        err_2 = 2 * dx - dz
        while z0 != z1:
            xyz.append([line_id, x0, y0, z0])
            if err_1 > 0:
                y0 += sy
                err_1 -= 2 * dz
            if err_2 > 0:
                x0 += sx
                err_2 -= 2 * dz
            err_1 += 2 * dy
            err_2 += 2 * dx
            z0 += sz

    xyz.append([line_id, x0, y0, z0])

    return np.vstack(xyz)


def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: float) -> list:
    """
    Calculate the position on a quadratic Bézier curve at a given parameter t.

    The function computes the coordinates of a point on a quadratic Bézier curve using
    the given control points: p0, p1, and p2, and the parameter t. Quadratic Bézier
    curves are a common representation of smooth curves in computer graphics and
    geometry. The computed point is returned as a list of rounded integer coordinates.

    :param p0: The first control point of the Bézier curve as a NumPy array.
    :param p1: The second control point of the Bézier curve as a NumPy array.
    :param p2: The third control point of the Bézier curve as a NumPy array.
    :param t: The parameter value along the curve where the computation is to be
        performed. The value of t must lie within the interval [0, 1].
    :return: A list of integers representing the coordinates of the computed
        point on the Bézier curve.
    """
    x_ = float((1 - t) ** 2 * p0[2] + 2 * (1 - t) * t * p1[2] + t**2 * p2[2])
    y_ = float((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
    z_ = float((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0])

    return [int(round(x_)), int(round(y_)), int(round(z_))]


def draw_curved_line(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, line_id: int
) -> np.ndarray:
    """
    Draws a quadratic Bézier curve using three control points and returns the resultant points
    concatenated with a line identifier. The method calculates an approximate number of points
    required to represent the curve based on distances between the given control points and
    generates evenly distributed points along the curve.

    :param p0: The first control point as a NumPy array.
    :param p1: The second control point as a NumPy array.
    :param p2: The third control point as a NumPy array.
    :param line_id: The identifier to prepend to each point in the resultant curve.
    :return: A NumPy array containing the computed points of the curve. Each row represents
             a point with the structure [line_id, x, y], where x and y are the coordinates
             of the curve point.
    """
    curve_points = []

    # Estimate the number of points needed to represent the curve
    distance_p0_p1 = np.linalg.norm(np.array(p0) - np.array(p1))
    distance_p1_p2 = np.linalg.norm(np.array(p1) - np.array(p2))
    num_points = int(np.sqrt(distance_p0_p1**2 + distance_p1_p2**2))

    for t in np.linspace(0, 1, num_points):
        point = quadratic_bezier(p0, p1, p2, t)
        curve_points.append([line_id, *point])

    return np.vstack(curve_points)


def draw_circle(
    center: np.ndarray,
    radius: float,
    circle_id: int,
    _3d=False,
    size=None,
) -> np.ndarray:
    """
    Generates the points of a circle in 2D or a sphere in 3D by calculating the coordinates
    of points lying on their respective perimeters based on the given center and radius.

    This function supports both 2D and 3D spaces. If the parameter `_3d` is set to True,
    it calculates the points of a 3D sphere within the specified bounds (`size`). Otherwise,
    it computes the 2D circle points with XY and optional Z-plane configurations.

    :param center:
        The center of the circle or sphere. Should be a numpy array with either
        2 or 3 components (x, y[, z]).
    :param radius:
        The radius of the circle or sphere.
    :param circle_id:
        An identifier assigned to each point in the circle/sphere. This value
        will be included as the first element in each resulting coordinate.
    :param _3d:
        A boolean to determine whether to compute a 3D sphere (True) or a 2D
        circle (False). Defaults to False.
    :param size:
        Optional size constraints for the 3D sphere (e.g., bounding box
        dimensions). This parameter is only used when `_3d` is True.
    :return:
        An array of point coordinates, each including the `circle_id`. For 2D,
        the coordinates are in (circle_id, x, y, z) format. For 3D, they are in
        (circle_id, z, y, x) format.
    :rtype:
        numpy.ndarray
    """
    circle_points = []

    if _3d:
        z0, y0, x0 = center
        sphere_points = []
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array(size)

        # Determine the angle increments based on the desired distance between points
        d_phi = d_theta = 1 / radius
        pi1 = np.random.randint(1, 5)
        pi2 = np.random.randint(1, 5)

        for theta in np.arange(0, 2 * np.pi // pi1, d_theta):
            for phi in np.arange(0, np.pi // pi2, d_phi):
                x = x0 + radius * np.sin(phi) * np.cos(theta)
                y = y0 + radius * np.sin(phi) * np.sin(theta)
                z = z0 + radius * np.cos(phi)

                sphere_points.append([circle_id, z, y, x])

        sphere_points = np.array(
            [
                p
                for p in sphere_points
                if np.all(p[1:] >= lower_bound) and np.all(p[1:] <= upper_bound)
            ]
        )

        return np.array(sphere_points)
    else:
        if len(center) == 3:
            z0, y0, x0 = center
        else:
            y0, x0 = center
            z0 = 0

        x, y = radius - 1, 0
        dx, dy = 1, 1
        err = dx - (radius * 2)

        while x >= y:
            points = [
                [x0 + x, y0 + y],
                [x0 + x, y0 - y],
                [x0 - x, y0 + y],
                [x0 - x, y0 - y],
                [x0 + y, y0 + x],
                [x0 + y, y0 - x],
                [x0 - y, y0 + x],
                [x0 - y, y0 - x],
            ]
            for point in points:
                circle_points.append([circle_id, *point, z0])

            if err <= 0:
                y += 1
                err += dy * 2
                dy += 1
            if err > 0:
                x -= 1
                dx += 1
                err += dx * 2 - (radius * 2)

    return np.vstack(circle_points)


def draw_sphere(center, radius, sheet_id):
    """
    Generates a set of points on the surface of a sphere with a specified radius and center
    and associates them with a given sheet ID. The points are approximately evenly
    distributed across the surface of the sphere.

    :param center: The 3D coordinates of the center of the sphere. Should be an array-like
        structure of three numeric values representing (x, y, z).
    :param radius: The radius of the sphere. A non-negative float value.
    :param sheet_id: An identifier for the generated points, typically a numeric or string
        value. Each point will be associated with this ID.
    :return: A 2D NumPy array where each row represents a point. The first column contains
        the sheet ID, and the subsequent columns contain the 3D coordinates
        (x, y, z) of the point.
    """
    # Approximate surface area of the sphere
    surface_area = 4 * np.pi * radius**2
    # Desired distance between points, roughly
    distance = 1

    # Estimate the number of points needed for the given spacing
    point_area = distance**2
    num_points = int(surface_area / point_area)

    # Generate points
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Scale points to the sphere radius
    points = np.stack([x, y, z], axis=-1)

    points += center

    return np.hstack((np.repeat(sheet_id, len(points))[:, None], points))


def draw_sheet(center: np.ndarray, size: tuple, sheet_id: int) -> np.ndarray:
    """
    Generates a 3D representation of a sheet with specified size, position, and a
    unique identifier. The generated sheet includes transformations and rotations
    to simulate a more varied geometric structure, and filters out points outside
    the specified bounds.

    :param center: A 3D ndarray representing the center point of the sheet.
    :param size: A tuple defining the 3D dimensions (depth, height, width) of the sheet.
    :param sheet_id: An integer unique identifier for the sheet being generated.
    :return: A ndarray containing the generated points of the sheet. Each row
        includes the sheet ID and the coordinates (x, y, z) of a point. If no points
        lie within the bounds, an empty ndarray is returned.
    """
    x_range = (0, 100)
    y_range = (0, 100)
    z_range = (0, 100)

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array((size[2], size[1], size[0]))

    # Random coefficients for more varied shapes
    coeffs = np.random.uniform(-10, 10, 12)

    # Create grid of x and y coordinates
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # # Apply transformations for Z
    Z = (
        coeffs[0] * np.sin(coeffs[1] * X)
        + coeffs[2] * np.cos(coeffs[3] * Y)
        + coeffs[4] * X**2
        + coeffs[5] * Y**2
    )
    Z = np.interp(Z, (Z.min(), Z.max()), z_range)

    # Apply transformations for Y
    Y = (
        coeffs[6] * np.sin(coeffs[7] * X)
        + coeffs[8] * np.cos(coeffs[9] * Z)
        + coeffs[10] * X**2
        + coeffs[11] * Z**2
    )
    Y = Y / z_range[1]

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Define random rotation angles
    angle_x, angle_y, angle_z = np.random.uniform(0, 2 * np.pi, 3)

    # Rotation matrices
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    # Apply rotations
    points = np.dot(points, Rx.T)
    points = np.dot(points, Ry.T)

    points[:, 0] = points[:, 0] + center[2] / 2
    points[:, 1] = points[:, 1] + center[1] / 2
    points[:, 2] = points[:, 2] + center[0] / 2

    points = np.array(
        [p for p in points if np.all(p >= lower_bound) and np.all(p <= upper_bound)]
    )

    if len(points) > 0:
        return np.hstack((np.repeat(sheet_id, len(points))[:, None], points))
    else:
        return points


def create_simulated_dataset(size, sim_type: str):
    """
    Generates a simulated dataset based on parameters specifying the type of simulation
    and the size of the dataset. The function creates different geometric shapes, such as
    lines, curves, circles, and spheres, depending on the specified simulation type.

    :param size: The dimensions of the dataset as a tuple of integers, representing its
        shape (e.g., (height, width, depth)).
    :param sim_type: A string specifying the type of simulation to create. Must be one of
        the following valid options: "mix3d", "mix2d", "filaments", "membranes",
        "membranes2d".
    :return: The generated dataset. The output is a NumPy array where each row represents
        a coordinate or shape in the simulated space.
    """
    assert sim_type in ["mix3d", "mix2d", "filaments", "membranes", "membranes2d"]
    coord = []
    i = 0
    if sim_type in ["mix2d", "mix3d"] or sim_type == "filaments":
        for _ in range(10):  # Drawing n random lines
            p1 = np.random.randint(0, size, size=(3,))
            p2 = np.random.randint(0, size, size=(3,))
            coord.append(draw_line(p1, p2, i))
            i += 1

        for _ in range(10):  # Drawing n random curves
            sp = np.random.randint(0, size, size=(3,))
            cp = np.random.randint(0, size, size=(3,))
            ep = np.random.randint(0, size, size=(3,))
            coord.append(draw_curved_line(sp, cp, ep, i))
            i += 1

    if sim_type == "mix2d" or sim_type == "membranes2d":
        for _ in range(100):  # Drawing n random circles
            radius = np.random.randint(10, size[1] // 20)

            center = np.random.randint(0, (size[1] - radius, size[2] - radius))
            coord.append(draw_circle(center, radius, i))
            i += 1

    if sim_type == "mix3d" or sim_type == "membranes":
        for _ in range(100):  # Drawing n random circles
            radius = np.random.randint(10, size[1] // 20)

            center = np.random.randint(
                0, (size[0] - radius, size[1] - radius, size[2] - radius)
            )
            coord.append(draw_sphere(center, radius, i))
            i += 1

    return np.vstack(coord)
