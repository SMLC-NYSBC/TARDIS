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
from skimage.morphology import skeletonize, skeletonize_3d

from tardis_em.dist_pytorch.utils.utils import VoxelDownSampling
from tardis_em.utils.errors import TardisError


class BuildPointCloud:
    """
    MAIN MODULE FOR SEMANTIC MASK IMAGE DATA TRANSFORMATION INTO POINT CLOUD

    Module build point cloud from semantic mask based on skeletonization in 3D.
    Optionally, user can trigger point cloud correction with Euclidean distance
    transformation to correct for skeletonization artefact. It will not fix
    issue with heavily overlapping objects.

    The workflows follows: optional(edt_2d -> edt_thresholds -> edt_binary) ->
        skeletonization_3d -> output point cloud -> (down-sampling)
    """

    @staticmethod
    def check_data(image: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
        """
        Check image data and correct it if needed to uint8 type.

        Args:
            image (np.ndarray, str): 2D/3D image array

        Returns:
            np.ndarray: Check and converted to uint8 data
        """
        try:
            if isinstance(image, str):
                from tardis_em.utils.load_data import import_tiff

                image, _ = import_tiff(tiff=image)
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
        EDT=False,
        mask_size=1.5,
        down_sampling: Union[float, None] = None,
        as_2d=False,
    ) -> Union[Tuple[ndarray, ndarray], np.ndarray]:
        """
        Build point cloud data from semantic mask.

        Args:
            image (np.ndarray): Predicted semantic mask.
            EDT (bool): If True, compute EDT to extract line centers.
            mask_size (float): Mask size to filter with EDT.
            down_sampling (float, None): If not None, down-sample point cloud with open3d.
            as_2d: Treat data as 2D not 3D.

        Returns:
            Union[Tuple[ndarray, ndarray], np.ndarray]: Point cloud of 2D/3D
            semantic objects.
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
            image_point = skeletonize_3d(image)
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
    Draws a straight line in 3D space between two points.

    Args:
    p0 (tuple): (z0, y0, x0) representing the start point of the line.
    p1 (tuple): (z1, y1, x1) representing the end point of the line.
    line_id (int): An identifier for the line.

    Returns:
        np.ndarray: A numpy array containing points (line_id, x, y, z) along the line.
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
    Calculate a point on a quadratic Bézier curve.

    Args:
        p0 (np.ndarray): (z0, y0, x0) representing the start point of the Bézier curve.
        p1 (np.ndarray): (z1, y1, x1) representing the control point of the Bézier curve.
        p2 (np.ndarray): (z2, y2, x2) representing the end point of the Bézier curve.
        t (float): A float between 0 and 1, representing the parameter of the curve.

    Returns:
        list: A list of integers [x, y, z], representing the calculated point on the curve.
    """
    x_ = float((1 - t) ** 2 * p0[2] + 2 * (1 - t) * t * p1[2] + t**2 * p2[2])
    y_ = float((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
    z_ = float((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0])

    return [int(round(x_)), int(round(y_)), int(round(z_))]


def draw_curved_line(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, line_id: int
) -> np.ndarray:
    """
    Draw a quadratic Bézier curve in a 3D array.

    Args:
        p0 (np.ndarray): (z0, y0, x0) representing the start point of the curve.
        p1 (np.ndarray): (z1, y1, x1) representing the control point of the curve.
        p2 (np.ndarray): (z2, y2, x2) representing the end point of the curve.
        line_id (int): An identifier for the curve.

    Returns:
        np.ndarray: A numpy array containing points [line_id, z, y, x] along the curve.
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
    Draw a circle in a 3D space.

    Args:
        center ( np.ndarray): (z0, y0, x0) representing the center of the circle.
        radius (float): The radius of the circle.
        circle_id (int): An identifier for the circle.
        _3d (bool):

    Returns:
        np.ndarray: A numpy array containing points [circle_id, z, y, x] on the circle.
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
    Generate n points on a 3D sheet.

    Args:
        center (np.ndarray): (z0, y0, x0) representing the center of the circle.
        size (tuple): Max size to fit sheet.
        sheet_id (int): Sheet unique id.

    Returns:
        np.ndarray: A numpy array containing points [x, y, z] on the sheet.
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
