#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import gc
from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from skimage.morphology import skeletonize, skeletonize_3d

from tardis_pytorch.dist_pytorch.utils.utils import VoxelDownSampling
from tardis_pytorch.utils.errors import TardisError
from tardis_pytorch.utils.spline_metric import sort_segment


class BuildPointCloud:
    """
    MAIN MODULE FOR SEMANTIC MASK IMAGE DATA TRANSFORMATION INTO POINT CLOUD

    Module build point cloud from semantic mask based on skeletonization in 3D.
    Optionally, user can trigger point cloud correction with Euclidean distance
    transformation to correct for skeletonization artefact. It will not fix
    issue with heavily overlapping objects.

    The workflow follows: optional(edt_2d -> edt_thresholds -> edt_binary) ->
        skeletonization_3d -> output point cloud -> (down-sampling)
    """

    @staticmethod
    def check_data(image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Check image data and correct it if needed to uint8 type.

        Args:
            image (np.ndarray, str): 2D/3D image array

        Returns:
            np.ndarray: Check and converted to uint8 data
        """
        try:
            if isinstance(image, str):
                from tardis_pytorch.utils.load_data import import_tiff

                image, _ = import_tiff(tiff=image)
        except RuntimeWarning:
            TardisError(
                "121",
                "tardis_pytorch/dist/utils",
                "Directory/input .tiff file/array is not correct...",
            )

        if image.ndim not in [2, 3]:
            TardisError(
                "113",
                "tardis_pytorch/dist/utils",
                f"Image dim expected to be 2 or 3 bu got {image.ndim}",
            )

        """Check for binary"""
        unique_val = np.sort(pd.unique(image.flatten()))  # Use panda for speed
        if len(unique_val) != 2:
            TardisError(
                "115",
                "tardis_pytorch/dist/utils",
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
                    "tardis_pytorch/dist/utils",
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

        if EDT:
            import edt

            """Calculate EDT and apply threshold based on predefine mask size"""
            if image.ndim == 2:
                image_edt = edt.edt(image)
                image_edt = np.where(image_edt > mask_size, 1, 0)
            else:
                image_edt = np.zeros(image.shape, dtype=np.uint8)

                if as_2d:
                    for i in range(image_edt.shape[0]):
                        df_edt = edt.edt(image[i, :])

                        image_edt[i, :] = np.where(df_edt > mask_size, 1, 0)
                else:
                    image_edt = edt.edt(image)
                    image_edt = np.where(image_edt > mask_size, 1, 0)

            image_edt = image_edt.astype(np.uint8)

            """Skeletonization"""
            if image.ndim == 2:
                image_point = skeletonize(image_edt)
            elif as_2d:
                image_point = np.zeros(image_edt.shape, dtype=np.uint8)

                for i in range(image_point.shape[0]):
                    image_point[i, :] = np.where(skeletonize(image_edt[i, :]), 1, 0)
            else:
                image_point = skeletonize_3d(image_edt)
            image_point = np.where(image_point > 0)

            """CleanUp to avoid memory loss"""
            del image, image_edt
        else:
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


def generate_random_3d_point(low=-10, high=10) -> np.ndarray:
    """
    Generate a random 3D point.

    Args:
        low (float): The lower bound for each coordinate of the 3D point.
        high (float): The upper bound for each coordinate of the 3D point.

    Returns:
        ndarray: A 3-element array representing the random 3D point with coordinates between `low` and `high`.
    """
    return np.random.uniform(low, high, 3)


def generate_random_rotation_matrix() -> np.ndarray:
    """
    Generate a random 3D rotation matrix.

    This function creates a random rotation matrix that can be used to rotate points in 3D space.
    It does this by generating three random angles corresponding to rotations about the x, y, and z axes.
    Individual rotation matrices are then computed for each of these angles, which are combined to produce
    the final rotation matrix.

    Returns:
        ndarray: A 3x3 matrix representing the random 3D rotation.
    """
    # Random angles for rotation around x, y, and z
    rx, ry, rz = np.random.uniform(-np.pi, np.pi, 3)

    # Rotation matrix around x-axis
    rot_x = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )

    # Rotation matrix around y-axis
    rot_y = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    # Rotation matrix around z-axis
    rot_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    # Combined rotation matrix
    rot_matrix = np.dot(rot_z, np.dot(rot_y, rot_x))

    return rot_matrix


def bezier_curve(control_points, n_points=100):
    """
    Evaluate a Bezier curve at a set of parameter values.

    Given a list of control points, this function computes the Bezier curve's points using the Bernstein basis
    polynomial. The resulting curve starts at the first control point and ends at the last control point.

    Args:
        control_points (List[ndarray]): List of 3D control points that define the Bezier curve.
        n_points (int): Number of points on the curve to compute.

    Returns:
        ndarray: A set of 3D points (shape: [n_points, 3]) on the Bezier curve.
    """
    t_values = np.linspace(0, 1, n_points)
    n = len(control_points) - 1
    curve_points = np.zeros((n_points, 3))

    for i, t in enumerate(t_values):
        for j in range(n + 1):
            curve_points[i] += (
                control_points[j]
                * (
                    np.math.factorial(n)
                    / (np.math.factorial(j) * np.math.factorial(n - j))
                )
                * ((1 - t) ** (n - j))
                * (t**j)
            )

    return curve_points


def resample_curve(points, desired_distance=1.0):
    """
    Resample a 3D curve to ensure a consistent distance between consecutive points.

    This function takes in a list of 3D points that form a curve and resamples it so that the distance
    between consecutive points on the resampled curve is approximately equal to the provided `desired_distance`.

    Args:
        points (List[ndarray]): List of 3D points that form the curve.
        desired_distance (float): The desired distance between consecutive points on the resampled curve.

    Returns:
        ndarray: A set of 3D points on the resampled curve.
    """
    resampled_points = [points[0]]
    remaining_distance = desired_distance

    i = 0
    while i < len(points) - 1:
        segment_length = np.linalg.norm(points[i + 1] - points[i])

        if segment_length > remaining_distance:
            fraction = remaining_distance / segment_length
            new_point = points[i] + fraction * (points[i + 1] - points[i])
            resampled_points.append(new_point)
            remaining_distance = desired_distance
        else:
            remaining_distance -= segment_length
            i += 1  # Move to the next point

    return np.array(resampled_points)


def generate_random_bezier_curve(id=0):
    """
    Generate a random 3D Bezier curve with a random origin.

    This function generates a random Bezier curve in 3D space. The control points of the curve are first
    generated randomly. These points are then rotated using a random rotation matrix and moved to a new
    random origin. The Bezier curve is then calculated based on these transformed control points.
    Optionally, the resulting curve points might be shuffled and reduced to half.

    Args:
        id (int): An identifier for the curve. Default is 0.

    Returns:
        List[ndarray]: If the number of curve points is greater than 3, the function returns
        the sorted segment of points; otherwise, it returns an empty list. Each point has an additional
        dimension at the beginning indicating the curve id.
    """
    # Generate random control points
    origin_range = np.random.randint(10, 100)
    origin_range = (-origin_range, origin_range)
    n_points = np.random.randint(2, 4)

    control_points = np.array(
        [
            generate_random_3d_point(origin_range[0], origin_range[1])
            for _ in range(n_points)
        ]
    )

    # Generate a random rotation matrix
    rotation_matrix = generate_random_rotation_matrix()

    # Apply rotation to control points
    rotated_control_points = np.dot(control_points, rotation_matrix)

    # Generate a random origin
    origin_range = np.random.randint(1, 5)
    origin_range = (-origin_range, origin_range)
    origin = generate_random_3d_point(low=origin_range[0], high=origin_range[1])

    # Move the control points to the new origin
    moved_control_points = rotated_control_points + origin

    # Calculate the points on the Bezier curve
    curve_points = resample_curve(bezier_curve(moved_control_points, 1000), 1)

    if np.random.randint(0, 10) > 5:
        np.random.shuffle(curve_points)
        curve_points = curve_points[: len(curve_points) // 2, :]
    points = np.zeros((len(curve_points), 4))
    points[:, 0] = id
    points[:, 1] = curve_points[:, 0]
    points[:, 2] = curve_points[:, 1]
    points[:, 3] = curve_points[:, 2]

    if len(points) > 3:
        return sort_segment(points)
    else:
        return []


def generate_bezier_curve_dataset(n=50):
    """
    Generate a dataset of random 3D Bezier curves.

    This function generates a dataset of random Bezier curves by repeatedly calling the
    `generate_random_bezier_curve` function. Only curves with more than 0 points are included
    in the final dataset.

    Args:
        n (int): The number of random Bezier curves to generate. Default is 50.

    Returns:
        ndarray: A concatenated set of 3D points representing the generated Bezier curves.
    """
    return np.concatenate(
        [j for j in [generate_random_bezier_curve(i) for i in range(n)] if len(j) > 0]
    )
