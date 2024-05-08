#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from math import pow, sqrt
from typing import Tuple, Union

import numpy as np
from skimage import draw

from tardis_em.cnn.data_processing.interpolation import interpolation
from tardis_em.utils.errors import TardisError


def draw_instances(
    mask_size: Union[list, tuple],
    coordinate: np.ndarray,
    pixel_size: float,
    circle_size=250,
    label=True,
    dtype=None,
) -> np.ndarray:
    """
    Module to build semantic mask from corresponding coordinates

    Args:
        mask_size (tuple): Size of array that will hold created mask.
        coordinate (np.ndarray): Segmented coordinates of a shape [Label x X x Y x (Z)].
        pixel_size (float): Pixel size in Angstrom.
        circle_size (int): Size of a circle the label mask in Angstrom.
        label (bool): If True, expect label point cloud.
        dtype (dtype):

    Returns:
        np.ndarray: Binary mask with drawn all coordinates as lines.
    """
    if label:
        if coordinate.ndim != 2 and coordinate.shape[1] not in [3, 4]:
            TardisError(
                "113",
                "tardis_em/cnn/data_processing.md/draw_mask.py",
                "Coordinates are of not correct shape, expected: "
                f"shape [Label x X x Y x (Z)] but {coordinate.shape} given!",
            )

    if label:
        label_mask = np.zeros(mask_size, dtype=np.uint16 if dtype is None else dtype)
    else:
        label_mask = np.zeros(mask_size, dtype=np.uint8 if dtype is None else dtype)

    if pixel_size == 0:
        pixel_size = 1

    r = (circle_size // 2) // pixel_size

    if coordinate.shape[1] == 3 or not label:  # Draw 2D mask
        mask_shape = "c"
    else:  # Draw 3D mask
        mask_shape = "s"

    # Number of segments in coordinates
    if label:
        segments = np.unique(coordinate[:, 0])

        """Build semantic mask by drawing circle in 2D for each coordinate point"""
        for i in segments:
            # Pick coordinates for each segment
            points = coordinate[np.where(coordinate[:, 0] == i)[0]][:, 1:]

            label = interpolation(points)
            all_cz, all_cy, all_cx = [], [], []

            """Draw label"""
            for j in range(len(label)):
                c = label[j, :]  # Point center

                if len(c) == 3:
                    cz, cy, cx = draw_mask(
                        r=r, c=c, label_mask=label_mask, segment_shape=mask_shape
                    )
                    all_cz.append(cz)
                    all_cy.append(cy)
                    all_cx.append(cx)
                else:
                    cy, cx = draw_mask(
                        r=r, c=c, label_mask=label_mask, segment_shape=mask_shape
                    )
                    all_cy.append(cy)
                    all_cx.append(cx)

            if len(c) == 3:
                all_cz, all_cy, all_cx = (
                    np.concatenate(all_cz),
                    np.concatenate(all_cy),
                    np.concatenate(all_cx),
                )
                label_mask[all_cz, all_cy, all_cx] = i + 1
            else:
                all_cy, all_cx = (
                    np.concatenate(all_cy),
                    np.concatenate(all_cx),
                )
                label_mask[all_cy, all_cx] = i + 1

        return label_mask
    else:
        # Pick coordinates for each point
        all_cz, all_cy, all_cx = [], [], []

        if np.all(coordinate[:, 2] == 0):
            coordinate = coordinate[:, :2]

        for c in coordinate:
            if len(c) == 3:
                cz, cy, cx = draw_mask(
                    r=r, c=c, label_mask=label_mask, segment_shape=mask_shape
                )
                all_cz.append(cz)
                all_cy.append(cy)
                all_cx.append(cx)
            else:
                cy, cx = draw_mask(
                    r=r, c=c, label_mask=label_mask, segment_shape=mask_shape
                )
                all_cy.append(cy)
                all_cx.append(cx)
                all_cz.append(np.repeat(0, len(cx)))

        all_cz, all_cy, all_cx = (
            np.concatenate(all_cz),
            np.concatenate(all_cy),
            np.concatenate(all_cx),
        )

        if coordinate.shape[1] == 2:
            label_mask[all_cy, all_cx] = 1
        else:
            label_mask[all_cz, all_cy, all_cx] = 1

        return np.where(label_mask == 1, 1, 0).astype(np.uint8)


def draw_semantic_membrane(
    mask_size: tuple, coordinate: np.ndarray, pixel_size: float, spline_size=70
) -> np.ndarray:
    """
    Draw semantic membrane

    For each Z pick individual instance and draw a fitted spline of given thickness.

    Args:
        mask_size (tuple): Size of array that will hold created mask.
        coordinate (np.ndarray): Segmented coordinates of a shape [Label x X x Y x (Z)].
        pixel_size (float): Pixel size in Angstrom.
        spline_size (int): Size of a circle the label mask in Angstrom.

    Returns:
        np.ndarray: Binary mask with drawn all coordinates as lines.
    """
    # Ensure ints
    coordinate = coordinate.astype(np.int32)

    # Initiate mask size
    r = round((spline_size / 2) / pixel_size)

    # Initiate mask
    label_mask = np.zeros(mask_size, dtype=np.uint8)

    """Iterate throw each coord"""
    for i in coordinate:
        c = i[1:]

        cz, cy, cx = draw_mask(r=r, c=c, label_mask=label_mask, segment_shape="s")

        label_mask[cz, cy, cx] = 1

    return label_mask


def draw_instances_membrane(
    mask_size: tuple, coordinate: np.ndarray, pixel_size: float, spline_size=70
) -> np.ndarray:
    """
    Draw instances membrane

    For each Z pick individual instance and draw a fitted spline of given thickness.

    Args:
        mask_size (tuple): Size of array that will hold created mask.
        coordinate (np.ndarray): Segmented coordinates of a shape [Label x X x Y x (Z)].
        pixel_size (float): Pixel size in Angstrom.
        spline_size (int): Size of a circle the label mask in Angstrom.

    Returns:
        np.ndarray: Binary mask with drawn all coordinates as lines.
    """
    # Ensure ints
    coordinate = coordinate.astype(np.int32)

    # Initiate mask size
    r = round((spline_size / 2) / pixel_size)

    # Initiate mask
    label_mask = np.zeros(mask_size, dtype=np.uint16)

    """Iterate throw each coord"""
    for i in coordinate:
        c = i[1:]
        id = i[0]

        cz, cy, cx = draw_mask(r=r, c=c, label_mask=label_mask, segment_shape="s")

        label_mask[cz, cy, cx] = id

    return label_mask


def draw_mask(
    r: int, c: np.ndarray, label_mask: np.ndarray, segment_shape: str
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Module draw_mask to construct sphere shape of a label

    Args:
        r (int): radius of a circle in Angstrom.
        c (np.ndarray): point in 3D indicating center of a circle [X x Y x Z].
        label_mask (np.ndarray): array of a mask on which circle is drawn.
        segment_shape (str): Type of shape to draw. Expect ['s', 'c'].

    Returns:
        np.ndarray: Binary mask.
    """
    if label_mask.ndim not in [2, 3]:
        TardisError(
            "113",
            "tardis_em/cnn/data_processing.md/draw_mask.py"
            f"Unsupported dimensions given {label_mask.ndim} expected 2!",
        )

    x = int(c[0])
    y = int(c[1])
    if len(c) == 3:
        z = int(c[2])
    else:
        z = None

    if segment_shape not in ["s", "c"]:
        TardisError(
            "123",
            "tardis_em/cnn/data_processing.md/draw_mask.py"
            f"Unsupported shape type to draw given {segment_shape} but "
            'expected "c" - circle or "s" - sphere',
        )
    if segment_shape == "s":
        cz, cy, cx = draw_sphere(r=r, c=(z, y, x), shape=label_mask.shape)
        return cz, cy, cx
    else:
        if z is not None:
            cz, cy, cx = draw_circle(r=r, c=(z, y, x), shape=label_mask.shape)
            return cz, cy, cx
        else:
            cy, cx = draw_circle(r=r, c=(y, x), shape=label_mask.shape)
            return cy, cx


def draw_circle(
    r: int, c: tuple, shape: tuple
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Draw a circle and shift coordinate to c position.

    Args:
        r (int): radius of a circle in Angstrom.
        c (tuple): point in 3D indicating center of a circle [(Z), Y, X].
        shape (tuple): Shape of mask to eliminated ofe-flowed pixel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of array's with zyx coordinates
    """
    r_dim = round(r * 3)
    c_frame = round(r)

    ny, nx = draw.disk((r, r), r, shape=(r_dim, r_dim))
    if len(c) == 3:
        nz = np.repeat(c[0], len(nx))

        c = ((c[1] - c_frame), (c[2] - c_frame))
        z, y, x = nz, ny + c[0], nx + c[1]

        # Remove pixel out of frame
        zyx = np.array((z, y, x)).T
        del_id = []
        for id_, i in enumerate(zyx):
            if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
                del_id.append(id_)

        zyx = np.delete(zyx, del_id, 0)

        return zyx[:, 0], zyx[:, 1], zyx[:, 2]
    else:
        c = ((c[0] - c_frame), (c[1] - c_frame))
        y, x = ny + c[0], nx + c[1]

        # Remove pixel out of frame
        yx = np.array((y, x)).T
        del_id = []
        for id_, i in enumerate(yx):
            if i[0] >= shape[0] or i[1] >= shape[1]:
                del_id.append(id_)

        yx = np.delete(yx, del_id, 0)

        return yx[:, 0], yx[:, 1]


def draw_sphere(
    r: int, c: tuple, shape: tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw a sphere and shift coordinate to c position.

    Args:
        r (int): radius of a sphere in Angstrom.
        c (tuple): point in 3D indicating center of a sphere [Z, Y, X].
        shape (tuple): Shape of mask to eliminated ofe-flowed pixel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of array's with zyx coordinates
    """
    r_dim = round(r * 2)
    sphere_frame = np.zeros((r_dim, r_dim, r_dim), dtype=np.int8)
    trim = round(r_dim / 6)
    c_frame = round(r)

    z, y, x = sphere_frame.shape
    for z_dim in range(z):
        for y_dim in range(y):
            for x_dim in range(x):
                dist_zyx_to_c = sqrt(
                    pow(abs(z_dim - c_frame), 2)
                    + pow(abs(y_dim - c_frame), 2)
                    + pow(abs(x_dim - c_frame), 2)
                )
                if dist_zyx_to_c > c_frame:
                    sphere_frame[z_dim, y_dim, x_dim] = False
                else:
                    sphere_frame[z_dim, y_dim, x_dim] = True
    sphere_frame[:trim, :] = False  # Trim bottom of the sphere
    sphere_frame[-trim:, :] = False  # Trim top of the sphere
    z, y, x = np.where(sphere_frame)

    c = ((c[0] - c_frame), (c[1] - c_frame), (c[2] - c_frame))
    z, y, x = z + c[0], y + c[1], x + c[2]

    # Remove pixel out of frame
    zyx = np.array((z, y, x)).T
    del_id = []
    for id_, i in enumerate(zyx):
        if i[0] >= shape[0] or i[1] >= shape[1] or i[2] >= shape[2]:
            del_id.append(id_)
        if i[0] < 0:  # Remove negative value
            del_id.append(id_)

    zyx = np.delete(zyx, del_id, 0)

    return zyx[:, 0], zyx[:, 1], zyx[:, 2]
