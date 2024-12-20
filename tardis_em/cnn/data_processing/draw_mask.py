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
    Draws labeled or binary masks based on the input coordinates, mask size, and additional parameters.

    This function generates either 2D or 3D masks, depending on the shape of the provided coordinates.
    If label generation is enabled, unique segment labels are created to distinguish between different segments.
    Circles centered around specific points are drawn in the masks, with sizes determined by the input circle size
    and pixel size. Input parameters related to the shape, size, and type of the mask, as well as the labeling behavior,
    are fully customizable. This function is suitable for constructing semantic or instance masks.

    :param mask_size: The dimensions of the mask to be created.
    :type mask_size: list | tuple
    :param coordinate: An array of coordinates specifying the locations to draw the mask.
    :type coordinate: np.ndarray
    :param pixel_size: Size of a pixel in the mask, used to scale the mask appropriately.
    :type pixel_size: float
    :param circle_size: Diameter of the circle to be drawn at each coordinate point. Defaults to 250.
    :type circle_size: int, optional
    :param label: Flag indicating whether a labeled mask or a binary mask should be created. Defaults to True.
    :type label: bool, optional
    :param dtype: Data type for the output mask. If not provided, defaults to np.uint16 or np.uint8.
    :type dtype: str | None, optional

    :return: A 2D or 3D mask generated based on the input parameters.
    :rtype: np.ndarray
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
    Draws a semantic membrane mask based on given coordinates, pixel size, and spline size.
    The function generates a mask of the specified size and marks regions centered
    around the provided coordinates. The region size is determined by the given spline size,
    adjusted by the pixel size.

    :param mask_size: The dimensions of the output mask, specified as a tuple of integers (z, y, x).
    :type mask_size: tuple
    :param coordinate: A numpy array of coordinates specifying the center points of each region to be drawn in the mask.
    :type coordinate: np.ndarray
    :param pixel_size: Resolution of the mask, defining the relationship between spline size and the mask grid.
    :type pixel_size: float
    :param spline_size: The diameter of the spline in real-world size, measured in the same unit as pixel_size. Default is 70.
    :type spline_size: int, optional

    :return: A numpy ndarray representing the mask, where regions centered at the provided coordinates are filled.
    :rtype: np.ndarray
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
    Draws the instances of membranes on a specified mask by iterating through given
    coordinates, creating circular "stamp"-like regions based on the spline size and
    pixel size, and assigning unique identifiers to each region in the mask.

    The mask is a 3D array where each element corresponds to a region assigned to a
    specific identifier. The process converts coordinates to integers, calculates a
    radius from the given spline and pixel sizes, and applies the circular regions
    to the mask for each set of input coordinates.

    :param mask_size: A tuple representing the size of the mask, typically in 3D, that
        specifies its spatial dimensions.
    :param coordinate: A 2D numpy array where each row corresponds to an instance,
        containing the identifier in the first column and the coordinates in the remaining
        columns.
    :param pixel_size: A float representing the physical distance represented by
        a single pixel in the spatial dimensions of the mask.
    :param spline_size: An integer specifying the base size of the "stamped" mask
        regions. Optional, default value is 70.

    :return: A 3D numpy array with the same dimensions as the input mask, containing
        unique identifiers based on the given coordinate for each "stamped" membrane region.
    :rtype: numpy.ndarray
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
    Draws a mask on the given label array by creating a circle or sphere depending
    on the specified shape. The method can handle both 2D and 3D label masks.

    :param r: The radius of the circle (2D) or sphere (3D) to be drawn.
    :param c: A numpy array containing the center coordinates of the shape,
        where the length of the input determines if it is 2D or 3D.
    :param label_mask: A numpy array representing the label mask where
        the circle or sphere will be drawn. Must have 2 or 3 dimensions.
    :param segment_shape: A string specifying the shape to draw; `"c"` for
        circle (default) or `"s"` for sphere.

    :return: A tuple of numpy arrays representing the coordinates of the
        drawn mask shape. For 3D shapes, the tuple will include `z`, `y`,
        and `x` arrays. For 2D shapes, the tuple will include `y` and `x`
        arrays, omitting the `z` array.
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
    Generates the coordinates of pixels to draw a filled circle in a given array shape.
    Depending on the input `c` (2D or 3D), it generates either 2D (y, x) or 3D (z, y, x)
    coordinates restricted within the specified shape.

    The method uses the `skimage.draw.disk` function to calculate the circular region
    and adjusts for its position in the provided coordinate frame `c`. It ensures
    that all pixels remain within the bounds of the array defined by `shape`.

    :param r: Radius of the circle to be drawn.
    :param c: Tuple representing the center coordinates of the circle. In 2D, it requires (y, x),
        and for 3D, it requires (z, y, x).
    :param shape: Tuple representing the shape of the frame or array where the circle needs to be
        drawn. For 2D: (rows, columns). For 3D: (depth, rows, columns).

    :return: Coordinates of pixels forming the circle. A tuple of arrays (y, x) for 2D or
        (z, y, x) for 3D representing the indices of the circle's pixels.
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
    Generates the coordinates of a 3D spherical structure within a volume of specified shape.
    This function simulates a sphere using a 3D binary array, applies trimming to the sphere,
    and shifts its position within a given coordinate frame. The output consists of the
    adjusted coordinates of the sphere within the specified 3D volume.

    :param r: Radius of the sphere
    :type r: int
    :param c: Center coordinates of the sphere in the 3D volume
    :type c: tuple
    :param shape: Shape of the 3D volume to contain the sphere
    :type shape: tuple

    :return: Coordinates of the sphere within the 3D volume along z, y, and x axes
    :rtype: tuple of numpy.ndarray
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
