"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> SpindleTorch - Data_Processing - semantic_mask

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
import cv2
import numpy as np

from tardis.spindletorch.data_processing.draw_mask_2D import draw_2d
from tardis.spindletorch.data_processing.interpolation import interpolation
from tardis.utils.errors import TardisError


def fill_gaps_in_semantic(image: np.ndarray) -> np.ndarray:
    """
    !DEPRECIATED! Restore semantic mask after interpolation when some labels
        where up- down-scale incorrectly.

    Args:
        image (np.ndarray): Mask data.

    Returns:
        np.ndarray: Fixed mask with fill out holes.
    """
    if image.ndim == 3:
        for id, _ in enumerate(image):
            des = np.array(image[id, :], dtype=image.dtype)
            contour, _ = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contour:
                cv2.drawContours(des, [cnt], 0, 1, -1)

            gray = cv2.bitwise_not(des)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            res = np.where(res == 255, 0, 1)
            image[id, :] = res

        return image
    else:
        des = np.array(image, dtype=image.dtype)
        contour, _ = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des, [cnt], 0, 1, -1)

        gray = cv2.bitwise_not(des)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        res = np.where(res == 255, 0, 1)

        return res


def draw_semantic(mask_size: tuple,
                  coordinate: np.ndarray,
                  pixel_size: float,
                  circle_size=250,
                  multi_layer=False) -> np.ndarray:
    """
    Module to build semantic mask from corresponding coordinates

    Args:
        mask_size (tuple): Size of array that will hold created mask.
        coordinate (np.ndarray): Segmented coordinates of a shape [Label x X x Y x (Z)].
        pixel_size (float): Pixel size in Angstrom.
        circle_size (int): Size of a circle the label mask in Angstrom.
        multi_layer (bool): single, or unique value for each line.

    Returns:
        np.ndarray: Binary mask with drawn all coordinates as lines.
    """
    assert coordinate.ndim == 2 and coordinate.shape[1] in [3, 4], \
        TardisError('113',
                    'tardis/spindletorch/data_processing/semantic_mask.py',
                    'Coordinates are of not correct shape, expected: '
                    f'shape [Label x X x Y x (Z)] but {coordinate.shape} given!')

    label_mask = np.zeros(mask_size)
    if pixel_size == 0:
        pixel_size = 1

    r = round((circle_size / 2) / pixel_size)

    segment_color = [1]
    if multi_layer:
        label_mask = np.stack((label_mask,) * 3, axis=-1)

    # Number of segments in coordinates
    segments = np.unique(coordinate[:, 0])

    """Build semantic mask by drawing circle in 2D for each coordinate point"""
    for i in range(len(segments)):
        points = coordinate[np.where(coordinate[:, 0] == i)[0]][:, 1:]

        label = interpolation(points)

        if multi_layer:
            segment_color = list(np.random.choice(range(255), size=3))

        """Draw 2D label"""
        for j in range(len(label)):
            c = label[j, :]  # Point center

            label_mask = draw_2d(r=r,
                                 c=c,
                                 label_mask=label_mask,
                                 segment_color=segment_color)
    return label_mask.astype(np.uint8)
