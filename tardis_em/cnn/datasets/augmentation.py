#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from tardis_em.utils.errors import TardisError


class CenterCrop:
    """
    CENTER CROP ARRAY

    Rescale the image and mask to a given size for 3D [DxHxW], or 2D images [HxW].

    Args:
        size (tuple): Output size of image in DHW/HW.
    """

    def __init__(self, size: tuple):
        if len(size) not in [2, 3]:
            TardisError(
                "146",
                "tardis_em/cnn/dataset/augmentation.py",
                "Image crop supported only for 3D and 2D! " f"But {size} was given.",
            )
        self.size = size

        if len(self.size) == 2:
            self.size = (0, size[0], size[1])

    def __call__(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call for centre crop.

        Args:
            x (np.ndarray): Image 2D or 3D arrays.
            y (np.ndarray, optional): Optional label mask 2D or 3D arrays.

        Returns:
            np.ndarray: Cropped array.
        """
        if x.ndim not in [2, 3]:
            TardisError(
                "146",
                "tardis_em/cnn/dataset/augmentation.py",
                "Image crop supported only for 3D and 2D!",
            )
        if y is not None:
            if y.ndim not in [2, 3]:
                TardisError(
                    "146",
                    "tardis_em/cnn/dataset/augmentation.py",
                    "Image crop supported only for 3D and 2D!",
                )

        if x.ndim == 3:
            d, h, w = x.shape

            down_d = int(d // 2 + self.size[0] // 2)
            up_d = int(d // 2 - self.size[0] // 2)

        else:
            h, w = x.shape

        """"Calculate padding for crop"""
        bottom_h = int(h // 2 + self.size[1] // 2)
        top_h = int(h // 2 - self.size[1] // 2)
        right_w = int(w // 2 + self.size[2] // 2)
        left_w = int(w // 2 - self.size[2] // 2)

        """Crop"""
        if y is not None:
            if x.ndim == 3 and y.ndim == 3:
                return (
                    x[up_d:down_d, top_h:bottom_h, left_w:right_w],
                    y[up_d:down_d, top_h:bottom_h, left_w:right_w],
                )
            elif x.ndim == 2 and y.ndim == 2:
                return (
                    x[top_h:bottom_h, left_w:right_w],
                    y[top_h:bottom_h, left_w:right_w],
                )
        else:
            if x.ndim == 3:
                return x[up_d:down_d, top_h:bottom_h, left_w:right_w]
            elif x.ndim == 2:
                return x[top_h:bottom_h, left_w:right_w]


class RandomFlip:
    """
    180 RANDOM FLIP ARRAY

    Perform 180 degree flip randomly in z,x or y axis for 3D or 4D

        - 0 is z axis, 1 is x axis, 2 is y axis for 3D
        - 0 is x axis, 1 is y axis for 2D
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call for random flip.

        Args:
            x: 2D/3D image array.
            y: 2D/3D label mask array.

        Returns:
            np.ndarray, np.ndarray: Flipped array.
        """
        random_state = np.random.randint(0, 9)
        if x.ndim == 2 or y.ndim == 2:
            random_state = np.random.randint(0, 6)

        if random_state <= 3:
            random_state = 0
        elif 6 >= random_state > 3:
            random_state = 1
        elif 9 >= random_state > 6:
            random_state = 2

        return np.flip(x, random_state), np.flip(y, random_state)


class RandomRotation:
    """
    MULTIPLE 90 RANDOM ROTATION

    Perform 90, 180 or 270-degree rotation for 2D or 3D in left or right

        - 0 is 90, 1 is 180, 2 is 270
        - 0 is left, 1 is right
    """

    def __init__(self):
        """Randomize rotation"""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call for random rotation.

        Args:
            x: 2D/3D image array.
            y: 2D/3D label mask array.

        Returns:
            np.ndarray, np.ndarray: Rotated array.
        """
        random_rotation = np.random.randint(0, 9)
        if random_rotation <= 3:
            random_rotation = 0
        elif 6 >= random_rotation > 3:
            random_rotation = 1
        elif 9 >= random_rotation > 6:
            random_rotation = 2

        random_direction = np.random.randint(0, 10)
        if random_direction <= 3:
            random_direction = 0
        else:
            random_direction = 1

        # Evaluate data structure
        if random_direction == 0:
            if x.ndim == 2:
                axis = (0, 1)
            elif x.ndim == 3:
                axis = (1, 2)
        elif random_direction == 1:
            if x.ndim == 2:
                axis = (1, 0)
            elif x.ndim == 3:
                axis = (2, 1)

        return np.rot90(x, random_rotation, axis), np.rot90(y, random_rotation, axis)


class ComposeRandomTransformation:
    """
    RANDOM TRANSFORMATION WRAPPER

    Double wrapper for image and mask to perform random transformation

    Args:
        transformations: list of transforms objects from which single
            or multiple transformations will be selected.
    """

    def __init__(self, transformations: list):
        self.transforms = transformations
        self.random_repetition = np.random.randint(1, 4)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call for random transformation.

        Args:
            x: 2D/3D image array.
            y: 2D/3D label mask array.

        Returns:
            np.ndarray, np.ndarray: Transformed array.
        """
        # Run randomize transformation
        for _ in range(self.random_repetition):
            random_transform = np.random.randint(0, len(self.transforms))
            transform = self.transforms[random_transform]
            x, y = transform(x, y)

        return x, y


def preprocess(
    image: np.ndarray,
    transformation: bool,
    size: Optional[Union[tuple, int]] = int,
    mask: Optional[np.ndarray] = None,
    output_dim_mask=1,
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    Module to augment dataset.

    Args:
        image (np.ndarray): 2D/3D array with image data.
        mask (np.ndarray, optional): 2D/3D array with semantic labels.
        transformation (bool): If True perform transformation on img and mask with
            same random effect.
        size (tuple, int): Image size output for center crop.
        output_dim_mask (int): Number of output channel dimensions for label mask.

    Returns:
        np.ndarray: Image and optionally label mask after transformation.
    """
    # Check if image is 2D or 3D
    if image.ndim not in [2, 3]:
        TardisError(
            "146",
            "tardis_em/cnn/dataset/augmentation.py",
            "Image crop supported only for 3D and 2D!",
        )

    if isinstance(size, tuple):
        if sum(size) / len(size) == size[0]:  # Check if image has uniform size
            size = size[0]
        else:
            raise Exception

    """Resize image if image != size"""
    if image.ndim == 3:
        if image.shape != (size, size, size):
            # resize image
            crop = CenterCrop((size, size, size))
            if mask is not None:
                image, mask = crop(image, mask)
            else:
                image = crop(image)
    elif image.ndim == 2:
        if image.shape != (size, size):
            # resize image
            crop = CenterCrop((size, size))
            if mask is not None:
                image, mask = crop(image, mask)
            else:
                image = crop(image)

    """Transform image randomly"""
    if transformation:
        random_transform = ComposeRandomTransformation([RandomFlip(), RandomRotation()])
        if mask is None:
            image, _ = random_transform(image, image)
        else:
            image, mask = random_transform(image, mask)

    """Expand dimension order for HW / DHW to CHW / CDHW"""
    image = np.expand_dims(image, axis=0)

    if mask is not None:
        mask = np.expand_dims(mask, axis=0)

        if output_dim_mask != mask.shape[0]:
            mask = np.tile(mask, [output_dim_mask, 1, 1, 1])
        return image, mask
    else:
        return image
