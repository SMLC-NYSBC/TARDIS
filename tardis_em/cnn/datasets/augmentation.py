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
    CenterCrop is used to crop the center region of 2D or 3D image arrays.

    The purpose of this class is to allow cropping of image arrays to a predefined
    central size for further processing, training, or analysis. The class supports
    both 2D and 3D image formats. The usage involves initializing the crop size
    and then calling the instance with the input image array and an optional mask
    array.
    """

    def __init__(self, size: tuple):
        """
        Initializes the class with the given size parameter. This class is expected to be
        used for initializing object attributes to define the crop size for images. Only
        2D and 3D image crops are supported by this implementation. If the given size
        does not conform to these dimensions, an error is raised. If the size is 2D, it
        transforms the size into a 3D representation with a default depth of 0.

        :param size: The tuple representing the dimensions of the image crop. It must
            contain either 2 or 3 elements. The first element represents the depth for
            3D crops, or it is set to 0 for 2D ones.
        """
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
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Processes a given image array and an optional target mask array to crop their dimensions
        based on a provided target size. This will handle both 2D and 3D arrays for image inputs
        and ensures consistency with the target mask when provided. Throws an error if the input
        arrays are not 2D or 3D.

        :param x: The primary input image array to be processed.
        :param y: Optional secondary input array (e.g., target mask) to be cropped similarly
                  to the primary image.

        :return: A tuple containing the cropped image and mask if both x and y are provided,
                 or a cropped image array alone if only x is given.
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
    RandomFlip class for flipping 2D or 3D images and their corresponding label masks
    along a random axis. This class is commonly used in data augmentation for machine
    learning models, particularly in image processing tasks.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly flips multidimensional numpy arrays along specific axes.

        The method determines a random axis based on the generated random state,
        which depends on the input dimensions. The flipping operation is conducted
        along the selected axis for both input arrays.

        :param x:
            A numpy array that is multidimensional. Array to perform flipping along
            a random axis-derived dimension.

        :param y:
            A numpy array that is multidimensional. Array to perform flipping along
            the same axis-derived dimension as `x`.

        :return:
            A tuple of numpy arrays, where both are flipped along the selected axis.
            The flipping axis is determined randomly based on the input dimensions.

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
    Randomize rotation.

    This class provides a mechanism to apply random rotations to 2D/3D image arrays and their
    corresponding label mask arrays. It can determine the rotation angle and direction randomly,
    then apply the transformation to the input data. The main purpose is to augment the data
    by introducing random rotations.
    """

    def __init__(self):
        """Randomize rotation"""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform random rotation of the provided numpy arrays `x` and `y` based on specified
        conditions of rotation and direction. Depending on a randomly determined rotation angle and
        direction, the function rotates the arrays along appropriate axes. The outcome of this
        process ensures variability in the data transformation; especially useful in scenarios
        such as data augmentation for machine learning tasks involving image data.

        :param x: Input numpy array representing the source data to be rotated. It may have
            dimensions of 2D or 3D.
        :param y: Input numpy array representing the corresponding label data to be rotated.
            It should have the same dimensional structure as `x`.
        :return: A tuple of numpy arrays where both `x` and `y` have undergone the same
            random rotation transformation.
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
    ComposeRandomTransformation applies a random sequence of transformations
    from a given list to the input data. The number of transformations applied
    in a sequence is determined randomly between 1 and 3. This class is designed
    to augment data by applying non-deterministic transformations to images and
    their corresponding label masks during preprocessing.
    """

    def __init__(self, transformations: list):
        """
        Applies a set of transformations to data with random repetitions.

        A utility class used to handle a sequence of transformations. The transformations
        are repeatedly applied to the data upon request, where the number of repetitions is
        chosen randomly between 1 and 3.

        :param transformations: The collection of transformations to apply.
        """
        self.transforms = transformations
        self.random_repetition = np.random.randint(1, 4)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly applies a series of transformations to the provided input data ``x`` and ``y``. Each transformation
        is selected randomly from a predefined collection of transformations for a specified number of repetitions.
        The method modifies the input data through these transformations and returns the final transformed data.

        :param x: Input data to be transformed
        :type x: numpy.ndarray
        :param y: Corresponding labels or data linked with ``x`` to be transformed
        :type y: numpy.ndarray

        :return: A tuple containing the transformed ``x`` and corresponding ``y``
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
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
    Preprocesses an image with optional transformations, resizing, and mask handling. The function
    supports both 2D and 3D images. It resizes the image and optionally a mask to the desired size,
    applies random transformations, and prepares the data for model input by expanding its
    dimensions as needed.

    :param image: The input image to preprocess. Must be a NumPy array of 2 or 3 dimensions.
    :param transformation: A boolean flag to determine whether random transformations such as
        flipping and rotation should be applied to the image.
    :param size: The target size for resizing the image. Can be an integer for uniform scaling
        or a tuple for specific dimensions.
    :param mask: Optional. A mask corresponding to the input image. Must match the dimensions
        of the image.
    :param output_dim_mask: Optional. The number of desired output dimensions for the mask. Default
        is 1.
    :return: If a mask is provided, returns a tuple containing the preprocessed image and mask.
        If no mask is provided, returns only the preprocessed image.

    :raises Exception: Raised when a tuple size for resizing does not represent uniform scaling
        (e.g., dimensions that are non-uniform or inconsistent).
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
