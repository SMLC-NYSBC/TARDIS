from typing import Optional

import numpy as np


class CenterCrop:
    """
        Rescale the image and mask to a given size for 3D [DxHxW],
         or 4D images [CxDxHxW]

    Object as an input required
            x: image 3D or 4D arrays
            y: target/mask 3D or 4D arrays

    Args:
        size: output size of image in DHW
    """

    def __init__(self,
                 size: tuple):
        assert len(size) == 3
        self.size = size

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        assert x.ndim in [3, 4] and y.ndim in [3, 4]

        if x.ndim == 3:
            d, h, w = x.shape
        else:
            d, h, w = x.shape[1:]

        down_d = int(d // 2 + self.size[0] // 2)
        up_d = int(d // 2 - self.size[0] // 2)
        bottom_h = int(h // 2 + self.size[1] // 2)
        top_h = int(h // 2 - self.size[1] // 2)
        right_w = int(w // 2 + self.size[2] // 2)
        left_w = int(w // 2 - self.size[2] // 2)

        if x.ndim == 3 and y.ndim == 3:
            return x[up_d:down_d, top_h:bottom_h, left_w:right_w], \
                y[up_d:down_d, top_h:bottom_h, left_w:right_w]
        elif x.ndim == 4 and y.ndim == 4:
            return x[:, up_d:down_d, top_h:bottom_h, left_w:right_w], \
                y[:, up_d:down_d, top_h:bottom_h, left_w:right_w]
        elif x.ndim == 4 and y.ndim == 3:
            return x[:, up_d:down_d, top_h:bottom_h, left_w:right_w], \
                y[up_d:down_d, top_h:bottom_h, left_w:right_w]
        else:
            return x[up_d:down_d, top_h:bottom_h, left_w:right_w], \
                y[:, up_d:down_d, top_h:bottom_h, left_w:right_w]


class SimpleNormalize:
    """
    Normalize image vale by simple division

        Object as an input required
            x: image or target 3D or 4D arrays
    """

    def __call__(self,
                 x: np.ndarray):
        assert x.min() >= 0 and x.max() <= 255

        return x / 255


class MinMaxNormalize:
    """
    Normalize image vale between 0 and 1 and

    Object as an input required
        x: image or target 3D or 4D arrays

    Args:
        min: Minimal value for initialize normalization e.g. 0
        max: Maximal value for initialize normalization e.g. 255
    """

    def __init__(self,
                 min: int,
                 max: int):
        assert max > min
        self.min = int(min)
        self.max = int(max)

        self.range = self.max - self.min

    def __call__(self, x):
        return (x - self.min) / self.range


class RandomFlip:
    """
    Perform 180 degree flip randomly in z,x or y axis for 3D or 4D

    Object as an input required
        x: image 3D or 4D arrays
        y: target/mask 3D or 4D arrays
    """

    def __init__(self):
        self.random_state = np.random.randint(0, 2)
        # 0 is z axis, 1 is x axis, 2 is y axis

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        if self.random_state == 0:
            return np.flipud(x), np.flipud(y)
        return np.fliplr(x), np.fliplr(y)


class RandomRotation:
    """
    Perform 90, 180 or 270 degree rotation for 3D or 4D in left or right

    Object as an input required
        x: image 3D or 4D arrays
        y: target/mask 3D or 4D arrays
    """

    def __init__(self):
        self.random_rotation = np.random.randint(0, 2)
        # 0 is 90, 1 is 180, 2 is 270
        self.random_direction = np.random.randint(0, 1)
        # 0 is left, 1 is right

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):

        if self.random_direction == 0:
            self.random_rotation *= -1

        if x.ndim == 3:
            x_dim = (1, 2)
        else:
            x_dim = (2, 3)

        if y.ndim == 3:
            y_dim = (1, 2)
        else:
            y_dim = (2, 3)

        x = np.rot90(x, self.random_rotation, x_dim)
        y = np.rot90(y, self.random_rotation, y_dim)

        return x, y


class ComposeRandomTransformation:
    """
    Double wrapper for image and mask to perform random transformation

    Object as an input required
        x: image 3D or 4D arrays
        y: target/mask 3D or 4D arrays

    Args:
        transformations: list of transforms objects from which single
            or multiple transformations will be selected
    """

    def __init__(self,
                 transformations: list):
        self.transforms = transformations
        self.random_repetition = np.random.randint(0, len(transformations))

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        if self.random_repetition > 0:
            for i in range(self.random_repetition):
                random_transf = np.random.randint(0, len(self.transforms))
                transform = self.transforms[random_transf]
                x, y = transform(x, y)
        return x, y


def preprocess(image: np.ndarray,
               mask: np.ndarray,
               normalization: str,
               transformation: bool,
               size: Optional[tuple] = None,
               output_dim_mask=1):
    """
        Module to transform dataset and prepare them for learning

    Args:
        image: 3D tiff file with image data
        mask: 3d tiff file with semantic labels
        normalization: normalize object to bring img to 0 and 1 values
        transformation: perform transformation on img and mask with
            same random effect
        size: image size output for center crop
        output_dim_mask: output channel diemsions for label mask
    """

    assert len(image.shape) in [3, 4]
    z, h, w = image.shape[:3]

    """ resize image """
    if size != "None" and size is not None:
        if (z, h, w) != (size, size, size):
            # resize image
            crop = CenterCrop(size)
            image, mask = crop(image, mask)

    """ Transform image randomly """
    if transformation:
        transformations = [RandomFlip(), RandomRotation()]
        random_transform = ComposeRandomTransformation(transformations)

        image, mask = random_transform(image, mask)

    """ Normalize image for value between 1 and 0 """
    if image.max() > 1:
        if normalization == "simple":
            normalization = SimpleNormalize()
        elif normalization == "minmax":
            normalization = MinMaxNormalize(image.min(), image.max())

        image = normalization(image)

    """ Expand dimension order from DHW to CDHW """
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=0)

        if output_dim_mask != mask.shape[0]:
            mask = np.tile(mask, [output_dim_mask, 1, 1, 1])

    return image, mask
