import numpy as np


class CenterCrop:
    """
    Rescale the image and mask to a given size for 3D [DxHxW],
    or 2D images [HxW]

    Args:
        size: output size of image in DHW/HW
        x: image 2D or 3D arrays
        y: target/mask 2D or 3D arrays
    """

    def __init__(self,
                 size: tuple):
        assert len(size) in [2, 3]
        self.size = size

        if len(self.size) == 2:
            self.size = (0, size[0], size[1])

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        """Evaluate data structure"""
        assert x.ndim in [2, 3] and y.ndim in [2, 3]

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
        if x.ndim == 3 and y.ndim == 3:
            return x[up_d:down_d, top_h:bottom_h, left_w:right_w], \
                y[up_d:down_d, top_h:bottom_h, left_w:right_w]
        elif x.ndim == 2 and y.ndim == 2:
            return x[top_h:bottom_h, left_w:right_w], \
                y[top_h:bottom_h, left_w:right_w]


class SimpleNormalize:
    """
    Normalize image vale by simple division

    Args:
        x: image or target
    """

    def __call__(self,
                 x: np.ndarray):
        assert x.min() >= 0 and x.max() <= 255

        return x / 255


class MinMaxNormalize:
    """
    Normalize image vale between 0 and 1

    Args:
        min: Minimal value for initialize normalization e.g. 0
        max: Maximal value for initialize normalization e.g. 255
        x: image or target
    """

    def __init__(self,
                 min: int,
                 max: int):
        assert max > min
        self.min = int(min)
        self.max = int(max)

        self.range = self.max - self.min

    def __call__(self,
                 x: np.ndarray):
        return (x - self.min) / self.range


class RandomFlip:
    """
    Perform 180 degree flip randomly in z,x or y axis for 3D or 4D

    Args:
        x: image 2D or 3D arrays
        y: target/mask 2D or 3D arrays
    """

    def __init__(self):
        self.random_state = np.random.randint(0, 2)
        # 0 is z axis, 1 is x axis, 2 is y axis for 3D
        # 0 is x axis, 1 is y axis for 2D

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        if x.ndim == 2 or y.ndim == 2:
            if self.random_state == 2:
                self.random_state = np.random.randint(0, 1)

        return np.flip(x, self.random_state), np.flip(y, self.random_state)


class RandomRotation:
    """
    Perform 90, 180 or 270 degree rotation for 2D or 3D in left or right

    Args:
        x: image 2D or 3D arrays
        y: target/mask 2D or 3D arrays
    """

    def __init__(self):
        """Randomize rotation"""
        self.random_rotation = np.random.randint(0, 3)
        # 0 is 90, 1 is 180, 2 is 270
        self.random_direction = np.random.randint(0, 1)
        # 0 is left, 1 is right

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        """Evaluate data structure"""
        if self.random_direction == 0:
            if x.ndim == 2:
                axis = (0, 1)
            elif x.ndim == 3:
                axis = (1, 2)
        elif self.random_direction == 1:
            if x.ndim == 2:
                axis = (1, 0)
            elif x.ndim == 3:
                axis = (2, 1)

        return np.rot90(x, self.random_rotation, axis), \
            np.rot90(y, self.random_rotation, axis)


class ComposeRandomTransformation:
    """
    Double wrapper for image and mask to perform random transformation

    Args:
        transformations: list of transforms objects from which single
            or multiple transformations will be selected
        x: image 2D or 3D arrays
        y: target/mask 2D or 3D arrays
    """

    def __init__(self,
                 transformations: list):
        self.transforms = transformations
        self.random_repetition = np.random.randint(0, len(transformations))

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        """Run randomize transformation"""
        if self.random_repetition > 0:
            for _ in range(self.random_repetition):
                random_transf = np.random.randint(0, len(self.transforms))
                transform = self.transforms[random_transf]
                x, y = transform(x, y)

        return x, y


def preprocess(image: np.ndarray,
               mask: np.ndarray,
               normalization: str,
               transformation: bool,
               size: int,
               output_dim_mask=1):
    """
        Module to transform dataset and prepare them for learning

    Args:
        image: 2D/3D array with image data
        mask: 2D/3D array with semantic labels
        normalization: normalize object to bring img to 0 and 1 values
        transformation: perform transformation on img and mask with
            same random effect
        size: image size output for center crop
        output_dim_mask: output channel dimensions for label mask
    """
    """Evaluate data structure"""
    assert image.ndim in [2, 3]
    if image.ndim == 3:
        z, h, w = image.shape
        dim = 3
    else:
        h, w = image.shape
        dim = 2

    """Resize image"""
    if dim == 3:
        if (z, h, w) != (size, size, size):
            # resize image
            crop = CenterCrop((size, size, size))
            image, mask = crop(image, mask)
        elif dim == 2:
            if (h, w) != (size, size):
                # resize image
                crop = CenterCrop((size, size))
                image, mask = crop(image, mask)

    """Transform image randomly"""
    if transformation:
        transformations = [RandomFlip(), RandomRotation()]
        random_transform = ComposeRandomTransformation(transformations)

        image, mask = random_transform(image, mask)

    """Normalize image for value between 1 and 0"""
    if image.max() > 1:
        if normalization == "simple":
            normalization = SimpleNormalize()
        elif normalization == "minmax":
            normalization = MinMaxNormalize(image.min(), image.max())

        image = normalization(image)

    """Expand dimension order for HW / DHW to CHW / CDHW"""
    image = np.expand_dims(image, axis=0)
    mask = np.expand_dims(mask, axis=0)

    if output_dim_mask != mask.shape[0]:
        mask = np.tile(mask, [output_dim_mask, 1, 1, 1])

    return image, mask
