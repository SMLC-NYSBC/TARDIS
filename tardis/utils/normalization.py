#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import numpy as np

from skimage import exposure


class SimpleNormalize:
    """
    SIMPLE IMAGE NORMALIZATION

    Take int8-int32 image file with 0 - 255 value. All image value are spread
    between 0 - 1.
    """

    def __call__(self,
                 x: np.ndarray) -> np.ndarray:
        """
        Call for image normalization.

        Args:
            x (np.ndarray): Image array.

        Returns:
            np.ndarray: Normalized image.
        """
        if x.dtype == np.uint8:
            x = x / 255
        elif x.dtype == np.int8:
            x = (x + 128) / 255
        elif x.dtype == np.uint16:
            x = x / 65535
        elif x.dtype == np.int16:
            x = (x + 32768) / 65535
        elif x.dtype == np.uint32:
            x = x / 4294967295
        else:
            x = (x + 2147483648) / 4294967295

        return x.astype(np.float32)


class MinMaxNormalize:
    """
    IMAGE NORMALIZATION BETWEEN MIN AND MAX VALUE
    """

    def __call__(self,
                 x: np.ndarray) -> np.ndarray:
        """
        Call for normalization.

        Args:
            x (np.ndarray): Image or label mask.

        Returns:
            np.ndarray: Normalized array.
        """
        MIN = np.min(x)
        MAX = np.max(x)

        if MIN >= 0:
            if MAX <= 1:
                return x.astype(np.float32)
            elif MAX <= 255:
                x = (x - 0) / 255
            elif MAX <= 65535:
                x = (x - 0) / 65535
            elif MAX <= 4294967295:
                x = (x - 0) / 4294967295
        elif 0 > MIN >= -1 and MAX <= 1:
            x = (x + 1) / 2

        return x.astype(np.float32)


class RescaleNormalize:
    """
    NORMALIZE IMAGE VALUE USING Skimage

    Rescale intensity with top% and bottom% percentiles as default

    Args:
        clip_range: Histogram percentiles range crop.
    """

    def __init__(self,
                 clip_range=(2, 98)):
        self.range = clip_range

    def __call__(self,
                 x: np.ndarray) -> np.ndarray:
        """
        Call for normalization.

        Args:
            x (np.ndarray): Image or label mask.

        Returns:
            np.ndarray: Normalized array.
        """
        p2, p98 = np.percentile(x, self.range)
        if x.dtype == np.uint8:
            if p98 >= 250:
                p98 = 256
            if p2 <= 5:
                p2 = 0

        return exposure.rescale_intensity(x, in_range=(p2, p98))
