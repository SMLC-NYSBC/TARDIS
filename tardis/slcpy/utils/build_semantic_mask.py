import numpy as np
from tardis.slcpy.utils.draw_mask_2D import draw_2D
from tardis.slcpy.utils.interpolation import interpolation_3D


def draw_semantic(mask_size: tuple,
                  coordinate: np.ndarray,
                  pixel_size: float,
                  circle_size=250,
                  multi_layer=False,
                  tqdm=True):
    """
    MODULE TO BUILD SEMANTIC MASK FROM CORRESPONDING COORDINATES

    Args:
        mask_size: Size of array that will hold created mask
        coordinate: Segmented coordinates of a shape [Label x X x Y x Z]
        pixel_size: Pixel size in Angstrom
        circle_size: Size of a circle the label mask in Angstrom
        multi_layer: single, or unique value for each lines
        tqdm: If True build with progress bar
    """
    assert coordinate.ndim == 2 and coordinate.shape[1] == 4, \
        'Included coordinate array is not of a correct shape.'

    label_mask = np.zeros(mask_size)
    if pixel_size == 0:
        pixel_size = 1

    r = round((circle_size / 2) / pixel_size)

    if multi_layer:
        label_mask = np.stack((label_mask,) * 3, axis=-1)
    else:
        segment_color = [1]

    # Number of segments in coordinates
    segments = np.unique(coordinate[:, 0])

    if tqdm:
        from tqdm import tqdm

        batch_iter = tqdm(range(len(segments)),
                          'Building semantic mask',
                          leave=False)
    else:
        batch_iter = range(len(segments))

    """Build semantic mask by drawing circle in 2D for each coordinate point"""
    for i in batch_iter:
        points = coordinate[np.where(coordinate[:, 0] == i)[0]][:, 1:]

        label = interpolation_3D(points)

        if multi_layer:
            segment_color = list(np.random.choice(range(255), size=3))

        """Draw 2D label"""
        for j in range(len(label)):
            c = label[j, :]  # Point center

            label_mask = draw_2D(r=r,
                                 c=c,
                                 label_mask=label_mask,
                                 segment_color=segment_color)
    return label_mask
