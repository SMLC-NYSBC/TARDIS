"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> SpindleTorch - Data_Processing - draw_mask

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
import numpy as np
from skimage import draw

from tardis.utils.errors import TardisError


def draw_2d(r: int,
            c: np.ndarray,
            label_mask: np.ndarray,
            segment_color: list) -> np.ndarray:
    """
    Module draw_label to construct shape of a label

    Args:
        r (int): radius of a circle in Angstrom.
        c (np.ndarray): point in 3D indicating center of a circle.
        label_mask (np.ndarray): array of a mask on which circle is drawn.
        segment_color (list): single list value for naming drawn line in RGB.

    Returns:
        np.ndarray: Binary mask.
    """
    assert isinstance(segment_color, list)
    assert label_mask.ndim in [2, 3, 4], \
        TardisError('113',
                    'tardis/spindletorch/data_processing/draw_mask_2D.py'
                    f'Unsupported dimensions given {label_mask.ndim} expected [2, 3]!')

    nz, ny, nx, nc = 0, 0, 0, 0
    dim = 0
    if label_mask.ndim == 4:  # 3D multi label
        nz, ny, nx, nc = label_mask.shape
        dim = 3
    elif label_mask.ndim == 3:
        if label_mask.shape[2] == 3:  # 2D multi label
            ny, nx, nc = label_mask.shape
            nz = 0
            dim = 2
        else:  # 3D single label
            nz, ny, nx = label_mask.shape
            nc = 1
            dim = 3
    elif label_mask.ndim == 2:  # 2D single label
        nc = 1
        nz = 0
        ny, nx = label_mask.shape
        dim = 2

    assert len(segment_color) == nc, \
        TardisError('113',
                    'tardis/spindletorch/data_processing/draw_mask_2D.py',
                    f'Incorrect color channel given {nc} but expect {len(segment_color)}')

    x = int(c[0])
    y = int(c[1])
    z = int(c[2])

    if z == nz:
        z = z - 1

    """Quickfix for 2D images"""
    if dim == 2:
        z = 0

    if z in range(nz + 1):
        cy, cx = draw.disk((y, x), r, shape=(ny, nx))

        if nc > 1:
            if dim == 3:
                for i in range(nc):
                    label_mask[z, cy, cx, i] = segment_color[i]
            elif dim == 2:
                for i in range(nc):
                    label_mask[cy, cx, i] = segment_color[i]
        else:
            if dim == 3:
                label_mask[z, cy, cx] = segment_color
            elif dim == 2:
                label_mask[cy, cx] = segment_color

    return label_mask
