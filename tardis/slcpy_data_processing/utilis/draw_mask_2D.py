import numpy as np
from skimage import draw


def draw_2D(r: int,
            c: int,
            label_mask: np.ndarray,
            segment_color: list):
    """
    Module draw_label to construct shape of a label

    Args:
        r: radius of a circle in Angstrom
        c: point in 3D indicating center of a circle
        label_mask: array of a mask on which circle is drawn
        segment_color: single list value for naming drawn line in RGB
    """
    assert type(segment_color) == list
    assert label_mask.ndim in [2, 3, 4], \
        f'Unsupported dimmensions given {label_mask.ndim} expected [2, 3]!'

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
        f'Not enough colors were supply needed {nc} given {len(segment_color)}!'

    x = int(c[0])
    y = int(c[1])
    z = int(c[2])

    if z == nz:
        z = z - 1

    # Quickfix for 2D images
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
