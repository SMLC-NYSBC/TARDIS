from tardis.dist_pytorch.utils.build_point_cloud import ImageToPointCloud
import numpy as np

def test_build_PC():
    builder = ImageToPointCloud()

    # Build rand data
    x = np.arange(start=0, stop=5, step=0.1)
    n_data = len(x)
    n_rnd = 50
    m = np.random.normal(loc=1, scale=0.3, size=n_rnd) 
    b = np.random.normal(loc=5, scale=0.3, size=n_rnd)
    y = m * x[:, np.newaxis] + b
    y = np.where(y[:, :50] > 5.5, 1, 0)


    pc = builder(image=y)

    assert pc.ndim == 2
    assert len(pc) > 0
