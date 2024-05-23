#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import numpy as np

from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud


def test_build_pc():
    builder = BuildPointCloud()

    # Build rand data
    x = np.arange(start=0, stop=5, step=0.1)

    n_rnd = 50
    m = np.random.normal(loc=1, scale=0.3, size=n_rnd)
    b = np.random.normal(loc=5, scale=0.3, size=n_rnd)
    y = m * x[:, np.newaxis] + b
    y = np.where(y[:, :50] > 5.5, 1, 0)

    pc = builder.build_point_cloud(image=y)
    assert pc.ndim == 2
    assert len(pc) > 0

    pc = builder.build_point_cloud(image=y)
    assert pc.ndim == 2
    assert len(pc) > 0

    pc = builder.build_point_cloud(image=y, as_2d=True)
    assert pc.ndim == 2
    assert len(pc) > 0

    pc_hd, pc_ld = builder.build_point_cloud(image=y, down_sampling=1)
    assert pc_hd.ndim == 2 and pc_ld.ndim == 2
    assert len(pc_hd) > 0
    assert len(pc_ld) > 0
