#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import pytest
import numpy as np
import os
from tardis_em.analysis.analysis import (
    analyse_filaments,
    save_analysis,
)


# Test cases for `analyse_filaments`
def test_analyse_filaments_no_image():
    # Test with data only (no image)
    data = [np.array([[0, 1], [2, 3]])]
    result = analyse_filaments(data)

    assert len(result) == 7
    assert result[0] is not None  # length
    assert result[1] is not None  # curvature
    assert result[2] is not None  # tortuosity
    assert result[3] is None  # avg_intensity
    assert result[4] is None  # avg_length_intensity


def test_analyse_filaments_invalid_data():
    # Test with invalid input data type
    data = "invalid"
    with pytest.raises(TypeError):
        analyse_filaments(data)


# Test cases for `save_analysis`
def test_save_analysis_return_array():
    names = ["file1", "file2"]
    analysis = (
        [[1.0, 2.0], [3.0, 4.0]],  # lengths
        [[0.1, 0.2], [0.3, 0.4]],  # curvatures
        [[1.1, 1.2], [1.3, 1.4]],  # tortuosities
        [[0.5, 0.6], [0.7, 0.8]],  # avg_intensities
        [[0.05, 0.06], [0.07, 0.08]],  # avg_length_intensities
        [[10, 20], [30, 40]],  # sum_intensities
        [[5, 5], [10, 10]],  # sum_length_intensities
    )
    px_ = [1, 1]  # pixel size

    result = save_analysis(names, analysis, px_)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 10  # columns


def test_save_analysis_to_file(tmp_path):
    names = ["file1"]
    analysis = (
        [[1.0]],  # lengths
        [[0.1]],  # curvatures
        [[1.1]],  # tortuosities
        [[0.5]],  # avg_intensities
        [[0.05]],  # avg_length_intensities
        [[10]],  # sum_intensities
        [[5]],  # sum_length_intensities
    )

    save_dir = tmp_path
    save_analysis(names, analysis, save=str(save_dir))

    # Check that file exists
    files = os.listdir(str(save_dir))
    assert len(files) == 1
    assert files[0].endswith(".csv")
