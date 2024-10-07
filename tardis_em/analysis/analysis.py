#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from datetime import datetime
from os.path import join
from typing import Union, List, Tuple
import numpy as np

from tardis_em.analysis.geometry_metrics import (
    length_list,
    curvature_list,
    intensity_list,
)

from tardis_em.analysis.mt_classes import (
    assign_filaments_to_poles,
    pick_pole_to_surfaces,
)
from tardis_em.utils.errors import TardisError

from tardis_em._version import version


def analyse_filaments(
    data: Union[np.ndarray, List, Tuple], image: Union[np.ndarray, List, Tuple] = None
) -> tuple:
    """

    Args:
        data (np.ndarray, List, Tuple): Directory to file path
        image (np.ndarray, None): Optional image to compute intensity along predicted splines.
    """
    if isinstance(data, np.ndarray):
        data = [data]
    if image is not None:
        if isinstance(image, np.ndarray):
            image = [image]

    length, curvature, tortuosity = [], [], []
    avg_intensity, sum_intensity = [], []
    avg_length_intensity, sum_length_intensity = [], []

    for id_, d_ in enumerate(data):
        length.append(length_list(d_))

        curvature_, tortuosity_ = curvature_list(d_, tortuosity_=True, mean_=True)
        curvature.append(curvature_)
        tortuosity.append(tortuosity_)

        if image is not None:
            intensity = intensity_list(d_, image[id_])

            avg_intensity_ = [np.mean(i).item(0) for i in intensity]
            avg_intensity.append(avg_intensity_)

            avg_length_intensity_ = [i / l for l, i in zip(length[id_], avg_intensity_)]
            avg_length_intensity.append(avg_length_intensity_)

            sum_intensity_ = [np.sum(i).item(0) for i in intensity]
            sum_intensity.append(sum_intensity_)

            sum_length_intensity_ = [i / l for l, i in zip(length[id_], sum_intensity_)]
            sum_length_intensity.append(sum_length_intensity_)

    if image is not None:
        return (
            length,
            curvature,
            tortuosity,
            avg_intensity,
            avg_length_intensity,
            sum_intensity,
            sum_length_intensity,
        )
    else:
        return length, curvature, tortuosity, None, None, None, None


def save_analysis(
    names: Union[List, Tuple], analysis: Union[List, Tuple], px_=None, save: str = None
):

    length, curvature, tortuosity = analysis[0], analysis[1], analysis[2]
    avg_intensity, avg_length_intensity = analysis[3], analysis[4]
    sum_intensity, sum_length_intensity = analysis[5], analysis[6]

    rows = np.sum([len(i) for i in length]).item(0)
    analysis_file = np.zeros((rows, 10), dtype=object)
    analysis_file[0, :] = [
        "File_Name",
        "No. of Filament",
        "Pixel_Size [nm]",
        "Length [nm]",
        "Curvature [0-inf]",
        "Tortuosity [1-inf]",
        "Avg. Intensity [U]",
        "Avg. Intensity / Length [U/nm]",
        "Sum. Intensity [U]",
        "Avg. Intensity / Length [U/nm]",
    ]

    iter_ = 0
    for i in range(len(length)):  # Iterate throw every file
        for j in range(len(length[i])):
            analysis_file[iter_, :] = [
                names[i][:-21],
                str(j),
                str(1.0) if px_ is None else px_[i][j],
                str(length[i][j]),
                str(curvature[i][j]),
                str(tortuosity[i][j]),
                "" if avg_intensity[i] is None else str(avg_intensity[i][j]),
                (
                    ""
                    if avg_length_intensity[i] is None
                    else str(avg_length_intensity[i][j])
                ),
                "" if sum_intensity[i] is None else str(sum_intensity[i][j]),
                (
                    ""
                    if sum_length_intensity[i] is None
                    else str(sum_length_intensity[i][j])
                ),
            ]
            iter_ += 1

    if save is not None:
        date = datetime.now()
        tardis_version = version

        file_name = f"TARDIS_V{tardis_version}_{date.day}_{date.month}_{date.year}-{date.hour}_{date.minute}.csv"

        np.savetxt(
            join(save, file_name), analysis_file.astype(str), fmt="%s", delimiter=","
        )
        return
    return analysis_file


def analyse_filaments_list(
    data: Union[List, Tuple],
    names_: Union[List, Tuple],
    path: str,
    images: Union[List, Tuple] = None,
    px_: Union[List, Tuple] = None,
):
    """
    Arge:
        data (list, tuple):
        names_ (list, tuple):
        path (str):
        images (list, tuple):
        px_ (list, tuple):
    """

    if images is not None:
        assert_ = len(data) == len(images)
    else:
        assert_ = len(data) > 0

    if not assert_:
        TardisError(
            id_="117",
            py="analysis.analysis:analise_filaments_list",
            desc="List of analysed files do not match or is 0",
            warning_=False,
        )

    if images is None:
        images = [None for _ in range(len(data))]

    save_analysis(names_, analyse_filaments(data, images), px_=px_, save=path)


def analyse_mt_classes(
    filaments: np.ndarray, poles: np.ndarray, vertices: list, triangles: list
):
    # Sort poles to surfaces
    poles = pick_pole_to_surfaces(poles, vertices)
    # Sort filaments to poles
    filament_poles = assign_filaments_to_poles(filaments, poles)

    """
     Get plus and minus ends
    """
    plus_ends = []
    minus_ends = []
    for f in filament_poles:
        _, first_indices = np.unique(f[:, 0], return_index=True)
        first_points = first_indices

        last_points = [i + len(f[f[:, 0] == i, :]) - 1 for i in first_points]

        plus_ends.append(first_points)
        minus_ends.append(last_points)

    """
    Return:
        - List of filaments ID,X,Y,Z for each pole
        - Indices of plus and minus ends per pole
        - 
    """
    return (
        filament_poles,
        (plus_ends, minus_ends),
    )
