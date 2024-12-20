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
import pandas as pd

from tardis_em.analysis.geometry_metrics import (
    length_list,
    curvature_list,
    intensity_list,
)

from tardis_em.analysis.mt_classification.mt_classes import (
    assign_filaments_to_poles,
    pick_pole_to_surfaces,
)
from tardis_em.utils.errors import TardisError

from tardis_em._version import version


def analyse_filaments(
    data: Union[np.ndarray, List, Tuple],
    image: Union[np.ndarray, List, Tuple] = None,
    thickness=1,
    px=None,
) -> tuple:
    """
    Analyzes filament data to compute various attributes including length, curvature,
    tortuosity, and intensity-related metrics. If image data is provided, intensity
    metrics such as average intensity, average length intensity, total intensity,
    and total length intensity are computed. Lengths can be scaled based on pixel size
    (px) if specified.

    :param data: Filament data used for analysis. Can be a numpy array or a sequence
        containing numpy arrays.
    :type data: Union[np.ndarray, List, Tuple]
    :param image: Optional image data corresponding to the filaments. Used for
        computing intensity metrics. Can be a numpy array or a sequence containing
        numpy arrays.
    :type image: Union[np.ndarray, List, Tuple], optional
    :param thickness: Thickness of filaments used for intensity calculation.
        Defaults to 1.
    :type thickness: int, optional
    :param px: Scaling factors for pixel size to convert lengths into physical
        units. If None, lengths are computed directly without scaling. Defaults to
        None.
    :type px: Sequence, optional

    :return: A tuple containing:

        - length (list): Computed lengths of the filaments.
        - curvature (list): Computed mean curvatures of the filaments.
        - tortuosity (list): Computed mean tortuosity of the filaments.
        - avg_intensity (list): Computed average intensity per filament, or None
          if no image data is provided.
        - avg_length_intensity (list): Computed average filament intensity
          normalized by filament length, or None if no image data is provided.
        - sum_intensity (list): Computed total intensity per filament, or None
          if no image data is provided.
        - sum_length_intensity (list): Computed total filament intensity normalized
          by filament length, or None if no image data is provided.
    :rtype: tuple
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
        if px is None:
            length.append(length_list(d_))
        else:
            length.append([i * px[id_] for i in length_list(d_)])  # length in A

        curvature_, tortuosity_ = curvature_list(d_, tortuosity_b=True, mean_b=True)
        curvature.append(curvature_)
        tortuosity.append(tortuosity_)

        if image is not None:
            intensity = intensity_list(d_, image[id_], thickness)

            avg_intensity_ = [np.mean(i).item(0) for i in intensity]
            avg_intensity.append(avg_intensity_)

            avg_length_intensity_ = [i / l for l, i in zip(length[id_], avg_intensity_)]
            avg_length_intensity.append(avg_length_intensity_)

            sum_intensity_ = [np.sum(i).item(0) for i in intensity]
            sum_intensity.append(sum_intensity_)

            sum_length_intensity_ = [
                i / len_ for len_, i in zip(length[id_], sum_intensity_)
            ]
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
    names: Union[List, Tuple], analysis: Union[List, Tuple], px=None, save: str = None
) -> Union[None, np.ndarray]:
    """
    Saves or generates a detailed analysis file from the given `analysis` data. The function
    is capable of saving the processed analysis to a CSV file if a save path is provided
    or alternatively returns the processed analysis as a NumPy array. The analysis includes
    details such as filament lengths, curvatures, tortuosities, and intensity-related data. It
    is designed to handle multiple files and their corresponding filament data.

    :param names: List or tuple containing the names of the files being analyzed.
    :param analysis: Tuple or list containing multiple analysis metrics such as
        length, curvature, tortuosity, intensity, and their derivatives. Each metric
        corresponds to a specific aspect of the filament data.
    :param px: Optional parameter containing pixel size for each file, which is used
        to calculate some metrics. If not specified, a default value is assumed.
    :param save: Optional parameter specifying the path where the CSV file will be
        saved. If not provided, the function will not save the file and returns the
        analysis as a NumPy array.

    :return: Returns a NumPy array containing the processed analysis, or None if the
        analysis is saved to a CSV file.
    """

    length, curvature, tortuosity = analysis[0], analysis[1], analysis[2]
    avg_intensity, avg_length_intensity = analysis[3], analysis[4]
    sum_intensity, sum_length_intensity = analysis[5], analysis[6]

    rows = np.sum([len(i) for i in length]).item(0)
    analysis_file = np.zeros((rows, 10), dtype=object)

    iter_ = 0
    for i in range(len(length)):  # Iterate throw every file
        for j in range(len(length[i])):
            analysis_file[iter_, :] = [
                names[i],
                str(j),
                str(1.0) if px is None else px[i],
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

        segments = pd.DataFrame(analysis_file)
        segments.to_csv(
            join(save, file_name),
            header=[
                "File_Name",
                "No. of Filament",
                "Pixel_Size [nm]",
                "Length [nm]",
                "Curvature [0-inf]",
                "Tortuosity [1-inf]",
                "Avg. Intensity [U]",
                "Avg. Intensity / Length [U/nm]",
                "Sum. Intensity [U]",
                "Sum. Intensity / Length [U/nm]",
            ],
            index=False,
            sep=",",
        )

        return
    return analysis_file


def analyse_filaments_list(
    data: Union[List, Tuple],
    names_l: Union[List, Tuple],
    path: str,
    images: Union[List, Tuple] = None,
    px: Union[List, Tuple] = None,
    thickness=1,
):
    """
    Analyzes and processes a list of filament data along with optional corresponding images.
    The function validates the input to ensure consistency between the length of the data
    and images before analyzing the filaments. The processed analysis is then saved to the
    specified path.

    :param data: A collection of data representing filament structures. It must be a list
        or tuple.
    :type data: Union[List, Tuple]
    :param names_l: A collection of filenames or identifiers corresponding to each set of data.
        It must be a list or tuple.
    :type names_l: Union[List, Tuple]
    :param path: A string representing the file path where the analysis results will be saved.
    :type path: str
    :param images: Optional collection of images corresponding to the data, where each
        image represents the visual context of the filament. Default is None.
    :type images: Optional[Union[List, Tuple]]
    :param px: Optional collection indicating pixel calibration or scaling of the images,
        if provided. Default is None.
    :type px: Optional[Union[List, Tuple]]
    :param thickness: An integer specifying the thickness parameter used during filament
        analysis. Default is 1.
    :type thickness: int

    :return: None
    """

    if images is not None:
        assert_i = len(data) == len(images)
    else:
        assert_i = len(data) > 0

    if not assert_i:
        TardisError(
            id_="117",
            py="analysis.analysis:analise_filaments_list",
            desc="List of analysed files do not match or is 0",
            warning_b=False,
        )

    if images is None:
        images = [None for _ in range(len(data))]

    save_analysis(
        names_l, analyse_filaments(data, images, thickness, px=px), px=px, save=path
    )


def analyse_mt_classes(
    filaments: np.ndarray, poles: np.ndarray, vertices: list
) -> tuple:
    """
    Analyzes microtubule (MT) classes by assigning filaments to poles, then categorizing
    their ends as plus or minus ends. The process involves sorting poles relative to
    defined vertices, assigning filaments to corresponding poles, and subsequently
    determining the extremities of the filaments.

    :param filaments: A NumPy array of filaments where each filament is defined as a
        collection of points.
    :param poles: A NumPy array containing pole positions.
    :param vertices: A list of vertex points used to determine pole-to-surface assignments.

    :return: A tuple where the first element contains the mapping of filaments to poles
        and the second element is a tuple with two lists: one for the plus ends and
        another for the minus ends of the filaments.
    """
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

    return filament_poles, (plus_ends, minus_ends)
