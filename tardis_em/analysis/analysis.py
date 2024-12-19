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
    px_=None,
) -> tuple:
    """
    Analyzes the morphological and intensity-related properties of filaments in data.

    This function takes filament data and optionally associated images to calculate
    various geometric and intensity-based metrics. It computes the length, curvature,
    and tortuosity of the filaments. If images are provided, it can also calculate
    average and sum intensity values along filaments, as well as intensity metrics
    normalized by length. The function supports processing of multiple datasets and
    images at once.

    Args:
        data (np.ndarray, List, Tuple): A list, tuple, or NumPy array representing filament data.
        image (np.ndarray, List, Tuple): A list, tuple, or NumPy array representing associated image data, optional.
        thickness (int): Thickness of the filaments to consider during processing, optional.
        px_ (float, None): Scaling factors for the filament lengths.

    Returns:
        tuple: A tuple containing calculated metrics for the filaments, including
        (length, curvature, tortuosity, avg_intensity, avg_length_intensity,
        sum_intensity, sum_length_intensity). If no image is provided, intensity-related
        metrics will be returned as None.
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
        if px_ is None:
            length.append(length_list(d_))
        else:
            length.append([i * px_[id_] for i in length_list(d_)])  # length in A

        curvature_, tortuosity_ = curvature_list(d_, tortuosity_=True, mean_=True)
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
    names: Union[List, Tuple], analysis: Union[List, Tuple], px_=None, save: str = None
) -> Union[None, np.ndarray]:
    """
    Saves or returns a structured analysis of filaments. The function processes data
    such as length, curvature, tortuosity, and intensity-related metrics from given
    analysis input. Results are either returned as a NumPy array or saved as a CSV file
    at the specified location. The CSV file includes detailed headers for each analysis
    metric and is named with the current date and TARDIS version.

    Args:
        names (List, Tuple): List or tuple of file names corresponding to filament data.
        analysis (List, Tuple): A tuple or list containing the following filament analysis metrics:
            (lengths, curvatures, tortuosities, avg_intensities, avg_length_intensities,
            sum_intensities, sum_length_intensities), respectively.
        px_ (List, Tuple): Optional. List or tuple representing pixel sizes for given files. If not provided,
            the pixel size is assumed to be 1.0.
        save (str): Optional. The directory path to save the analysis as a CSV file. If not provided,
            the processed analysis is returned as an output.

    Returns:
        np.ndarray: Returns a structured NumPy array with filament analysis data if `save` is not
            provided. Otherwise, no return value (None) when the CSV file is successfully saved.
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
                str(1.0) if px_ is None else px_[i],
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
    names_: Union[List, Tuple],
    path: str,
    images: Union[List, Tuple] = None,
    px_: Union[List, Tuple] = None,
    thickness=1,
):
    """
    Analyzes a list of filaments data and performs operations such as matching data with
    provided images, checking their validity, and saving the processed analysis results.

    Args:
        data (List, Tuple): List or tuple containing the data to be analyzed.
        names_ (List, Tuple): List or tuple of names corresponding to the data items.
        path (str): Output path where the analysis results will be saved.
        images (List, Tuple): (Optional) List or tuple of images associated with the data.
        px_ (List, Tuple): (Optional) List or tuple of pixel values for each data item.
        thickness (int): Integer indicating the thickness to be used in analysis.
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

    save_analysis(
        names_, analyse_filaments(data, images, thickness, px_=px_), px_=px_, save=path
    )


def analyse_mt_classes(
    filaments: np.ndarray, poles: np.ndarray, vertices: list
) -> tuple:
    """
    Analyzes microtubule (MT) classes by assigning filaments to specific poles
    and identifying plus and minus ends of filaments. This function processes
    input data, which includes filaments, pole points, vertices, and triangulated
    surfaces, to categorize and evaluate relationships among these structures.

    The procedure includes:
    - Assigning poles to relevant vertices and surfaces.
    - Associating filaments with the poles.
    - Identifying and calculating indices for "plus" and "minus" ends of
      filaments for each pole.

    This analysis is essential for structural and spatial interpretations of
    microtubules within specific configurations.

    Args:
        filaments (np.ndarray): A numpy ndarray representing the filaments data. Each row
            should represent a filament with its associated properties such as
            position and metadata.
        poles (np.ndarray): A numpy ndarray representing the points or coordinates of
            poles.
        vertices (List): A list of vertices representing polygonal meshes
            associated with structural representations.

    Returns:
        A tuple where the first element is a list representing associations
        of filaments with poles. The second element is a tuple containing two
        lists: indices of plus ends and minus ends of the filaments for each
        pole.
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
