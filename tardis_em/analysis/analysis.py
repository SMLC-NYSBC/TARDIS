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
    intensity_list, calculate_spline_correlations,
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
    thickness=[1, 1],
    anal_list=None,
    px=None,
) -> list:
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
    :rtype: list
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
            intensity_mean, intensity_sum = intensity_list(d_, image[id_], thickness)

            # avg_intensity_ = [np.mean(i).item(0) for i in intensity]
            avg_intensity.append(intensity_mean)

            avg_length_intensity_ = [i / l for l, i in zip(length[id_], intensity_mean)]
            avg_length_intensity.append(avg_length_intensity_)

            sum_intensity.append(intensity_sum)

            sum_length_intensity_ = [
                i / len_ for len_, i in zip(length[id_], intensity_sum)
            ]
            sum_length_intensity.append(sum_length_intensity_)

    analysis = [length, curvature, tortuosity, avg_intensity, avg_length_intensity, sum_intensity, sum_length_intensity]

    return analysis


def save_analysis(
    names: Union[List, Tuple], analysis: Union[List, Tuple], anal_list: list, px=None, save: str = None,
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
    correlation = analysis[7]

    correlation_len = len(correlation[0][0])
    rows = np.sum([len(i) for i in length]).item(0)

    analysis_file = np.zeros((rows, 10+correlation_len), dtype=object)

    iter_ = 0
    for i in range(len(length)):  # Iterate throw every file
        for j in range(len(length[i])):
            analysis_file[iter_, :10] = [
                names[i],  # 0
                str(j),  # 1
                str(1.0) if px is None else px[i],  # 2
                str(length[i][j]),  # 3
                str(curvature[i][j]),  # 4
                str(tortuosity[i][j]),  # 5
                str(avg_intensity[i][j]),  # 6
                str(avg_length_intensity[i][j]),  # 7
                str(sum_intensity[i][j]),  # 8
                str(sum_length_intensity[i][j]),  # 9
            ]

            for k in range(correlation_len):
                analysis_file[iter_, 10 + k] = str(correlation[i][j][k])

            iter_ += 1

    if save is not None:
        date = datetime.now()
        tardis_version = version

        file_name = f"TARDIS_V{tardis_version}_analysis_{date.day}_{date.month}_{date.year}-{date.hour}_{date.minute}.csv"
        header = [
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
            ]
        header = header + [f"Correlation [Pearson] CH_{i}" for i in range(correlation_len)]

        if anal_list != []:
            df = {
                "length": 3,
                "curvature": 4,
                "tortuosity": 5,
                "avg_intensity": 6,
                "avg_length_intensity": 7,
                "sum_intensity": 8,
                "sum_length_intensity": 9
            }

            keep_ = [0, 1, 2] + [df[i] for i in anal_list if i in df]
            if "correlation" in anal_list:
                keep_ = keep_ + [10]

            remove_ = [item for item in list(range(analysis_file.shape[1])) if item not in keep_]

            if 10 in keep_:
                keep = keep_.extend([i for i in remove_ if i > 10])
                remove_ = [i for i in remove_ if i < 10]
            analysis_file = np.delete(analysis_file, remove_, axis=1)

            header = header + [f"Correlation [Pearson] CH_{i}" for i in range(2)]
            header = [h for id, h in enumerate(header) if id in keep_]

        segments = pd.DataFrame(analysis_file)
        segments.to_csv(
            join(save, file_name),
            header=header,
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
    anal_list=None,
    thicknesses=[1, 1],
    image_corr=None,
    frame_id=None,
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
    :param anal_list: List of analysis to perform, if None that all.
    :type anal_list: None or List[str]
    :param thicknesses: An integer specifying the thickness parameter used during filament
        analysis. Default is 1.
    :type thicknesses: List[int]

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

    analysis = analyse_filaments(data, images, thicknesses, px=px, anal_list=anal_list)
    correlations, correlations_px = calculate_spline_correlations(image_corr, data[0], frame_id=frame_id,
                                                                  thickness=[1, 1])
    analysis.append([correlations])
    if "correlation" in anal_list or anal_list == []:
        # Prepare a dictionary to store our columns
        csv_data = {}

        # Iterate through the main dictionary
        for main_key in correlations_px:
            # Add reference column
            csv_data[f"{main_key}_reference"] = correlations_px[main_key]["reference"]

            # Add MT columns
            for mt_key in correlations_px[main_key]["MT"]:
                csv_data[f"{main_key}_MT_{mt_key}"] = correlations_px[main_key]["MT"][mt_key]

        # Find the maximum length of all lists to ensure proper DataFrame creation
        max_length = max(len(lst) for col in csv_data.values() for lst in [col])

        # Pad shorter lists with None to make them equal length
        for col in csv_data:
            current_length = len(csv_data[col])
            if current_length < max_length:
                csv_data[col].extend([None] * (max_length - current_length))

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        date = datetime.now()
        tardis_version = version
        file_name = f"TARDIS_V{tardis_version}_correlation_{date.day}_{date.month}_{date.year}-{date.hour}_{date.minute}.csv"
        df.to_csv(join(path, file_name), index=False)

    save_analysis(
        names_l, analysis, px=px, save=path, anal_list=anal_list
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
