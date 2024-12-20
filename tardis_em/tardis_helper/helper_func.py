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
import os
from os.path import join
import tardis_em.utils.load_data as Loader
import tardis_em.utils.export_data as Exporter
import tifffile.tifffile as tiff

array_formats = (".csv", ".npy")


def tardis_helper(func: str, dir_s=None, px: float = None):
    """
    This function assists in the format-specific processing and conversion of files
    within a specified directory. It determines the required operations based on
    the function name format, identifying the file type and output format. The
    function handles both coordinate data and image files, ensuring that requisite
    conversions are performed successfully.

    :param func: A string specifying the operation and file type format, structured
        as 'inputFileType_outputFileType'. This determines both the expected input
        file format and the desired output file format (e.g., 'csv_am', 'mrc_tiff').
    :type func: str
    :param dir_s: The directory path containing files to process. If None, the
        function will not operate on any files.
    :type dir_s: Optional[str]
    :param px: A float representing the pixel size, which is necessary for certain
        operations involving image files. Can be optionally passed or initially set
        to None.
    :type px: Optional[float]
    :return: None. The processed files are saved directly in the specified
        directory with the converted output format.
    :rtype: None
    """
    function = func.split("_")  # Expect e.g: csv_am
    in_files = [f for f in os.listdir(dir_s) if f.endswith(function[0])]

    for i in in_files:
        # Detect images or coord arrays
        if i.endswith(array_formats):
            # Convert coordinates
            if function[0] == "csv":
                coord = np.genfromtxt(
                    join(dir_s, i), delimiter=",", skip_header=1, dtype=np.float32
                )
            elif function[0] == "am":
                try:
                    am_import = Loader.ImportDataFromAmira(os.path.join(dir_s, i))
                    if am_import is not None:
                        am_import = am_import.get_segmented_points()
                except:
                    continue
            elif function[0] == "npy":
                coord = np.load(join(dir_s, i))

            if isinstance((coord[0, 0]), int) or isinstance((coord[0, 0]), float):
                if coord.shape[1] == 3:  # add 0 Z dimension
                    coord = np.vstack((coord, np.repeat(0, len(coord)).reshape(-1, 1)))

            if function[1] == "csv":
                np.savetxt(
                    join(dir_s, i[: -len(function[0])] + "csv"),
                    coord,
                    delimiter=",",
                )
            elif function[1] == "am":
                Exporter.NumpyToAmira().export_amira(
                    join(dir_s, i[: -len(function[0])] + "am"), coord
                )
        else:  # Convert image file
            image, px = Loader.load_image(join(dir_s, i))

            if function[0] == "mrc":
                header = Loader.mrc_read_header(dir_s)
            else:
                header = None

            if function[1] == "mrc":
                Exporter.to_mrc(
                    image, px, join(dir_s, i[: -len(function[0])] + "mrc"), header
                )
            elif function[1] == "am":
                Exporter.to_am(image, px, join(dir_s, i[: -len(function[0])] + "mrc"))
            elif function[1] == "tif" or function[1] == "tiff":
                tiff.imwrite(join(dir_s, i[: -len(function[0])] + "mrc"), image)
