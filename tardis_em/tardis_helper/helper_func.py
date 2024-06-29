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
import click
from os.path import join
import tardis_em.utils.load_data as Loader
import tardis_em.utils.export_data as Exporter
import tifffile.tifffile as tiff

array_formats = [".csv", ".npy"]


def tardis_helper(func: str, dir_=None):
    from tardis_em.utils.export_data import to_am

    function = func.split("_")
    in_files = [f for f in os.listdir(dir_) if f.endswith(function[0])]

    for i in in_files:
        # Detect image or coord array
        if i.endswith(array_formats):
            # Convert coordinates
            px = click.prompt(
                "What is the pixel size of the data:",
                type=float,
            )

            if function[0] == "csv":
                coord = np.genfromtxt(
                    join(dir_, i), delimiter=",", skip_header=1, dtype=np.float32
                )
            elif function[0] == "am":
                try:
                    am_import = Loader.ImportDataFromAmira(os.path.join(dir_, i))
                    if am_import is not None:
                        am_import = am_import.get_segmented_points()
                except:
                    continue

            if isinstance((coord[0, 0]), int) or isinstance((coord[0, 0]), float):
                if coord.shape[1] == 3:
                    coord = np.vstack((coord, np.repeat(0, len(coord)).reshape(-1, 1)))

            if function[1] == "csv":
                np.savetxt(
                    join(dir_, i[: -len(function[0])] + "csv"),
                    coord,
                    delimiter=",",
                )
            elif function[1] == "am":
                Exporter.NumpyToAmira().export_amira(
                    join(dir_, i[: -len(function[0])] + "am"), coord
                )
        else:
            image, px = Loader.load_image(join(dir_, i))

            if function[0] == "mrc":
                header = Loader.mrc_read_header(dir_)
            else:
                header = None

            if function[1] == "mrc":
                Exporter.to_mrc(
                    image, px, join(dir_, i[: -len(function[0])] + "mrc"), header
                )
            elif function[1] == "am":
                Exporter.to_am(image, px, join(dir_, i[: -len(function[0])] + "mrc"))
            elif function[1] == "tif" or function[1] == "tiff":
                tiff.imwrite(join(dir_, i[: -len(function[0])] + "mrc"), image)
