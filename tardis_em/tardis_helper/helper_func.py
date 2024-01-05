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


def tardis_helper(func: str, dir_=None):
    from tardis_em.utils.export_data import to_am

    if func == "csv_am":
        px = click.prompt(
            "Detected .tif files, please provide pixel size "
            "(It will be used for all .tif images):",
            type=float,
        )

        csv_files = [f for f in os.listdir(dir_) if f.endswith(".csv")]
        for i in csv_files:
            try:
                coord = np.genfromtxt(os.path.join(dir_, i), delimiter=",")
                if isinstance((coord[0, 0]), int) or isinstance((coord[0, 0]), float):
                    if coord.shape[1] == 3:
                        coord = np.vstack(
                            (coord, np.repeat(0, len(coord)).reshape(-1, 1))
                        )
                    to_am(coord, px, os.path.join(dir_, i[:-4] + ".csv"))
            except:
                pass
    elif func == "am_csv":
        from tardis_em.utils.load_data import ImportDataFromAmira

        # find a all .am files
        am_files = [f for f in os.listdir(dir_) if f.endswith(".am")]

        for i in am_files:
            try:
                am_import = ImportDataFromAmira(os.path.join(dir_, i))
                if am_import is not None:
                    am_import = am_import.get_segmented_points()
                    np.savetxt(
                        os.path.join(dir_, i[:-3] + ".csv"),
                        am_import,
                        delimiter=",",
                    )
            except:
                pass
