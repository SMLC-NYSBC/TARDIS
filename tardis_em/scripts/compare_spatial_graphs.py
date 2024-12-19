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
from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

import click
import numpy as np

from tardis_em.utils.errors import TardisError
from tardis_em.utils.export_data import NumpyToAmira
from tardis_em.utils.load_data import ImportDataFromAmira
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.analysis.spatial_graph_utils import SpatialGraphCompare
from tardis_em._version import version


@click.command()
@click.option(
    "-dir",
    "--path",
    default=getcwd(),
    type=str,
    help="Directory with spatial graphs from amira and tardis_em.",
    show_default=True,
)
@click.option(
    "-out",
    "--output",
    default=join(getcwd(), "output"),
    type=str,
    help="Directory for output compared files.",
    show_default=True,
)
@click.option(
    "-pf_a",
    "--amira_prefix",
    default=".CorrelationLines",
    type=str,
    help="Prefix name for amira spatial graph.",
    show_default=True,
)
@click.option(
    "-pf_t",
    "--tardis_prefix",
    default="_instance_segments",
    type=str,
    help="Prefix name for tardis_em spatial graph.",
    show_default=True,
)
@click.option(
    "-th_dist",
    "--distance_threshold",
    default=0,
    type=int,
    help="Distance threshold used to evaluate similarity between two "
    "splines based on its coordinates.",
    show_default=True,
)
@click.option(
    "-th_inter",
    "--interaction_threshold",
    default=100.0,
    type=float,
    help="Interaction threshold used to evaluate reject splines that are"
    "similar below that threshold.",
    show_default=True,
)
@click.option(
    "-db",
    "--debug",
    default=False,
    type=bool,
    help="If True, save output from each step for debugging.",
    show_default=True,
)
@click.version_option(version=version)
def main(
    distance_threshold: int,
    interaction_threshold: float,
    path: str,
    output: str,
    amira_prefix: Optional[str] = None,
    tardis_prefix: Optional[str] = None,
    debug=False,
):
    """
    MAIN MODULE FOR COMPARING SPLINES (AKA MTS) WITH AMIRA OUTPUT
    """

    """Initial Setup"""
    if debug:
        str_debug = "<Debugging Mode>"
    else:
        str_debug = ""

    tardis_progress = TardisLogo()
    tardis_progress(
        title=f"Spline matching module {str_debug}",
        text_1="Found NA spatial graphs to compare.",
        text_5="Amira: Nan",
        text_7="Tardis: NaN",
        text_9="Task: Searching for data...",
    )

    if not isdir(output):
        mkdir(output)
    else:
        tardis_progress(
            title=f"Spline matching module {str_debug}",
            text_1="Found NA spatial graphs to compare.",
            text_5="Amira: Nan",
            text_7="Tardis: NaN",
            text_9="Task: Output folder already exist...",
        )

        if (
            click.prompt(
                f"Remove all files in {output} ?", type=click.Choice(["y", "n"])
            )
            == "y"
        ):
            rmtree(output)
            mkdir(output)
        else:
            while not isdir(output):
                output = click.prompt("Type new directory", type=str)

    with open(join(output, "log.txt"), "w") as f:
        f.write(f"Spline matching module {datetime.now()}")

    if path is None:
        TardisError(
            id_="122",
            py="tardis_em/compare_spatial_graphs.py",
            desc="Indicated Amira and Tardis prefixes but " "not directory!",
        )

    dir_list = [d for d in listdir(path) if d != ".DS_Store"]
    amira_files = [d for d in dir_list if d.endswith(amira_prefix + ".am")]
    tardis_files = [d for d in dir_list if d.endswith(tardis_prefix + ".am")]

    with open(join(output, "log.txt"), "a+") as f:
        f.write("List of Amira files: \n" f"{amira_files} \n")
        f.write("List of Tardis files: \n" f"{tardis_files}\n" "\n")

    if len(amira_files) == 0 and len(tardis_files) == 0:
        TardisError(
            id_="121",
            py="tardis_em/compare_spatial_graphs.py",
            desc="No file found in given folders!",
        )

    if len(amira_files) != len(tardis_files):
        TardisError(
            id_="121",
            py="tardis_em/compare_spatial_graphs.py",
            desc=f"Amira folder have {len(amira_files)} files but "
            f"Tardis folder have {len(tardis_files)} files!",
        )

    tardis_progress(
        title=f"Spline matching module {str_debug}",
        text_1=f"Found {len(amira_files)} spatial graphs to compare.",
        text_5="Amira: Nan",
        text_7="Tardis: NaN",
        text_9="Task: Starting comparison...",
    )

    compare_spline = SpatialGraphCompare(
        distance_threshold=distance_threshold,
        interaction_threshold=interaction_threshold,
    )
    export_to_amira = NumpyToAmira()

    for id_, i in enumerate(amira_files):
        amira_file = join(path, i)
        tardis_file = join(path, i[: (-3 - len(amira_prefix))] + tardis_prefix + ".am")
        output_file = join(output, i[: (-3 - len(amira_prefix))] + "_match" + ".am")

        with open(join(output, "log.txt"), "a+") as f:
            f.write(
                "\n"
                f"{datetime.now()}"
                "Amira file: \n"
                f"{amira_files} \n"
                "Tardis files: \n"
                f"{tardis_file}\n"
                "Output: \n"
                f"{output_file}"
            )

        tardis_progress(
            title=f"Spline matching module {str_debug}",
            text_1=f"Found {len(amira_files)} spatial graphs to compare.",
            text_5=f"Amira: {amira_file}",
            text_7=f"Tardis: {tardis_file}",
            text_9="Task: Loading data...",
            text_10=print_progress_bar(id_, len(amira_files)),
        )

        amira_sg = ImportDataFromAmira(src_am=amira_file)
        amira_px = amira_sg.get_pixel_size()
        amira_sg = amira_sg.get_segmented_points()

        tardis_sg = ImportDataFromAmira(src_am=tardis_file)
        tardis_px = tardis_sg.get_pixel_size()
        tardis_sg = tardis_sg.get_segmented_points()

        with open(join(output, "log.txt"), "a+") as f:
            f.write(
                "\n"
                f"{datetime.now()}"
                f"Pixel size Amira: {amira_px} \n"
                f"Pixel size Tardis: {tardis_px} \n"
            )

        tardis_progress(
            title=f"Spline matching module {str_debug}",
            text_1=f"Found {len(amira_files)} spatial graphs to compare.",
            text_5=f"Amira: {amira_file}",
            text_7=f"Tardis: {tardis_file}",
            text_8=f"Pixel size: Amira {amira_px} A; Tardis {tardis_px} A.",
            text_9="Task: Comparing and saving Amira file...",
            text_10=print_progress_bar(id_, len(amira_files)),
        )

        if amira_px == tardis_px:
            am_compare, am_label = compare_spline(
                amira_sg=amira_sg, tardis_sg=tardis_sg
            )
            if debug:
                np.save(
                    join(output, i[: (-3 - len(amira_prefix))] + "_debug" + ".npy"),
                    am_compare,
                )

            export_to_amira.export_amira(
                file_dir=output_file,
                coords=am_compare,
                labels=[
                    "TardisFilterBasedOnAmira",
                    "TardisNoise",
                    "AmiraFilterBasedOnTardis",
                    "AmiraNoise",
                ],
            )
        else:
            with open(join(output, "log.txt"), "a+") as f:
                f.write(
                    "\n"
                    f"{datetime.now()}"
                    f"Skipped comparison - not matching pixel size!"
                )

    if len(amira_files) == 0:
        amira_files = ["test"]

    tardis_progress(
        title=f"Spline matching module {str_debug}",
        text_1=f"Found {len(amira_files)} spatial graphs to compare.",
        text_5="All amira files compared successfully!",
        text_10=print_progress_bar(len(amira_files), len(amira_files)),
    )


if __name__ == "__main__":
    main()
