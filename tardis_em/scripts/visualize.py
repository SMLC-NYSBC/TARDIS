#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import click
import numpy as np
from tardis_em.utils.visualize_pc import VisualizeFilaments, VisualizePointCloud
from tardis_em.utils.load_data import ImportDataFromAmira
from tardis_em._version import version


@click.command()
@click.option(
    "-dir",
    "--dir_",
    type=str,
    help="File directory to visualize.",
    show_default=True,
)
@click.option(
    "-2d",
    "--_2d",
    type=bool,
    default=False,
    help="True if 2D data",
    show_default=True,
)
@click.option(
    "-t",
    "--type_",
    type=click.Choice(["f", "p"]),
    default="p",
    help="Visualize filaments or points",
    show_default=True,
)
@click.option(
    "-a",
    "--animate",
    type=bool,
    default=False,
    help="Animate if True.",
    show_default=True,
)
@click.option(
    "-wn",
    "--with_node",
    type=bool,
    default=False,
    help="If visualizing filaments, show color codded filaments with nodes.",
    show_default=True,
)
@click.version_option(version=version)
def main(dir_: str, _2d: bool, type_: str, animate: bool, with_node: bool):
    if dir_.endswith(".csv"):
        pc = np.genfromtxt(dir_, delimiter=",", skip_header=1, dtype=np.float32)
    elif dir_.endswith("am"):
        pc = ImportDataFromAmira(dir_).get_segmented_points()
    elif dir_.endswith(".npy"):
        pc = np.load(dir_)

    if type_ == "p":
        if pc.shape[1] == 4 or pc.shape[1] == 3 and _2d:
            VisualizePointCloud(pc, segmented=True, animate=animate)
        elif pc.shape[1] == 6:
            VisualizePointCloud(
                pc[:, :3], segmented=False, rgb=pc[:, 3:], animate=animate
            )
        else:
            VisualizePointCloud(pc, segmented=False, animate=animate)
    else:
        assert pc.shape[1] == 4, "Filament visualization require segmented point cloud"
        VisualizeFilaments(pc, animate=animate, with_node=with_node)


if __name__ == "__main__":
    main()
