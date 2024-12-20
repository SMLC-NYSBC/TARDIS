#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from os import mkdir

import click
import numpy as np
import os

from tardis_em.analysis.mt_classification.mt_classes import MicrotubuleClassifier
from tardis_em.utils.export_data import NumpyToAmira
from tardis_em._version import version


@click.command()
@click.option(
    "-s",
    "--path_surface",
    type=str,
    help="File directory to .surf Amira file.",
    show_default=True,
)
@click.option(
    "-f",
    "--path_filaments",
    type=str,
    help="File directory to .am spatial graph Amira file with microtubules.",
    show_default=True,
)
@click.option(
    "-p",
    "--path_poles",
    type=str,
    help="File directory to .am spatial graph Amira file with poles.",
    show_default=True,
)
@click.option(
    "-px",
    "--pixel_size",
    type=float,
    default=None,
    help="Pixel size value.",
    show_default=True,
)
@click.option(
    "-gap",
    "--gap_size",
    type=int,
    default=100,
    help="Gap size to fill out. Assuming very uneven surface to help the tool estimate when MT"
    "enter and exit the volume, the small gaps can be marge on the microtubule.",
    show_default=True,
)
@click.option(
    "-dist",
    "--kmts_to_surface_distance",
    type=int,
    default=1000,
    help="MT (+) end distance to the NN surface which allow it be be assigned as KMTs..",
    show_default=True,
)
@click.option(
    "-v",
    "--visualize",
    type=bool,
    default=False,
    help="MT (+) end distance to the NN surface which allow it be be assigned as KMTs..",
    show_default=True,
)
@click.version_option(version=version)
def main(
    path_surface: str,
    path_filaments: str,
    path_poles: str,
    pixel_size: float,
    gap_size: int,
    kmts_to_surface_distance: int,
    visualize: bool,
):
    dir_ = os.path.dirname(path_surface)
    out_ = os.path.join(dir_, "analysis")
    out_name = os.path.split(path_filaments)[-1][:-3] + "_classes.am"
    out_name = os.path.join(dir_, "analysis", out_name)

    if not os.path.isdir(out_):
        mkdir(out_)

    classifier = MicrotubuleClassifier(
        surfaces=path_surface,
        filaments=path_filaments,
        poles=path_poles,
        pixel_size=pixel_size,
        gaps_size=gap_size,
        kmt_dist_to_surf=kmts_to_surface_distance,
    )

    classifier.classified_MTs()

    classes = classifier.get_classified_fibers()

    am = NumpyToAmira(
        as_point_cloud=False,
        header=[
            f"Classification Analysis done with TARDIS_em v{version}",
            "Segmentation results:",
            f"KMTs: {len(classes[0][0]) + len(classes[0][1]) + len(classes[1][0]) + len(classes[1][1])}",
            f"Mid-MTs: {len(classes[2])}",
            f"Interdigitating-MTs: {len(classes[3])}",
            f"Bridging-MTs: {len(classes[4])}",
            f"SMTs: {len(classes[5])}",
        ],
    )

    am.export_amira(
        file_dir=out_name,
        coords=[
            classes[5],
            classes[0][0],
            classes[0][1],
            classes[1][0],
            classes[1][1],
            classes[2],
            classes[3],
            classes[4],
        ],
        labels=[
            "SMTs",
            "KMTs_pole1_Inside",
            "KMTs_pole1_Outside",
            "KMTs_pole2_Inside",
            "KMTs_pole2_Outside",
            "Mid-MTs",
            "Interdigitating-MTs",
            "Bridging-MTs",
        ],
    )

    if visualize:
        try:
            from tardis_em.utils.visualize_pc import (
                VisualizeFilaments,
                VisualizeSurface,
                VisualizeCompose,
            )

            vertices, triangles = classifier.get_vertices(simplify=128)
            vis_kmt = VisualizeFilaments(
                np.vstack((np.vstack(classes[0]), np.vstack(classes[1]))),
                False,
                filament_color="red",
                return_b=True,
            )
            vis_mid_mt = VisualizeFilaments(
                classes[2], False, filament_color="magenta", return_b=True
            )
            vis_brg_mt = VisualizeFilaments(
                classes[3], False, filament_color="white", return_b=True
            )
            vis_int_mt = VisualizeFilaments(
                classes[4], False, filament_color="orange", return_b=True
            )
            vis_smt = VisualizeFilaments(
                classes[5], False, filament_color="yellow", return_b=True
            )
            surf = VisualizeSurface(vertices, triangles, False, return_b=True)

            VisualizeCompose(
                False,
                vis_smt=vis_smt,
                vis_kmt=vis_kmt,
                vis_mid_mt=vis_mid_mt,
                vis_int_mt=vis_int_mt,
                vis_brg_mt=vis_brg_mt,
                surf=surf,
            )
        except ModuleNotFoundError:
            pass


if __name__ == "__main__":
    main()
