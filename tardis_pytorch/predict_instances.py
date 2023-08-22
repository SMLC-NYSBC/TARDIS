#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import sys
import warnings
from os import getcwd

import click

from tardis_pytorch.utils.predictor import DataSetPredictor
from tardis_pytorch._version import version
from tardis_pytorch.utils.errors import TardisError

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option(
    "-dir",
    "--dir",
    default=getcwd(),
    type=str,
    help="Directory with binary images to predict instances.",
    show_default=True,
)
@click.option(
    "-ch",
    "--checkpoint",
    default=None,
    type=str,
    help="Optional pre-trained weights",
    show_default=True,
)
@click.option(
    "-st",
    "--structure_type",
    default="Filament",
    type=click.Choice(
        [
            "Microtubule",
            "Membrane",
            "Membrane2D",
        ]
    ),
    help="Type of output files for instance segmentation of objects, which can be "
         "output as amSG [Amira], mrcM [mrc mask], tifM [tif mask], "
         "csv coordinate file [ID, X, Y, Z] or None [no instance prediction].",
    show_default=True,
)
@click.option(
    "-out",
    "--output_format",
    default="amSG",
    type=click.Choice(
        [
            "amSG",
            "mrc",
            "tif",
            "csv",
        ]
    ),
    help="Type of output files for instance segmentation of objects, which can be "
    "output as amSG [Amira], mrcM [mrc mask], tifM [tif mask], "
    "csv coordinate file [ID, X, Y, Z] or None [no instance prediction].",
    show_default=True,
)
@click.option(
    "-dt",
    "--dist_threshold",
    default=0.75,
    type=float,
    help="Threshold used for instance prediction.",
    show_default=True,
)
@click.option(
    "-pv",
    "--points_in_patch",
    default=1000,
    type=int,
    help="Size of the cropped point cloud, given as a max. number of points "
    "per crop. This will break generated from the binary mask "
    "point cloud into smaller patches with overlap.",
    show_default=True,
)
@click.option(
    "-fl",
    "--filter_by_length",
    default=500,
    type=int,
    help="Filtering parameters for filament, defining maximum filament "
    "length in angstrom. All filaments shorter then this length "
    "will be deleted.",
    show_default=True,
)
@click.option(
    "-cs",
    "--connect_splines",
    default=2500,
    type=int,
    help="Filtering parameter for filament. Some filament may be "
    "predicted incorrectly as two separate filaments. To overcome this "
    "during filtering for each spline, we determine the vector in which "
    "filament end is facing and we connect all filament that faces "
    "the same direction and are within the given connection "
    "distance in angstrom.",
    show_default=True,
)
@click.option(
    "-cr",
    "--connect_cylinder",
    default=250,
    type=int,
    help="Filtering parameter for filament. To reduce false positive "
    "from connecting filaments, we reduce the searching are to cylinder "
    "radius given in angstrom. For each spline we determine vector "
    "in which filament end is facing and we search for a filament "
    "that faces the same direction and their end can be found "
    "within a cylinder.",
    show_default=True,
)
@click.option(
    "-dv",
    "--device",
    default=0,
    type=str,
    help="Define which device to use for training: "
    "gpu: Use ID 0 GPU"
    "cpu: Usa CPU"
    "mps: Apple silicon (experimental)"
    "0-9 - specified GPU device id to use",
    show_default=True,
)
@click.option(
    "-db",
    "--debug",
    default=False,
    type=bool,
    help="If True, save the output from each step for debugging.",
    show_default=True,
)
@click.version_option(version=version)
def main(
    dir: str,
    checkpoint: str,
    structure_type: str,
    output_format: str,
    dist_threshold: float,
    points_in_patch: int,
    filter_by_length: float,
    connect_splines: int,
    connect_cylinder: int,
    device: str,
    debug: bool,
):
    """
    MAIN MODULE FOR PREDICTION GENERAL FILAMENT WITH TARDIS-PYTORCH
    """
    if checkpoint is None:
        checkpoint = [None, None]
    else:
        checkpoint = [None, checkpoint]

    predictor = DataSetPredictor(
        predict=structure_type,
        dir_=dir,
        checkpoint=checkpoint,
        feature_size=0,
        output_format="None_"+output_format,
        patch_size=0,
        cnn_threshold=0,
        dist_threshold=dist_threshold,
        points_in_patch=points_in_patch,
        predict_with_rotation=False,
        filter_by_length=filter_by_length,
        connect_splines=connect_splines,
        connect_cylinder=connect_cylinder,
        instances=True,
        device_=str(device),
        debug=debug,
        amira_prefix=None,
        amira_compare_distance=None,
        amira_inter_probability=None,
    )

    predictor()


if __name__ == "__main__":
    main()
