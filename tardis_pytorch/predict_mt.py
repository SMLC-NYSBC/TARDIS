#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import warnings
from os import getcwd

import click

from tardis_pytorch.utils.predictor import DataSetPredictor
from tardis_pytorch._version import version

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option(
    "-dir",
    "--dir",
    default=getcwd(),
    type=str,
    help="Directory with images for prediction with CNN model.",
    show_default=True,
)
@click.option(
    "-out",
    "--output_format",
    default="None_amSG",
    type=click.Choice(
        [
            "None_amSG",
            "am_amSG",
            "mrc_amSG",
            "tif_amSG",
            "None_mrcM",
            "am_mrcM",
            "mrc_mrcM",
            "tif_mrcM",
            "None_tifM",
            "am_tifM",
            "mrc_tifM",
            "tif_tifM",
            "None_mrcM",
            "am_csv",
            "mrc_csv",
            "tif_csv",
            "am_None",
            "mrc_None",
            "tif_None",
        ]
    ),
    help="Type of output files. The First optional output file is the binary mask "
    "which can be of type None [no output], am [Amira], mrc or tif. "
    "Second output is instance segmentation of objects, which can be "
    "output as amSG [Amira], mrcM [mrc mask], tifM [tif mask], "
    "csv coordinate file [ID, X, Y, Z] or None [no instance prediction].",
    show_default=True,
)
@click.option(
    "-ps",
    "--patch_size",
    default=128,
    type=int,
    help="Size of image patch used for prediction. This will break "
    "the tomogram volumes into 3D patches where each patch will be "
    "separately predicted and then stitched back together "
    "with 25% overlap.",
    show_default=True,
)
@click.option(
    "-rt",
    "--rotate",
    default=True,
    type=bool,
    help="If True, during CNN prediction image is rotate 4x by 90 degrees."
    "This will increase prediction time 4x. However may lead to more cleaner"
    "output.",
    show_default=True,
)
@click.option(
    "-ct",
    "--cnn_threshold",
    default=0.5,
    type=float,
    help="Threshold used for CNN prediction.",
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
    "-ap",
    "--amira_prefix",
    default=".CorrelationLines",
    type=str,
    help="If dir/amira foldr exist, TARDIS will serach for files with "
    "given prefix (e.g. file_name.CorrelationLines.am). If the correct "
    "file is found, TARDIS will use its instance segmentation with "
    "ZiB Amira prediction, and output additional file called "
    "file_name_AmiraCompare.am.",
    show_default=True,
)
@click.option(
    "-fl",
    "--filter_by_length",
    default=500,
    type=int,
    help="Filtering parameters for microtubules, defining maximum microtubule "
    "length in angstrom. All filaments shorter then this length "
    "will be deleted.",
    show_default=True,
)
@click.option(
    "-cs",
    "--connect_splines",
    default=2500,
    type=int,
    help="Filtering parameter for microtubules. Some microtubules may be "
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
    help="Filtering parameter for microtubules. To reduce false positive "
    "from connecting filaments, we reduce the searching are to cylinder "
    "radius given in angstrom. For each spline we determine vector "
    "in which filament end is facing and we search for a filament "
    "that faces the same direction and their end can be found "
    "within a cylinder.",
    show_default=True,
)
@click.option(
    "-acd",
    "--amira_compare_distance",
    default=175,
    type=int,
    help="If dir/amira/file_amira_prefix.am is recognized, TARDIS runs "
    "a comparison between its instance segmentation and ZiB Amira prediction. "
    "The comparison is done by evaluating the distance of two filaments from "
    "each other. This parameter defines the maximum distance used to "
    "evaluate the similarity between two splines based on their "
    "coordinates [A].",
    show_default=True,
)
@click.option(
    "-aip",
    "--amira_inter_probability",
    default=0.25,
    type=float,
    help="If dir/amira/file_amira_prefix.am is recognized, TARDIS runs "
    "a comparison between its instance segmentation and ZiB Amira prediction. "
    "This parameter defines the interaction threshold used to identify splines "
    "that are similar overlaps between TARDIS and ZiB Amira.",
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
    output_format: str,
    patch_size: int,
    rotate: bool,
    cnn_threshold: float,
    dist_threshold: float,
    points_in_patch: int,
    filter_by_length: float,
    connect_splines: int,
    connect_cylinder: int,
    amira_prefix: str,
    amira_compare_distance: int,
    amira_inter_probability: float,
    device: str,
    debug: bool,
):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    out = output_format.split("_")
    if out[1] == "None":
        instances = False
    else:
        instances = True

    predictor = DataSetPredictor(
        predict="Microtubule",
        dir_=dir,
        output_format=output_format,
        patch_size=patch_size,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        points_in_patch=points_in_patch,
        predict_with_rotation=rotate,
        amira_prefix=amira_prefix,
        filter_by_length=filter_by_length,
        connect_splines=connect_splines,
        connect_cylinder=connect_cylinder,
        amira_compare_distance=amira_compare_distance,
        amira_inter_probability=amira_inter_probability,
        instances=instances,
        device_=str(device),
        debug=debug,
    )

    predictor()


if __name__ == "__main__":
    main()
