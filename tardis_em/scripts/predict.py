#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import warnings
from os import getcwd

import click

from tardis_em.utils.predictor import GeneralPredictor
from tardis_em.utils.errors import TardisError
from tardis_em._version import version
from tardis_em import format_choices

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option(
    "-dir",
    "--path",
    default=getcwd(),
    type=str,
    help="Directory with images for prediction with CNN model.",
    show_default=True,
)
@click.option(
    "-ms",
    "--mask",
    default=False,
    type=bool,
    help="Define if you input tomograms images or binary mask with pre segmented microtubules.",
    show_default=True,
)
@click.option(
    "-px",
    "--correct_px",
    default=None,
    type=float,
    help="Correct pixel size values.",
    show_default=True,
)
@click.option(
    "-norm_px",
    "--normalize_px",
    default=None,
    type=float,
    help="Normalize pixel size values do given resolution .",
    show_default=True,
)
@click.option(
    "-fi",
    "--filament",
    default=False,
    type=bool,
    help="If True, use general workflow for filament segmentation, else use general workflow"
    "for object instance segmentation.",
    show_default=True,
)
@click.option(
    "-it",
    "--image_type",
    default="3d",
    type=click.Choice(["2d", "3d"]),
    help="Select type of predicted image, 2D or 3D.",
    show_default=True,
)
@click.option(
    "-cch",
    "--cnn_checkpoint",
    default=None,
    type=str,
    help="Optional list of pre-trained CNN weights",
    show_default=True,
)
@click.option(
    "-dch",
    "--dist_checkpoint",
    default=None,
    type=str,
    help="Optional list of pre-trained DIST weights",
    show_default=True,
)
@click.option(
    "-cnn",
    "--convolution_nn",
    default="fnet_attn",
    type=str,
    help="Select CNN used for semantic segmentation.",
    show_default=True,
)
@click.option(
    "-out",
    "--output_format",
    default="None_amSG",
    type=click.Choice(format_choices),
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
    default="0.50",
    type=str,
    help="Threshold used for CNN prediction.",
    show_default=True,
)
@click.option(
    "-dt",
    "--dist_threshold",
    default=0.5,
    type=float,
    help="Threshold used for instance prediction.",
    show_default=True,
)
@click.option(
    "-pv",
    "--points_in_patch",
    default=900,
    type=int,
    help="Size of the cropped point cloud, given as a max. number of points "
    "per crop. This will break generated from the binary mask "
    "point cloud into smaller patches with overlap.",
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
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(
    path: str,
    mask: bool,
    filament: bool,
    image_type: str,
    cnn_checkpoint: str,
    dist_checkpoint: str,
    output_format: str,
    patch_size: int,
    rotate: bool,
    correct_px: float,
    normalize_px: float,
    convolution_nn: str,
    cnn_threshold: str,
    dist_threshold: float,
    points_in_patch: int,
    device: str,
    debug: bool,
    test_click=False,
):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    if output_format.split("_")[1] == "None":
        instances = False
    else:
        instances = True

    checkpoint = [cnn_checkpoint, dist_checkpoint]

    if filament:
        predict = "General_filament"
    else:
        predict = "General_object"

    predictor = GeneralPredictor(
        predict=predict,
        dir_s=path,
        binary_mask=mask,
        correct_px=correct_px,
        normalize_px=normalize_px,
        convolution_nn=convolution_nn,
        checkpoint=checkpoint,
        model_version=None,
        output_format=output_format,
        patch_size=patch_size,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        points_in_patch=points_in_patch,
        predict_with_rotation=rotate,
        instances=instances,
        device_s=str(device),
        debug=debug,
    )

    if image_type == "2d":
        predictor.expect_2d = True
    else:
        predictor.expect_2d = True

    if not test_click:
        predictor()


if __name__ == "__main__":
    main()
