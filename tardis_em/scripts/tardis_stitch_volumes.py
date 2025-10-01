#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################
import warnings
from os import getcwd
from os.path import join

import click

from tardis_em._version import version
from tardis_em_analysis.stitch_volume.utils import sort_tomogram_files
from tardis_em_analysis.stitch_volume.align_tomograms import AlignTomograms


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
    "-m",
    "--method",
    default="warp",
    type=click.Choice(["sift", "warp", "powell"]),
    help="Alignment method.",
    show_default=True,
)
@click.option(
    "-stitch",
    "--stitch_volumes",
    default=True,
    type=bool,
    help="If True, output stitched volume.",
    show_default=True,
)
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(path: str, method: str, stitch_volumes: bool, test_click: bool):
    path_images, path_coords = sort_tomogram_files(path)
    output = join(path, "aligned")

    stitcher = AlignTomograms(
        images_paths=path_images,
        coords_paths=path_coords,
        output_path=output,
        method=method,
    )

    if not test_click:
        stitcher.align_all_tomograms()

        if stitch_volumes:
            stitcher.stitch_align_volumes()


if __name__ == "__main__":
    main()
