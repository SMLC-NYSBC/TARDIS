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
from tardis_em_analysis.stitch_volume.align_tomograms import stitch_tomogram_stack


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
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(path: str, test_click: bool):
    output = join(path, "aligned")

    stitch_tomogram_stack(
    input_dir=path,
    output_dir=output,
    method='mt',
    )

    if not test_click:
        pass


if __name__ == "__main__":
    main()
