#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import time

import click
from tardis_em._version import version
from tardis_em.utils.logo import TardisLogo
from tardis_em.tardis_helper.helper_func import tardis_helper

ota = ""


@click.command()
@click.option(
    "-f",
    "--func",
    default=None,
    type=click.Choice(
        [
            "csv_am",
            "am_csv",
        ]
    ),
    help="Function name.",
    show_default=True,
)
@click.option(
    "-dir",
    "--dir_",
    default=None,
    type=str,
    help="Directory to files.",
    show_default=True,
)
@click.option(
    "-px",
    "--pixel_size",
    default=None,
    type=float,
    help="Optional pixel size value for image/coordinates conversion.",
    show_default=True,
)
@click.version_option(version=version)
def main(func=None, dir_=None, pixel_size=None):
    main_logo = TardisLogo()

    # Check if PyTorch was installed correctly with GPU support
    import torch

    res = torch.cuda.is_available()
    if not res:
        main_logo(
            title=f"| Transforms And Rapid Dimensionless Instance Segmentation | {ota}",
            text_0="WELCOME to TARDIS!",
            text_1="TARDIS detected no GPU support :(",
            text_3="Do not panic! Please uninstall pytorch,",
            text_4="and follow official instruction from https://pytorch.org",
            text_7="rkiewisz@nysbc.org | tbepler@nysbc.org",
            text_8="Join Slack community: https://tardis-em.slack.com",
        )
        time.sleep(10)

    if func is not None and dir_ is not None:
        tardis_helper(func, dir_, pixel_size)
    else:
        main_logo(
            title=f"| Transforms And Rapid Dimensionless Instance Segmentation | {ota}",
            text_0="WELCOME to TARDIS!",
            text_1="TARDIS is fully automatic segmentation software no need for model training!",
            text_3="Contact developers if segmentation of your organelle is not supported! ",
            text_4="rkiewisz@nysbc.org | tbepler@nysbc.org",
            text_5="Join Slack community: https://tardis-em.slack.com",
            text_7="FUNCTIONALITY:",
            text_8="To predict microtubule and filament instances:",
            text_9="  tardis_mt --help",
            text_10="To predict 3D membrane semantic and instances:",
            text_11=" tardis_mem --help |OR| tardis_mem2d --help",
        )


if __name__ == "__main__":
    main()
