#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import click
from tardis_em._version import version
from tardis_em.utils.logo import TardisLogo
from tardis_em.tardis_helper.helper_func import tardis_helper
from tardis_em.utils.ota_update import ota_update

ota = ota_update()


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
@click.version_option(version=version)
def main(func=None, dir_=None):
    if func is not None and dir_ is not None:
        tardis_helper(func, dir_)
    else:
        main_logo = TardisLogo()
        main_logo(
            title=f"| Transforms And Rapid Dimensionless Instance Segmentation | {ota}",
            text_0="WELCOME to TARDIS!",
            text_1="TARDIS is fully automatic segmentation software no need for model training!",
            text_3="Contact developers if segmentation of your organelle is not supported! ",
            text_4="rkiewisz@nysbc.org | tbepler@nysbc.org",
            text_5="Join Slack community: https://bit.ly/41hTCaP",
            text_7="FUNCTIONALITY:",
            text_8="To predict microtubule and filament instances:",
            text_9="    tardis_mt . | OR | tardis_mt --help          tardis_filament . | OR | tardis_filament --help",
            text_10="To predict 3D membrane semantic and instances:",
            text_11="    tardis_mem . | OR | tardis_mem --help       tardis_mem2d . | OR | tardis_mem2d --help",
        )


if __name__ == "__main__":
    main()
