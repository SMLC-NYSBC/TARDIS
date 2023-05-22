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
from tardis_pytorch._version import version
from tardis_pytorch.utils.logo import TardisLogo


@click.command()
@click.option(
    "-f",
    default=None,
    type=str,
    help="Directory with images for prediction with CNN model.",
    show_default=True,
)
@click.version_option(version=version)
def main(func=None):
    if func == "csv_am":
        from tardis_pytorch.tardis.csv_to_am import CsvToAmira

        CsvToAmira()
    else:
        main_logo = TardisLogo()
        main_logo(
            title="| Transforms And Rapid Dimensionless Instance Segmentation",
            text_0="WELCOME to TARDIS!",
            text_1="TARDIS is fully automatic segmentation software no need for model training!",
            text_3="Contact developers if segmentation of your organelle is not supported! "
            "(rkiewisz@nysbc.org | tbepler@nysbc.org).",
            text_4="Join Slack community: https://bit.ly/41hTCaP",
            text_6="FUNCTIONALITY:",
            text_7="To predict microtubule and filament instances:",
            text_8="    tardis_mt . | OR | tardis_mt --help          tardis_filament . | OR | tardis_filament --help",
            text_10="To predict 3D membrane semantic and instances:",
            text_11="    tardis_mem . | OR | tardis_mem --help       tardis_mem2d . | OR | tardis_mem2d --help",
        )


if __name__ == "__main__":
    main()
