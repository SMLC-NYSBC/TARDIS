#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import json
from os import mkdir
from os.path import expanduser, join, isdir
import subprocess
import requests
from tardis_pytorch.utils.logo import TardisLogo

from tardis_pytorch.utils.aws import aws_check_pkg_with_temp
import sys


def ota_update():
    # Check OTA-Update
    if not isdir(join(expanduser("~"), ".tardis_pytorch")):
        mkdir(join(expanduser("~"), ".tardis_pytorch"))

    ota_status = aws_check_pkg_with_temp()

    if not ota_status:
        # Download OTA-Update
        try:
            py_pkg = requests.get(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
                "tardis_pytorch/tardis_pytorch-x.x.x-py3-none-any.whl",
                timeout=(5, None),
            )
        except:
            return "OTA-Up-to-Data"

        # Save OTA-Update
        with open(
            join(
                expanduser("~"), ".tardis_pytorch/tardis_pytorch-x.x.x-py3-none-any.whl"
            ),
            "wb",
        ) as f:
            f.write(py_pkg.content)

        with open(join(expanduser("~"), ".tardis_pytorch/pkg_header.json"), "w") as f:
            json.dump(dict(py_pkg.headers), f)

        # Installed, uninstall old package version
        # subprocess.run(["pip", "uninstall", "-y", "tardis_pytorch"])
        subprocess.run(
            [
                "pip",
                "install",
                join(
                    expanduser("~"),
                    ".tardis_pytorch/" "tardis_pytorch-x.x.x-py3-none-any.whl",
                ),
                "--force-reinstall"
            ]
        )
        main_logo = TardisLogo()
        main_logo(
            title="| Transforms And Rapid Dimensionless Instance Segmentation",
            text_0="TARDIS_pytorch was updated via OTA-Update!",
            text_1="Please restart your previous operation.",
            text_3="Contact developers if segmentation of your organelle is not supported! "
            "(rkiewisz@nysbc.org | tbepler@nysbc.org).",
            text_4="Join Slack community: https://bit.ly/41hTCaP",
            text_6="FUNCTIONALITY:",
            text_7="To predict microtubule and filament instances:",
            text_8="    tardis_mt . | OR | tardis_mt --help          tardis_filament . | OR | tardis_filament --help",
            text_10="To predict 3D membrane semantic and instances:",
            text_11="    tardis_mem . | OR | tardis_mem --help       tardis_mem2d . | OR | tardis_mem2d --help",
        )
        sys.exit()
    else:
        return "OTA-Up-to-Data"
