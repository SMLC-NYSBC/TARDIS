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
from tardis_em.utils.logo import TardisLogo

from tardis_em.utils.aws import aws_check_pkg_with_temp
import sys
import time


def ota_update():
    timestamp = time.time()
    try:
        save = json.load(
            open(
                join(
                    expanduser("~"),
                    ".tardis_em",
                    "last_check.json",
                )
            )
        )["timestamp"]
    except:
        save = time.time()
        with open(
            join(join(expanduser("~"), ".tardis_em"), "last_check.json"), "w"
        ) as f:
            json.dump({"timestamp": timestamp}, f)

    if timestamp - save > 86400:
        # Check OTA-Update
        if not isdir(join(expanduser("~"), ".tardis_em")):
            mkdir(join(expanduser("~"), ".tardis_em"))

        ota_status = aws_check_pkg_with_temp()

        if not ota_status:
            main_logo = TardisLogo()
            main_logo(
                title="| Transforms And Rapid Dimensionless Instance Segmentation",
                text_0="TARDIS_pytorch has new update avaiable via OTA-Update!",
                text_1="Please in run this command to update tardis",
                text_3="tardis_ota",
                text_5="Contact developers if segmentation of your organelle is not supported! "
                "(rkiewisz@nysbc.org | tbepler@nysbc.org).",
                text_6="Join Slack community: https://tardis-em.slack.com",
            )
            time.sleep(10)
        with open(
            join(join(expanduser("~"), ".tardis_em"), "last_check.json"), "w"
        ) as f:
            json.dump({"timestamp": timestamp}, f)


def main():
    # Download OTA-Update
    try:
        py_pkg = requests.get(
            "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
            "tardis_em/tardis_em-x.x.x-py3-none-any.whl",
            timeout=(5, None),
        )
    except:
        return "OTA-Up-to-Data"

    # Save OTA-Update
    with open(
        join(expanduser("~"), ".tardis_em/tardis_em-x.x.x-py3-none-any.whl"),
        "wb",
    ) as f:
        f.write(py_pkg.content)

    with open(join(expanduser("~"), ".tardis_em/pkg_header.json"), "w") as f:
        json.dump(dict(py_pkg.headers), f)

    # Installed, uninstall old package version
    # subprocess.run(["pip", "uninstall", "-y", "tardis_em"])
    subprocess.run(
        [
            "pip",
            "install",
            join(
                expanduser("~"),
                ".tardis_em/" "tardis_em-x.x.x-py3-none-any.whl",
            ),
        ]
    )
    main_logo = TardisLogo()
    main_logo(
        title="| Transforms And Rapid Dimensionless Instance Segmentation",
        text_0="TARDIS_pytorch was updated via OTA-Update!",
        text_1="Please restart your previous operation.",
        text_3="Contact developers if segmentation of your organelle is not supported! "
        "(rkiewisz@nysbc.org | tbepler@nysbc.org).",
        text_4="Join Slack community: https://tardis-em.slack.com",
        text_6="FUNCTIONALITY:",
        text_7="To predict microtubule and filament instances:",
        text_8="    tardis_mt . | OR | tardis_mt --help          tardis_filament . | OR | tardis_filament --help",
        text_10="To predict 3D membrane semantic and instances:",
        text_11="    tardis_mem . | OR | tardis_mem --help       tardis_mem2d . | OR | tardis_mem2d --help",
    )
    sys.exit()


if __name__ == "__main__":
    main()
