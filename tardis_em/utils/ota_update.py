#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import json
from os import mkdir
from os.path import expanduser, join, isdir, isfile
import subprocess
import requests
from tardis_em.utils.logo import TardisLogo

from tardis_em.utils.aws import aws_check_pkg_with_temp
import sys
import time


def ota_update(status=False):
    """
    Check for Over-The-Air (OTA) updates and handle notifications for available updates.

    This function performs a check to determine whether a new version of the application
    is available. The function periodically checks for updates based on a saved timestamp.
    If an update check has not occurred in the last 24 hours, it consults AWS for new updates.
    Based on the `status` parameter, it either returns a simple message or displays a
    detailed notification about the availability of a new version.

    :param status: A boolean indicating the type of response. If True, returns a simple
                   notification message. If False, displays a graphical notification about
                   the update.
    :type status: bool
    :return: If `status` is True and a new version is available, returns a string indicating
             that a new version is available. Otherwise, returns an empty string.
    :rtype: str
    """
    if not isdir(join(expanduser("~"), ".tardis_em")):
        mkdir(join(expanduser("~"), ".tardis_em"))

    timestamp = time.time()
    if isfile(
        join(
            expanduser("~"),
            ".tardis_em",
            "last_check.json",
        )
    ):
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
            ota_status = aws_check_pkg_with_temp()
        else:
            ota_status = True
    else:
        # Check OTA-Update
        ota_status = aws_check_pkg_with_temp()

    if status:
        if not ota_status:
            return "New version is available"
        else:
            return ""
    else:
        if not ota_status:
            main_logo = TardisLogo()
            main_logo(
                title="| Transforms And Rapid Dimensionless Instance Segmentation",
                text_0="TARDIS has new update available via OTA-Update!",
                text_1="Please in run this command to update tardis",
                text_3="tardis_ota",
                text_5="Contact developers if segmentation of your organelle is not supported!",
                text_6="rkiewisz@nysbc.org | tbepler@nysbc.org",
                text_8="Join Slack community: https://tardis-em.slack.com",
            )
            time.sleep(10)
        with open(
            join(join(expanduser("~"), ".tardis_em"), "last_check.json"), "w"
        ) as f:
            json.dump({"timestamp": timestamp}, f)


def main():
    """
    The `main` function performs an Over-The-Air (OTA) update for the Tardis-EM package.
    It checks or creates a directory for the update, retrieves the new package, saves
    the package locally, uninstalls previous versions of the package, installs the
    new one, and provides update-related instructions to the user via a visual display.

    :return: None
    """
    if not isdir(join(expanduser("~"), ".tardis_em")):
        mkdir(join(expanduser("~"), ".tardis_em"))

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
    # Make sure to remove legacy files
    try:
        subprocess.run(["pip", "uninstall", "-y", "tardis-pytorch"])
    except:
        pass

    try:
        subprocess.run(["pip", "uninstall", "-y", "tardis-em"])
    except:
        pass

    subprocess.run(["pip", "uninstall", "-y", "tardis_em"])

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
        text_0="TARDIS was updated via OTA-Update!",
        text_1="Please restart your previous operation.",
        text_3="(rkiewisz@nysbc.org | tbepler@nysbc.org).",
        text_4="Join Slack community: https://tardis-em.slack.com",
        text_6="FUNCTIONALITY:",
        text_7="To predict microtubule and filament instances:",
        text_8="  tardis_mt --help",
        text_10="To predict membrane semantic and instances:",
        text_11=" tardis_mem --help |OR| tardis_mem2d --help",
    )
    sys.exit()


if __name__ == "__main__":
    main()
