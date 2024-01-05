# #####################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
# #####################################################################

import os
import subprocess
import subprocess as subp
import sys

from tardis_em.utils.errors import TardisError
from tardis_em.utils.logo import TardisLogo
from tardis_em._version import version


def env_exists(env_name: str) -> bool:
    """
    Simple check if environment exist

    Args:
        env_name (str): Environment name

    Returns:
        bool: If True, environment exist.
    """
    command = "conda info --envs"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    envs = output.decode().split("\n")

    return env_name in envs


def py(python: str):
    """
    Run pytest on specified python version inside conda environment

    Args:
        python (str): Python version.

    Returns:
        list: Output log from pytest.
    """
    os.chdir("../../")  # move to root dir

    if env_exists("PythonEnvTest"):
        # Remove test environment
        subp.run("conda remove -n PythonEnvTest --all -y", shell=True)
    else:
        # Create clean environment
        subp.run("conda create --name PythonEnvTest -y", shell=True)

    # Set up Python 3.X env and update
    subp.run(
        f"conda run -n PythonEnvTest conda install python={'3.' + python[1:]} -y",
        shell=True,
    )

    # Check and reinstall if needed requirements
    subp.run(
        "conda run -n PythonEnvTest pip install -r requirements-dev.txt", shell=True
    )
    subp.run("conda run -n PythonEnvTest pip install -r requirements.txt", shell=True)

    # Clean-up
    subp.run("conda run -n PythonEnvTest conda clean -a -y", shell=True)
    subp.run("conda run -n PythonEnvTest pip cache purge", shell=True)

    # Install tardis_em-pytorch
    subp.run("conda run -n PythonEnvTest pip install -e .", shell=True)

    # Test on Python 3.X.*
    return subp.run(
        "conda run -n PythonEnvTest pytest", shell=True, capture_output=True
    )


if __name__ == "__main__":
    tardis_progress = TardisLogo()
    tardis_progress(title=f"Development - TARDIS {version} - pytest")

    """ Run Pytest on python 3.7 - 3.11"""
    if sys.platform != "darwin":  # Python 3.7 on macOS is only available throw x64
        out = py(python="37")
        if not out.retuncode == 0:
            TardisError("20", f"{out}" "Pyton 3.7 pytest Failed")
            exit()

    out = py(python="38")
    if not out.returncode == 0:
        TardisError("20", f"{out}" "Pyton 3.8 pytest Failed")
        exit()

    out = py(python="39")
    if not out.returncode == 0:
        TardisError("20", f"{out}" "Pyton 3.9 pytest Failed")
        exit()

    out = py(python="310")
    if not out.returncode == 0:
        TardisError(f"{out}" "Pyton 3.10 pytest Failed")
        exit()

    # !!! Python 3.11 missing compatibility with numpy open3d and pytorch !!!
    #
    # out = py(python='311')
    # if not out.returncode == 0:
    #     TardisError(f'{out}'
    #                 'Pyton 3.11 pytest Failed')
    #     exit()

    """ Return output """
    tardis_progress(
        title=f"Development - TARDIS {version} - pytest",
        text_1="All test passed correctly on python 3.7, 3.8, 3.9, 3.10",
        text_2="Sphinx-build Completed.",
    )
