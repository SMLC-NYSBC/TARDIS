#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import os
import subprocess
import pytest

# Define the path to the scripts directory
SCRIPTS_DIR = "tardis_em/scripts/"

# Collect all script files in the `scripts/` directory
script_files = [
    os.path.join(SCRIPTS_DIR, f)
    for f in os.listdir(SCRIPTS_DIR)
    if os.path.isfile(os.path.join(SCRIPTS_DIR, f))
    and f.endswith(".py")
    and not f.endswith("__init__.py")
]


@pytest.mark.parametrize("script", script_files)
def test_script_executable(script):
    """
    Test that the script runs and displays the help message.

    Args:
        script (str): Path to the script file.
    """
    # Ensure the script is executed without errors with the `--help` option
    result = subprocess.run(
        ["python", script, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # The process should exit successfully
    assert result.returncode == 0, f"Script {script} failed to execute."
    # The output should include the word 'Usage', common in click's help output
    assert (
        "Usage" in result.stdout.decode()
    ), f"Script {script} does not provide proper help output."

    if script.startswith(("predict", "train")):
        result = subprocess.run(
            ["python", script, "--test_click True"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # The process should exit successfully
        assert result.returncode == 0, f"Script {script} failed to execute."
