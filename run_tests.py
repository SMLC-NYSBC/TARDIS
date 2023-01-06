"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

PyTest CI

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2022
"""
import shutil
import subprocess as subp
import sys

from tardis.utils.errors import TardisError
from tardis.utils.logo import TardisLogo
from tardis.version import version


def py(python: str):
    """
    Run pytest on specified python version inside conda environment

    Args:
        python (str): Python version.

    Returns:
        list: Output log from pytest.
    """

    # Set up Python 3.X env and update
    subp.run(f"conda run -n tardis{python} conda install python={'3.' + python[1:]} -y",
             shell=True)
    subp.run(f"conda run -n tardis{python} pip uninstall torch -y",
             shell=True)

    # Check and reinstall if needed requirements
    subp.run(f"conda run -n tardis{python} pip install -r requirements.txt",
             shell=True)
    subp.run(f"conda run -n tardis{python} pip install -r requirements-dev.txt",
             shell=True)
    subp.run(f"conda run -n tardis{python} conda clean -a -y",
             shell=True)
    subp.run(f"conda run -n tardis{python} pip cache purge",
             shell=True)

    # Install tardis-pytorch
    subp.run(f"conda run -n tardis{python} pip install -e .",
             shell=True)

    # Test on Python 3.X.*
    return subp.run(f"conda run -n tardis{python} pytest",
                    shell=True,
                    capture_output=True)


if __name__ == "__main__":
    tardis_progress = TardisLogo()
    tardis_progress(title=f'Development - TARDIS {version} - pytest')

    """ Run Pytest on python 3.7 - 3.11"""
    if sys.platform != "darwin":  # Python 3.7 on Macos is only available throw x64
        out = py(python='37')
        if not out.retuncode == 0:
            TardisError(f'{out}'
                        'Pyton 3.7 pytest Failed')
            exit()

    out = py(python='38')
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.8 pytest Failed')
        exit()

    out = py(python='39')
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.9 pytest Failed')
        exit()

    out = py(python='310')
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.10 pytest Failed')
        exit()

    # !!! Python 3.11 missing compatibility with numpy open3d and pytorch !!!
    #
    # out = py(python='311')
    # if not out.returncode == 0:
    #     TardisError(f'{out}'
    #                 'Pyton 3.11 pytest Failed')
    #     exit()

    """ Compile documentation """
    shutil.rmtree('docs/build')  # Remove old build
    subp.run('conda run -n tardis38 sphinx-build -b html docs/source/ docs/build/html')

    """ Return output """
    tardis_progress(title=f'Development - TARDIS {version} - pytest',
                    text_1='All test passed correctly on python 3.7, 3.8, 3.9, 3.10',
                    text_2='Sphinx-build Completed.')
