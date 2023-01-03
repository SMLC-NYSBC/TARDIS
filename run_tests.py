import subprocess as subp
import sys

from tardis.utils.errors import TardisError
from tardis.utils.logo import TardisLogo
from tardis.version import version


def py37():
    ################
    # PYTHON 3.7.* #
    ################
    # Set up Python 3.7 env and update
    subp.run("conda run -n tardis37 conda update python -y", shell=True)

    # Check and reinstall if needed requirements
    subp.run("conda run -n tardis37 pip install -r requirements.txt", shell=True)
    subp.run("conda run -n tardis37 pip install -r requirements-dev.txt", shell=True)
    subp.run("conda run -n tardis37 conda clean -a -y", shell=True)
    subp.run("conda run -n tardis37 pip cache purge", shell=True)

    # Install tardis-pytorch
    subp.run("conda run -n tardis37 pip install -e .", shell=True)

    # Test on Python 3.7.*
    return subp.run("conda run -n tardis37 pytest", shell=True, capture_output=True)


def py38():
    ################
    # PYTHON 3.8.* #
    ################
    # Set up Python 3.8 env and update
    subp.run("conda run -n tardis38 conda update python -y", shell=True)

    # Check and reinstall if needed requirements
    subp.run("conda run -n tardis38 pip install -r requirements.txt", shell=True)
    subp.run("conda run -n tardis38 pip install -r requirements-dev.txt", shell=True)
    subp.run("conda run -n tardis38 conda clean -a -y", shell=True)
    subp.run("conda run -n tardis38 pip cache purge", shell=True)

    # Install tardis-pytorch
    subp.run("conda run -n tardis38 pip install -e .", shell=True)

    # Test on Python 3.8.*
    return subp.run("conda run -n tardis38 pytest", shell=True, capture_output=True)


def py39():
    ################
    # PYTHON 3.9.* #
    ################
    # Set up Python 3.9 env and update
    subp.run("conda run -n tardis39 conda update python -y", shell=True)

    # Check and reinstall if needed requirements
    try:
        subp.run("conda run -n tardis39 pip install -r requirements.txt", shell=True)
        subp.run("conda run -n tardis39 pip install -r requirements-dev.txt", shell=True)
        subp.run("conda run -n tardis39 conda clean -a -y", shell=True)
        subp.run("conda run -n tardis39 pip cache purge", shell=True)
    except:
        print("Skipped conda and pip update!")

    # Install tardis-pytorch
    subp.run("conda run -n tardis39 pip install -e .", shell=True)

    # Test on Python 3.9.*
    return subp.run("conda run -n tardis39 pytest", shell=True, capture_output=True)


def py310():
    #################
    # PYTHON 3.10.* #
    #################
    # Set up Python 3.10 env and update
    subp.run("conda run -n tardis310 conda update python -y", shell=True)

    # Check and reinstall if needed requirements
    try:
        subp.run("conda run -n tardis310 pip install -r requirements.txt", shell=True)
        subp.run("conda run -n tardis310 pip install -r requirements-dev.txt", shell=True)
        subp.run("conda run -n tardis310 conda clean -a -y", shell=True)
        subp.run("conda run -n tardis310 pip cache purge", shell=True)
    except:
        print("Skipped conda and pip update!")

    # Install tardis-pytorch
    subp.run("conda run -n tardis310 pip install -e .", shell=True)

    # Test on Python 3.10.*
    return subp.run("conda run -n tardis310 pytest", shell=True, capture_output=True)


if __name__ == "__main__":
    tardis_progress = TardisLogo()
    tardis_progress(title=f'Development - TARDIS {version} - pytest')

    if sys.platform != "darwin":
        out = py37()
        print(out)

    out = py38()
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.7 pytest Failed')
        exit()

    out = py39()
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.8 pytest Failed')
        exit()

    out = py310()
    if not out.returncode == 0:
        TardisError(f'{out}'
                    'Pyton 3.10 pytest Failed')
        exit()

    tardis_progress(title=f'Development - TARDIS {version} - pytest',
                    text_1='All test passed correctly on pyton 3.7, 3.8, 3.9, 3.10')