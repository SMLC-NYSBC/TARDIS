import subprocess as subp
import sys
import pytest


def main():
    if sys.platform != 'darwin':
            ################
            # PYTHON 3.7.* #
            ################
            # Set up Python 3.7 env and update
            subp.run('conda run -n tardis37 conda update python -y', shell=True)

            # Check and reinstall if needed requirements
            try:
                subp.run('conda run -n tardis37 pip install -r requirements.txt', shell=True)
                subp.run('conda run -n tardis37 pip install -r requirements-dev.txt', shell=True)
                subp.run('conda run -n tardis37 conda clean -a -y', shell=True)
                subp.run('conda run -n tardis37 pip cache purge', shell=True)
            except:
                print('Skipped conda and pip update!')

            # Install tardis-pytorch
            subp.run('conda run -n tardis37 pip install -e . -y', shell=True)

            # Test on Python 3.7.*
            subp.run('conda run -n tardis37 pytest', shell=True)


    ################
    # PYTHON 3.8.* #
    ################
    # Set up Python 3.8 env and update
    subp.run('conda run -n tardis38 conda update python -y', shell=True)

    # Check and reinstall if needed requirements
    try:
        subp.run('conda run -n tardis38 pip install -r requirements.txt', shell=True)
        subp.run('conda run -n tardis38 pip install -r requirements-dev.txt', shell=True)
        subp.run('conda run -n tardis38 conda clean -a -y', shell=True)
        subp.run('conda run -n tardis38 pip cache purge', shell=True)
    except:
        print('Skipped conda and pip update!')

    # Install tardis-pytorch
    subp.run('conda run -n tardis38 pip install -e . -y', shell=True)

    # Test on Python 3.8.*
    subp.run('conda run -n tardis38 pytest', shell=True)


    ################
    # PYTHON 3.9.* #
    ################
    # Set up Python 3.9 env and update
    subp.run('conda run -n tardis39 conda update python -y', shell=True)

    # Check and reinstall if needed requirements
    try:
        subp.run('conda run -n tardis39 pip install -r requirements.txt', shell=True)
        subp.run('conda run -n tardis39 pip install -r requirements-dev.txt', shell=True)
        subp.run('conda run -n tardis39 conda clean -a -y', shell=True)
        subp.run('conda run -n tardis39 pip cache purge', shell=True)
    except:
        print('Skipped conda and pip update!')

    # Install tardis-pytorch
    subp.run('conda run -n tardis39 pip install -e . -y', shell=True)

    # Test on Python 3.9.*
    subp.run('conda run -n tardis39 pytest', shell=True)


    #################
    # PYTHON 3.10.* #
    #################
    # Set up Python 3.10 env and update
    subp.run('conda run -n tardis310 conda update python -y', shell=True)

    # Check and reinstall if needed requirements
    try:
        subp.run('conda run -n tardis310 pip install -r requirements.txt', shell=True)
        subp.run('conda run -n tardis310 pip install -r requirements-dev.txt', shell=True)
        subp.run('conda run -n tardis310 conda clean -a -y', shell=True)
        subp.run('conda run -n tardis310 pip cache purge', shell=True)
    except:
        print('Skipped conda and pip update!')

    # Install tardis-pytorch
    subp.run('conda run -n tardis310 pip install -e . -y', shell=True)

    # Test on Python 3.10.*
    subp.run('conda run -n tardis310 pytest', shell=True)


if __name__ == '__main__':
    main()
