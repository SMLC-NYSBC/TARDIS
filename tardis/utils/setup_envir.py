import glob
from os import mkdir, rename
from os.path import isdir, join
from shutil import rmtree


def build_new_dir(dir: str):
    """
    STANDRAD SET-UP FOR CREATING NEW DIRECOTORY FOR STORING ALL DATA

    Args:
        dir: Directory where folder will be build
    """
    """ Build temp files directory """
    output = join(dir, "output")
    image_list = glob.glob(dir + '/*.tif')
    assert len(image_list) > 0, \
        "At least one .tif image has to be in the directory!"

    if not isdir(output):
        mkdir(output)
    else:
        print("Output directory already exist! Files moved to new output_old "
              "directory.")
        rename(output, join(dir, "output_old"))
        mkdir(output)


def build_temp_dir(dir: str):
    """
    STANDRAD SET-UP FOR CREATING NEW TEMP DIR FOR CNN PREDICTION

    Args:
        dir: Directory where folder will be build
    """
    if isdir(join(dir, 'temp')):
        if isdir(join(dir, 'temp', 'Patches')) or isdir(join(dir, 'temp', 'Predictions')):
            clean_up(dir=dir)

            mkdir(join(dir, 'temp'))
            mkdir(join(dir, 'temp', 'Patches'))
            mkdir(join(dir, 'temp', 'Predictions'))
        else:
            mkdir(join(dir, 'temp', 'Patches'))
            mkdir(join(dir, 'temp', 'Predictions'))
    else:
        mkdir(join(dir, 'temp'))
        mkdir(join(dir, 'temp', 'Patches'))
        mkdir(join(dir, 'temp', 'Predictions'))

    if not isdir(join(dir, 'Predictions')):
        mkdir(join(dir, 'Predictions'))


def clean_up(dir: str):
    """
    CLEAN-UP TEMP FILES

    Args:
        dir: Main direcotry where temp dir is located
    """
    rmtree(join(dir, 'temp'))
