import glob
from os import mkdir, rename
from os.path import isdir, join
from shutil import rmtree


def build_new_dir(img_dir: str):
    """ Build temp files directory """
    output = join(img_dir, "output")
    image_list = glob.glob(img_dir + '/*.tif')
    assert len(image_list) > 0, \
        "At least one .tif image has to be in the directory!"

    if not isdir(output):
        mkdir(output)
    else:
        print("Output directory already exist! Files moved to new output_old "
              "directory.")
        rename(output, join(img_dir, "output_old"))
        mkdir(output)


def build_temp_dir(img_dir: str):
    if isdir(join(img_dir, "Patches")) or isdir(join(img_dir, "Predictions")):
        clean_up(img_dir=img_dir)
        mkdir(join(img_dir, "Patches"))
        mkdir((join(img_dir, "Predictions")))
    else:
        mkdir(join(img_dir, "Patches"))
        mkdir((join(img_dir, "Predictions")))


def clean_up(img_dir: str):
    """ Clean up temp files """
    rmtree(join(img_dir, "Patches"))
    rmtree(join(img_dir, "Predictions"))
