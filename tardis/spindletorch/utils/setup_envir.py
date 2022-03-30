import glob
import shutil
from os import listdir, mkdir, rename
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

import numpy as np
from tardis.slcpy_data_processing.build_semantic_mask import slcpy_semantic
from tardis.slcpy_data_processing.voxalize_image import ImageVoxalizer
from tqdm import tqdm


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


def build_train_dataset(dir: str,
                        img_size: Optional[int] = None):
    mkdir(join(dir, 'train'))
    mkdir(join(dir, 'train', 'imgs'))
    mkdir(join(dir, 'train', 'mask'))

    if img_size is None:
        trim_xy, trim_z = 64, 64
    else:
        trim_xy, trim_z = img_size, img_size

    idx = 0
    batch_iter = tqdm(listdir(dir),
                      'Building Semantic Patch Images',
                      total=len(listdir(dir)),
                      leave=False)

    for file in batch_iter:
        if file.endswith('.tif'):
            image, label_mask = slcpy_semantic(join(dir, file),
                                               mask=True,
                                               pixel_size=None,
                                               circle_size=250,
                                               multi_layer=False,
                                               trim_mask=False)

            idx = trim_images(image=image,
                              label_mask=label_mask,
                              trim_size_xy=trim_xy,
                              trim_size_z=trim_z,
                              multi_layer=False,
                              output=join(dir, 'train'),
                              image_counter=idx)


def build_test_dataset(dir: str,
                       set_size: int):
    mkdir((join(dir, 'test')))
    mkdir(join(dir, 'test', 'imgs'))
    mkdir(join(dir, 'test', 'mask'))

    for i in range(49):
        file_list = listdir(join(dir, 'train', 'imgs'))
        id = np.random.randint(0, len(file_list) - 1)

        shutil.move(join(dir, 'train', 'imgs', file_list[id]),
                    join(dir, 'test', 'imgs', file_list[id]))
        shutil.move(join(dir, 'train', 'mask', file_list[id][:-4] + '_mask.tif'),
                    join(dir, 'test', 'mask', file_list[id][:-4] + '_mask.tif'))
