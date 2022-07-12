from os import listdir, mkdir, rename
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

import tifffile.tifffile as tif
from tardis.slcpy.utils.trim import trim_image, trim_with_stride


def ImageVoxalizer(images_dir: str,
                   output_dir: str,
                   image_with_mask=True,
                   mask_prefix: Optional[None] = '_mask',
                   trim_xy=64,
                   trim_z=1,
                   clean_empty=False,
                   stride: Optional[int] = None,
                   tqdm=True):
    """
    MAIN MODULE FOR COMPOSING SEMANTIC LABEL FROM GIVEN POINT CLOUD

    Args:
        images_dir: Directory to the folder with image dataset
        output_dir: Output directory for saving transformed files
        image_with_mask: Define if semantic mask is inducted and should have
            same ids as images
        mask_prefix: Prefix name at the end of mask file name
        trim_xy: Voxal size in X and Y
        trim_z: Voxal size in Z. 1 if image is 2D (auto-corrected for 2D)
        stride: Optional stride value for building voxals
        tqdm: If True, voxalize with progressbar
    """
    """Check all directory"""
    if isdir(output_dir):
        """Move dir if output exist"""
        try:
            print("Folder for the output data already exist... "
                  "Trying to copy output to output_old...")
            rename(output_dir, output_dir + '_old')

        except Exception:
            print("Folder output_old  already exist... "
                  "Output_old folder will be overwrite...")
            rmtree(join(output_dir + '_old'))
            rename(output_dir, output_dir + '_old')

    """Build output dir"""
    mkdir(output_dir)
    mkdir(join(output_dir, 'imgs'))
    if image_with_mask:
        mkdir(join(output_dir, 'mask'))

    idx = 0

    if tqdm:
        from tqdm import tqdm

        image_iter = tqdm(listdir(images_dir),
                          'Building Semantic patch images',
                          leave=True)
    else:
        image_iter = listdir(images_dir)

    for file in image_iter:
        image = tif.imread(join(images_dir, file))
        if image.ndim == 2:
            trim_z = 1

        if image_with_mask:
            if file[-(len(mask_prefix) + 4)] == f'{mask_prefix}.tif':
                image_prefix = mask_prefix
                image_output = join(output_dir, 'mask')
            else:
                image_prefix = ''
                image_output = join(output_dir, 'imgs')
        else:
            image_prefix = ''
            image_output = join(output_dir, 'imgs')

        if stride is None:
            trim_image(image=image,
                       trim_size_xy=trim_xy,
                       trim_size_z=trim_z,
                       output=image_output,
                       image_counter=idx,
                       clean_empty=clean_empty,
                       prefix=image_prefix)
        else:
            trim_with_stride(image=image,
                             trim_size_xy=trim_xy,
                             trim_size_z=trim_z,
                             output=image_output,
                             image_counter=idx,
                             clean_empty=clean_empty,
                             prefix=image_prefix,
                             stride=stride)

        idx += 1
