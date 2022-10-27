from os import listdir
from os.path import isfile, join

import numpy as np
from tardis_dev.spindletorch.data_processing.semantic_mask import draw_semantic
from tardis_dev.spindletorch.data_processing.trim import trim_with_stride
from tardis_dev.spindletorch.datasets.augment import (MinMaxNormalize,
                                                      RescaleNormalize)
from tardis_dev.utils.load_data import ImportDataFromAmira, load_image
from tqdm import tqdm


def build_train_dataset(dataset_dir: str,
                        circle_size: int,
                        multi_layer: bool,
                        resize_pixel_size: float,
                        trim_xy: int,
                        trim_z: int):
    """
    Module for building train datasets from compatible files.

    This module building a train dataset from each file in the specified dir.
    Module working on the file as .tif/.mrc/.rec/.am and mask in image format
    of .csv and .am:
        - If mask is .csv, module expect image file in format of .tif/.mrc/.rec/.am
        - If mask is .am, module expect image file in format of .am

    For the given dir, module recognize file, and then for each file couple
    (e.g. image and mask) if needed (e.g. mask is not .tif) it's build mask from
    the point cloud. For this module expect that mask file (.csv, .am) contain
    coordinates values. Then for each point module draws 2D label.

    This image array's are then scale up or down to fit given pixel size, and
    finally array's are trimmed with overlay (stride) for specific size.

    Files are saved in ./dir/train/imgs and ./dir/train/masks.

    Mask are computed as np.uint8 arrays and images as np.float64 normalized image
    value between 0 and 1.

    Args:
        dataset_dir (str): Directory with train test folders.
        circle_size (int): Size of the segmented object in A.
        multi_layer (bool): If True mask is build in RGB channel instead of
            gray (0-1).
        resize_pixel_size (float): Pixel size for image resizing.
        trim_xy: Voxal size of output image in x and y dimension.
        trim_z: Voxal size of output image in z dimension.
    """
    normalize = RescaleNormalize(range=(1, 99))  # Normalize histogram
    minmax = MinMaxNormalize()

    IMG_FORMATS = ('.tif', '.am', '.mrc', '.rec')
    MASK_FORMATS = ('_mask.tif', "_mask.mrc", '_mask.rec', '_mask.am')

    """Check what file are in the folder to build dataset"""
    img_list = [f for f in listdir(dataset_dir) if f.endswith(IMG_FORMATS)]
    is_csv = [f for f in img_list if f.endswith('.csv')]
    is_mask_image = [f for f in img_list if f.endswith(MASK_FORMATS)]

    """Check for recognizable file types"""
    check_csv = False
    check_mask_image = False
    check_am = False
    if len(is_csv) > 0:  # Expect .csv as coordinate
        assert len(is_csv) * 2 == (len(img_list) / 2), \
            'Not all image file in directory has corresponding .csv file...'
        check_csv = True
    else:
        if len(is_mask_image) > 0:  # Expect image semantic mask as input
            assert len(is_mask_image) == (len(img_list) / 2), \
                'Not all image file in directory has corresponding _mask.* file...'
            check_mask_image = True
        else:
            # Expect .am as coordinate
            assert len([f for f in img_list
                        if f.endswith('.CorrelationLines.am')]) == (len(img_list) / 2), \
                'Not all image file in directory has corresponding .CorrelationLines.am file...'
            check_am = True

    assert np.any([check_csv, check_mask_image, check_am]), \
        'Could not find any compatible dataset. Refer to documentation for, ' \
        'how to structure your dataset properly!'

    """Build label mask file list"""
    if check_csv:
        mask_list = [f.endswith('.csv') for f in img_list]
    elif check_mask_image:
        mask_list = [f.endswith(MASK_FORMATS) for f in img_list]
    elif check_am:
        mask_list = [f.endswith(('.CorrelationLines.am')) for f in img_list]

    idx_mask = list(np.array(img_list)[mask_list])
    idx_img = list(np.array(img_list)[np.logical_not(mask_list)])
    assert len(idx_mask) == len(idx_img), \
        'Number of images and mask is not the same!'

    """Load data, build mask is not image and trim"""
    coord = None
    img_counter = 0

    batch_iter = tqdm(range(len(idx_img)))
    for i in batch_iter:
        """Load image data"""
        mask = None
        mask_name = idx_mask[i]

        if mask_name.endswith('.CorrelationLines.am'):
            if isfile(join(dataset_dir, f'{mask_name[:-20]}.am')):
                img_name = f'{mask_name[:-20]}.am'  # .am image file
            elif isfile(join(dataset_dir, f'{mask_name[:-20]}.rec')):
                img_name = f'{mask_name[:-20]}.rec'  # .rec image file
            elif isfile(join(dataset_dir, f'{mask_name[:-20]}.mrc')):
                img_name = f'{mask_name[:-20]}.mrc'  # .mrc image file
            elif isfile(join(dataset_dir, f'{mask_name[:-20]}.tif')):
                img_name = f'{mask_name[:-20]}.tif'  # .tif image file
            else:
                raise NameError(f'No image file found for {mask_name}')

        """Load image file"""
        if img_name.endswith(('.mrc', '.rec', '.tif')):
            image, pixel_size = load_image(join(dataset_dir, img_name))

            importer = ImportDataFromAmira(src_am=join(dataset_dir, mask_name))
            coord = importer.get_segmented_points()  # [ID x X x Y x Z]
            coord[:, 1:] = coord[:, 1:] // pixel_size
        elif img_name.endswith('.am'):
            importer = ImportDataFromAmira(src_am=join(dataset_dir,
                                                       mask_name),
                                           src_img=join(dataset_dir,
                                                        img_name))
            image, pixel_size = importer.get_image()
            coord = importer.get_segmented_points()  # [ID x X x Y x Z]

        """Calculate normalization factor"""
        if pixel_size == 0:
            scale_factor = 1
        else:
            scale_factor = pixel_size / resize_pixel_size

        batch_iter.set_description(f'Building Training dataset: \n'
                                   f'{img_name} {mask_name} '
                                   f'px: {pixel_size}\n'
                                   f'scale {round(scale_factor, 2)}')
        """Draw mask"""
        if coord is not None:
            assert coord.shape[1] == 4, \
                f'Coord file do not have shape [ID x X x Y x Z]. Given {coord.shape[1]}'
            if not is_mask_image:
                mask = draw_semantic(mask_size=image.shape,
                                     coordinate=coord,
                                     pixel_size=pixel_size,
                                     circle_size=circle_size,
                                     multi_layer=multi_layer)

        """Check image structure and normalize histogram"""
        image = normalize(image)  # Rescale image intensity
        image = minmax(image)

        """Voxalize Image and Mask"""
        trim_with_stride(image=image,
                         mask=mask,
                         trim_size_xy=trim_xy,
                         trim_size_z=trim_z,
                         scale=scale_factor,
                         output=join(dataset_dir, 'train'),
                         image_counter=img_counter,
                         clean_empty=True,
                         prefix='',
                         stride=25)
        img_counter += 1
