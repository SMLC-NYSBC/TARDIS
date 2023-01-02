from os import listdir
from os.path import isfile, join

import numpy as np

from tardis.spindletorch.data_processing.semantic_mask import draw_semantic
from tardis.spindletorch.data_processing.trim import trim_with_stride
from tardis.spindletorch.datasets.augment import MinMaxNormalize, RescaleNormalize
from tardis.utils.errors import TardisError
from tardis.utils.load_data import ImportDataFromAmira, load_image
from tardis.utils.logo import print_progress_bar, TardisLogo


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

    This image arrays are then scale up or down to fit given pixel size, and
    finally arrays are trimmed with overlay (stride) for specific size.

    Files are saved in ./dir/train/imgs and ./dir/train/masks.

    Mask are computed as np.uint8 arrays and images as np.float64 normalized image
    value between 0 and 1.

    Args:
        dataset_dir (str): Directory with train test folders.
        circle_size (int): Size of the segmented object in A.
        multi_layer (bool): If True mask is build in RGB channel instead of
            gray (0-1).
        resize_pixel_size (float): Pixel size for image resizing.
        trim_xy (int): Voxel size of output image in x and y dimension.
        trim_z (int): Voxel size of output image in z dimension.
    """
    """Setup"""
    tardis_progress = TardisLogo()
    tardis_progress(title='Data pre-processing for CNN training')

    normalize = RescaleNormalize(clip_range=(1, 99))  # Normalize histogram
    minmax = MinMaxNormalize()

    IMG_FORMATS = ('.am', '.mrc', '.rec')
    MASK_FORMATS = ('_mask.am', '_mask.mrc', '_mask.rec')

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
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        'Not all image file has corresponding .csv file...')
        check_csv = True
    else:
        if len(is_mask_image) > 0:  # Expect image semantic mask as input
            assert len(is_mask_image) == (len(img_list) / 2), \
                TardisError('TRAINING_DATASET_COMPATIBILITY',
                            'tardis/spindletorch/data_processing',
                            'Not all image file has corresponding _mask.* file...')
            check_mask_image = True
        else:  # Expect .am as coordinate
            assert len([f for f in img_list
                        if f.endswith('.CorrelationLines.am')]) == (len(img_list) / 2), \
                TardisError('TRAINING_DATASET_COMPATIBILITY',
                            'tardis/spindletorch/data_processing',
                            'Not all image file has corresponding .CorrelationLines.am')
            check_am = True

    assert np.any([check_csv, check_mask_image, check_am]), \
        TardisError('TRAINING_DATASET_COMPATIBILITY',
                    'tardis/spindletorch/data_processing/build_training_dataset.py',
                    'Could not find any compatible dataset. Refer to documentation for, '
                    'how to structure your dataset properly!')

    """Build label mask file list"""
    mask_list = []
    if check_csv:
        mask_list = [f.endswith('.csv') for f in img_list]
    elif check_mask_image:
        mask_list = [f.endswith(MASK_FORMATS) for f in img_list]
    elif check_am:
        mask_list = [f.endswith('.CorrelationLines.am') for f in img_list]

    idx_mask = list(np.array(img_list)[mask_list])
    idx_img = list(np.array(img_list)[np.logical_not(mask_list)])
    assert len(idx_mask) == len(idx_img), \
        TardisError('TRAINING_DATASET_COMPATIBILITY',
                    'tardis/spindletorch/data_processing',
                    'Number of images and mask is not the same!')

    """Load data, build mask is not image and trim"""
    coord = None
    img_counter = 0

    for id, i in enumerate(range(len(idx_img))):
        """Load image data and store image and mask name"""
        mask = None

        img_name = ''
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
                TardisError('TRAINING_DATASET_COMPATIBILITY',
                            'tardis/spindletorch/data_processing',
                            'Number of images and mask is not the same!')
        else:
            img_name = f'{mask_name[:-9]}.mrc'

        """Load image file"""
        pixel_size = 1
        image = None
        if img_name.endswith(IMG_FORMATS):
            if not check_am:
                image, pixel_size = load_image(join(dataset_dir, img_name))
            elif img_name.endswith('.am') and check_am:
                importer = ImportDataFromAmira(src_am=join(dataset_dir,
                                                           mask_name),
                                               src_img=join(dataset_dir,
                                                            img_name))
                image, pixel_size = importer.get_image()
                coord = importer.get_segmented_points()  # [ID x X x Y x Z]

        assert image is not None, \
            TardisError('LOADING_IMAGE_WHILE_BUILDING_DATASET',
                        'tardis/spindletorch',
                        f'Image {img_name} not in {IMG_FORMATS}!')

        """Load mask file"""
        # Try to load .CorrelationLines.am files
        mask_px = None
        if coord is None:
            importer = ImportDataFromAmira(src_am=join(dataset_dir, mask_name))
            coord = importer.get_segmented_points()  # [ID x X x Y x Z]
            mask_px = importer.get_pixel_size()
            coord[:, 1:] = coord[:, 1:] // mask_px
        elif check_mask_image:
            mask, mask_px = load_image(join(dataset_dir, mask_name))
        elif check_csv:
            coord = np.genfromtxt(join(dataset_dir, mask_name), delimiter=',')
            if str(coord[0, 0]) == 'nan':
                coord = coord[1:, :]

        """Calculate normalization factor for image and mask"""
        if mask_px is None:
            mask_px = pixel_size
        assert mask_px == pixel_size, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        f'Mask pixel size {mask_px} and image pixel size '
                        f'{pixel_size} and not the same!')

        scale_factor = pixel_size / resize_pixel_size
        scale_shape = tuple(np.multiply(image.shape, scale_factor).astype(np.int16))

        tardis_progress(title='Data pre-processing for CNN training',
                        text_1='Building Training dataset:',
                        text_2=f'Files: {img_name} {mask_name}',
                        text_3=f'px: {pixel_size}',
                        text_4=f'Scale: {round(scale_factor, 2)}',
                        text_6=f'Image dtype: {image.dtype} min: {image.min()} max: {image.max()}',
                        text_7=print_progress_bar(id,
                                                  len(idx_img)))

        """Draw mask"""
        if coord is not None:
            assert coord.shape[1] == 4, \
                TardisError('TRAINING_DATASET_COMPATIBILITY',
                            'tardis/spindletorch/data_processing',
                            f'Coord file do not have shape [ID x X x Y x Z]. '
                            f'Given {coord.shape[1]}')
            if not is_mask_image:
                # Scale mask to correct pixel size
                coord[:, 1:] = coord[:, 1:] * scale_factor

                # Draw mask from coordinates
                mask = draw_semantic(mask_size=scale_shape,
                                     coordinate=coord,
                                     pixel_size=resize_pixel_size,
                                     circle_size=circle_size,
                                     multi_layer=multi_layer)
        else:
            assert mask.min() >= 0, \
                TardisError('TRAINING_DATASET_COMPATIBILITY',
                            'tardis/spindletorch/data_processing',
                            f'Mask min: {mask.min()}; max: {mask.max()}')
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)  # Convert to binary
            if mask_name.endswith(('_mask.mrc', '_mask.rec')):
                mask = np.flip(mask, 1)

        """Normalize histogram"""
        # Rescale image intensity
        image = normalize(image)

        if not image.min() >= 0 or not image.max() <= 1:  # Image between in 0 and 255
            image = minmax(image)
        elif image.min() >= -1 and image.max() <= 1:  # int8 normalize
            image = minmax(image)

        assert image.dtype == np.float32, \
            TardisError('TRAINING_DATASET_COMPATIBILITY',
                        'tardis/spindletorch/data_processing',
                        f'Image data of type {image.dtype} not float32')

        tardis_progress(title='Data pre-processing for CNN training',
                        text_1='Building Training dataset:',
                        text_2=f'Files: {img_name} {mask_name}',
                        text_3=f'px: {pixel_size}',
                        text_4=f'Scale: {round(scale_factor, 2)}',
                        text_6=f'Image dtype: {image.dtype} min: {image.min()} max: {image.max()}',
                        text_7=print_progress_bar(id,
                                                  len(idx_img)))

        """Voxelize Image and Mask"""
        trim_with_stride(image=image,
                         mask=mask,
                         scale=scale_shape,
                         trim_size_xy=trim_xy,
                         trim_size_z=trim_z,
                         output=join(dataset_dir, 'train'),
                         image_counter=img_counter,
                         clean_empty=True,
                         stride=25,
                         prefix='')
        img_counter += 1
