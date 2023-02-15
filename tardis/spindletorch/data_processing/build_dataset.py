#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from os import listdir
from os.path import isfile, join
from typing import Tuple

import numpy as np

from tardis.spindletorch.data_processing.semantic_mask import draw_semantic
from tardis.spindletorch.data_processing.trim import trim_with_stride
from tardis.utils.errors import TardisError
from tardis.utils.load_data import ImportDataFromAmira, load_image
from tardis.utils.logo import print_progress_bar, TardisLogo
from tardis.utils.normalization import MinMaxNormalize, RescaleNormalize


def build_train_dataset(dataset_dir: str,
                        circle_size: int,
                        resize_pixel_size: float,
                        trim_xy: int,
                        trim_z: int,
                        benchmark=False):
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
        resize_pixel_size (float): Pixel size for image resizing.
        trim_xy (int): Voxel size of output image in x and y dimension.
        trim_z (int): Voxel size of output image in z dimension.
        benchmark (bool): If True construct data for benchmark.
    """
    """Setup"""
    # Activate Tardis progress bar
    tardis_progress = TardisLogo()
    tardis_progress(title='Data pre-processing for CNN')

    # Normalize histogram
    normalize = RescaleNormalize(clip_range=(1, 99))
    minmax = MinMaxNormalize()

    clean_empty = not benchmark

    # All expected formats
    IMG_FORMATS = ('.am', '.mrc', '.rec')
    MASK_FORMATS = ('.CorrelationLines.am', '_mask.am', '_mask.mrc', '_mask.rec', '.csv')

    """Check what file are in the folder to build dataset"""
    img_list = [f for f in listdir(dataset_dir) if
                f.endswith(IMG_FORMATS) and not f.endswith(MASK_FORMATS)]

    """For each image find matching mask, pre-process, trim and save"""
    img_counter = 0
    log_file = np.zeros((len(img_list), 4), dtype='|S50')

    for id, i in enumerate(img_list):
        log_file[id, 0] = id
        np.savetxt(join(dataset_dir, 'log.txt'), log_file, fmt='%s', delimiter=',')

        """Get image directory and check if img is a file"""
        img_dir = join(dataset_dir, i)
        if not isfile(img_dir):
            # Store fail in the log file
            log_file = error_log_build_data(dir=join(dataset_dir, 'log.txt'),
                                            log_file=log_file,
                                            id=id,
                                            i=i)
            continue

        """Get matching mask file and check if maks is a file"""
        mask_prefix = 0
        mask_name = ''
        mask_dir = join(dataset_dir, mask_name)
        while not isfile(mask_dir):
            mask_name = i[:-3] + MASK_FORMATS[mask_prefix] \
                if i.endswith('.am') else i[:-4] + MASK_FORMATS[mask_prefix]

            mask_dir = join(dataset_dir, mask_name)
            mask_prefix += 1

            if mask_prefix > len(MASK_FORMATS):
                break

        if not isfile(mask_dir):
            # Store fail in the log file
            log_file = error_log_build_data(dir=join(dataset_dir, 'log.txt'),
                                            log_file=log_file,
                                            id=id,
                                            i=i + '|' + mask_dir)
            continue

        """Load files"""
        image, mask, pixel_size = load_img_mask_data(img_dir, mask_dir)
        log_file[id, 1] = i
        np.savetxt(join(dataset_dir, 'log.txt'), log_file, fmt='%s', delimiter=',')

        if pixel_size is None:
            log_file = error_log_build_data(dir=join(dataset_dir, 'log.txt'),
                                            log_file=log_file,
                                            id=id,
                                            i=i + '||' + mask_dir)

        """Calculate scale factor"""
        scale_factor = pixel_size / resize_pixel_size
        scale_shape = tuple(np.multiply(image.shape, scale_factor).astype(np.int16))

        log_file[id, 2] = str(pixel_size)
        log_file[id, 3] = str(scale_factor)
        np.savetxt(join(dataset_dir, 'log.txt'), log_file, fmt='%s', delimiter=',')

        """Draw mask for coord or process mask if needed"""
        if mask.ndim == 2 and mask.shape[1] in [3, 4]:  # Detect coordinate array
            # Scale mask to correct pixel size
            mask[:, 1:] = mask[:, 1:] * scale_factor

            # Draw mask from coordinates
            mask = draw_semantic(mask_size=scale_shape,
                                 coordinate=mask,
                                 pixel_size=resize_pixel_size,
                                 circle_size=circle_size)
        else:  # Detect mask array
            if mask.min() == 0:
                TardisError(id='115',
                            py='tardis/spindletorch/data_processing/build_training_dataset',
                            desc=f'Mask min: {mask.min()}; max: {mask.max()}'
                                 'but expected min: 0 and max: >1')

            # Convert to binary
            if mask.min() == 0 and mask.max() > 1:
                mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            # Flip mask if MRC/REC
            if mask_dir.endswith(('_mask.mrc', '_mask.rec')):
                mask = np.flip(mask, 1)

        """Update progress bar"""
        tardis_progress(title='Data pre-processing for CNN training',
                        text_1='Building Training dataset:',
                        text_2=f'Files: {i} {mask_name}',
                        text_3=f'px: {pixel_size}',
                        text_4=f'Scale: {round(scale_factor, 2)}',
                        text_6=f'Image dtype: {image.dtype} min: {image.min()} max: {image.max()}',
                        text_7=print_progress_bar(id, len(img_list)))

        """Normalize histogram"""
        # Rescale image intensity
        image = normalize(image)

        if not image.min() >= -1 or not image.max() <= 1:  # Image between in 0 and 255
            image = minmax(image)

        if image.dtype != np.float32:
            TardisError('114',
                        'tardis/spindletorch/data_processing/build_training_dataset',
                        f'Image data of type {image.dtype} not float32')

        tardis_progress(title='Data pre-processing for CNN training',
                        text_1='Building Training dataset:',
                        text_2=f'Files: {i} {mask_name}',
                        text_3=f'px: {pixel_size}',
                        text_4=f'Scale: {round(scale_factor, 2)}',
                        text_6=f'Image dtype: {image.dtype} min: {image.min()} max: {image.max()}',
                        text_7=print_progress_bar(id, len(img_list)))

        """Voxelize Image and Mask"""
        trim_with_stride(image=image,
                         mask=mask,
                         scale=scale_shape,
                         trim_size_xy=trim_xy,
                         trim_size_z=trim_z,
                         clean_empty=clean_empty,
                         output=join(dataset_dir, 'train'),
                         image_counter=img_counter)
        img_counter += 1


def load_img_mask_data(image: str,
                       mask: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Define file format and load adequately image and mask/coordinate file.

    Expected combination are:
        - Amira (image) + Amira (coord)
        - Amira (image) + csv (coord)
        - Amira (image) + Amira (mask) or MRC/REC (mask)

        - MRC/REC(image) + Amira (coord) ! Need check if coordinate is not transformed !
        - MRC/REC(image) + csv (coord)
        - MRC/REC(image) + Amira (mask) or MRC/REC (mask)

    Args:
        image (str): Directory address to the image file
        mask(str): Directory address to the mask/coordinate file

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Ndarray of image, mask, and pixel size
    """
    coord = None
    img_px, mask_px = 1, 1

    """Load Amira or MRC/REC image"""
    if image.endswith(('.mrc', '.rec')):  # Image is MRC/REC(image)
        # Load image
        image, img_px = load_image(image)
    elif image.endswith('.am'):  # Image is Amira (image)
        if mask.endswith('.CorrelationLines.am'):  # Found Amira (coord)
            importer = ImportDataFromAmira(src_am=mask, src_img=image)
            image, img_px = importer.get_image()

            coord = importer.get_segmented_points()  # [ID x X x Y x Z]
            coord[:, 1:] = coord[:, 1:]

            mask_px = img_px
        else:  # Image is Amira (image)
            image, img_px = load_image(image)

    """Load Amira or MRC/REC or csv mask"""
    # Find maska file and load
    if mask.endswith(('_mask.mrc', '_mask.rec')):  # Mask is MRC/REC (mask)
        mask, mask_px = load_image(mask)
    elif mask.endswith('_mask.am'):  # Mask is Amira (mask)
        mask, mask_px = load_image(mask)
    elif mask.endswith('.CorrelationLines.am') and coord is None:  # Mask is Amira (coord)
        importer = ImportDataFromAmira(src_am=mask)
        coord = importer.get_segmented_points()  # [ID x X x Y x Z]
        mask_px = importer.get_pixel_size()
        coord[:, 1:] = coord[:, 1:] // mask_px
    elif mask.endswith('_mask.csv'):  # Mask is csv (coord)
        coord = np.genfromtxt(mask, delimiter=',')  # [ID x X x Y x (Z)]
        mask_px = img_px

    if not img_px == mask_px:
        img_px = None

    if coord is not None:
        return image, coord, img_px
    else:
        return image, mask, img_px


def error_log_build_data(dir: str,
                         log_file: np.ndarray,
                         id: int,
                         i: str) -> list:
    """
    Update log file with error for data that could not be loaded

    Args:
        dir (str): Save directory.
        log_file (np.ndarray): Current log file list.
        id (int): Data id.
        i (Str): Data name.

    Returns:
        list: List of updated logfile
    """

    # Store fail in the log file
    log_file[id, 0] = str(id)
    log_file[id, 1] = i
    log_file[id, 2] = 'NA'
    log_file[id, 3] = 'NA'
    np.savetxt(dir, log_file, fmt='%s', delimiter=',')

    return log_file
