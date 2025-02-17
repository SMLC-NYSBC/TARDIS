#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from os import listdir
from os.path import isfile, join
from typing import Tuple, Union

import numpy as np

from tardis_em.cnn.data_processing.draw_mask import draw_instances
from tardis_em.cnn.data_processing.trim import trim_with_stride
from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import ImportDataFromAmira, load_image
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.normalization import RescaleNormalize, MeanStdNormalize


def build_train_dataset(
    dataset_dir: str,
    circle_size: int,
    resize_pixel_size: Union[float, None],
    trim_xy: int,
    trim_z: int,
    benchmark=False,
    correct_pixel_size=None,
):
    """
    Build a training dataset by processing image files and their corresponding masks.

    This function performs a series of data pre-processing steps, including file validation,
    loading images and masks, calculating scaling factors based on pixel size, handling different
    mask formats, and generating the required training dataset with any necessary transformations.
    Logs the progress and any encountered errors into a log file during processing.

    :param dataset_dir: Directory path containing the dataset files
    :type dataset_dir: str
    :param circle_size: Size of the circle used for mask drawing
    :type circle_size: int
    :param resize_pixel_size: Target pixel size for resizing the images, if specified
    :type resize_pixel_size: Union[float, NoneType]
    :param trim_xy: Number of pixels to trim from the x and y dimensions during processing
    :type trim_xy: int
    :param trim_z: Number of pixels to trim from the z dimension during processing
    :type trim_z: int
    :param benchmark: Flag indicating whether to keep certain processing details for benchmark datasets
    :type benchmark: bool
    :param correct_pixel_size: Optional, explicitly specifies the correct pixel size to be used
    :type correct_pixel_size: Union[float, NoneType]

    :return: None
    :rtype: NoneType
    """
    """Setup"""
    # Activate Tardis progress bar
    tardis_progress = TardisLogo()
    tardis_progress(title="Data pre-processing for CNN")

    # Builder for point cloud
    # b_pc = BuildPointCloud()

    clean_empty = not benchmark

    # All expected formats
    IMG_FORMATS = (".am", ".mrc", ".rec", ".map", ".tif")
    MASK_FORMATS = (
        ".CorrelationLines.am",
        "_mask.am",
        "_mask.mrc",
        "_mask.rec",
        "_mask.csv",
        "_mask.tif",
    )

    """Check what file are in the folder to build dataset"""
    img_list = [
        f
        for f in listdir(dataset_dir)
        if f.endswith(IMG_FORMATS) and not f.endswith(MASK_FORMATS)
    ]

    """For each image find matching mask, pre-process, trim and save"""
    img_counter = 0
    log_file = np.zeros((len(img_list) + 1, 10), dtype=object)
    log_file[0, :] = np.array(
        (
            "ID",
            "File_Name",
            "Pixel_Size",
            "Scale_Factor",
            "Mask_Type",
            "Image_Min",
            "Image_Max",
            "Total_Patches_No",
            "Saved_Patches_No",
            "Saved_Ratio [%]",
        )
    )

    id_ = 1
    for i in img_list:
        """Update progress bar"""
        tardis_progress(
            title="Data pre-processing for CNN training",
            text_1="Building Training dataset:",
            text_2=f"Files: {i}",
            text_3="px: NaN",
            text_4="Scale: NaN",
            text_6="Image dtype: NaN",
            text_7=print_progress_bar(id_, len(img_list)),
        )

        log_file[id_, 0] = str(id_)
        np.savetxt(
            join(dataset_dir, "log.csv"), log_file.astype(str), fmt="%s", delimiter=","
        )

        """Get image directory and check if img is a file"""
        img_dir = join(dataset_dir, i)
        if not isfile(img_dir):
            # Store fail in the log file
            log_file = error_log_build_data(
                dir_name=join(dataset_dir, "log.csv"), log_file=log_file, id_i=id_, i=i
            )
            continue

        """Get matching mask file and check if maks is a file"""
        mask_prefix = 0
        mask_name = ""
        mask_dir = ""
        while not isfile(join(dataset_dir, mask_name)):
            if mask_prefix >= len(MASK_FORMATS):
                break

            mask_name = (
                i[:-3] + MASK_FORMATS[mask_prefix]
                if i.endswith(".am")
                else i[:-4] + MASK_FORMATS[mask_prefix]
            )

            mask_dir = join(dataset_dir, mask_name)
            mask_prefix += 1

        if not isfile(mask_dir):
            # Store fail in the log file
            log_file = error_log_build_data(
                dir_name=join(dataset_dir, "log.csv"),
                log_file=log_file,
                id_i=id_,
                i=i + "|" + mask_dir,
            )
            continue

        """Load files"""
        image, mask, pixel_size = load_img_mask_data(img_dir, mask_dir)

        if correct_pixel_size is not None:
            pixel_size = correct_pixel_size

        log_file[id_, 1] = str(i + "||" + mask_name)
        np.savetxt(
            join(dataset_dir, "log.csv"), log_file.astype(str), fmt="%s", delimiter=","
        )

        if image is None:
            continue

        if pixel_size is None:
            log_file = error_log_build_data(
                dir_name=join(dataset_dir, "log.csv"),
                log_file=log_file,
                id_i=id_,
                i=i + "||" + mask_dir,
            )

        """Calculate scale factor"""
        if resize_pixel_size is None:
            scale_factor = 1.0
        else:
            scale_factor = pixel_size / resize_pixel_size
            pixel_size = resize_pixel_size

        scale_shape = [int(i * scale_factor) for i in image.shape]

        log_file[id_, 2] = str(pixel_size)
        log_file[id_, 3] = str(scale_factor)
        np.savetxt(
            join(dataset_dir, "log.csv"), log_file.astype(str), fmt="%s", delimiter=","
        )

        """Update progress bar"""
        tardis_progress(
            title="Data pre-processing for CNN training",
            text_1="Building Training dataset:",
            text_2=f"Files: {i} {mask_name}",
            text_3=f"px: {pixel_size}",
            text_4=f"Scale: {round(scale_factor, 2)}",
            text_6=f"Image dtype: {image.dtype} min: {image.min()} max: {image.max()}",
            text_7=print_progress_bar(id_, len(img_list)),
        )

        """Draw mask for coord or process mask if needed"""
        # Detect coordinate mask array
        if mask.ndim == 2 and mask.shape[1] in [3, 4]:
            # Scale mask to correct pixel size
            mask[:, 1:] = mask[:, 1:] * scale_factor

            # Draw mask from coordinates
            mask = draw_instances(
                mask_size=scale_shape,
                coordinate=mask,
                pixel_size=pixel_size,
                circle_size=circle_size,
            )
            log_file[id_, 4] = "coord"
            np.savetxt(
                join(dataset_dir, "log.csv"),
                log_file.astype(str),
                fmt="%s",
                delimiter=",",
            )
        else:  # Detect an image mask array
            # Convert to binary
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            if mask.min() != 0 and mask.max() != 1:
                TardisError(
                    id_="115",
                    py="tardis_em/cnn/data_processing.md/build_training_dataset",
                    desc=f"Mask min: {mask.min()}; max: {mask.max()} "
                    "but expected min: 0 and max: >1",
                )

            # mask borders
            if mask.ndim == 2:
                mask[:1, :] = 0
                mask[:, :1] = 0
                mask[-1:, :] = 0
                mask[:, -1:] = 0
            else:
                mask[:, 1, :] = 0
                mask[:, -1:, :] = 0
                mask[..., :1] = 0
                mask[..., -1:] = 0

            log_file[id_, 4] = "mask"
            np.savetxt(
                join(dataset_dir, "log.csv"),
                log_file.astype(str),
                fmt="%s",
                delimiter=",",
            )

        """Update progress bar"""
        tardis_progress(
            title="Data pre-processing for CNN training",
            text_1="Building Training dataset:",
            text_2=f"Files: {i} {mask_name}",
            text_3=f"px: {pixel_size}",
            text_4=f"Scale: {round(scale_factor, 2)}",
            text_6=f"Image dtype: {image.dtype} min: {image.min()} max: {image.max()}",
            text_7=print_progress_bar(id_, len(img_list)),
        )

        """Normalize image histogram"""
        log_file[id_, 5] = str(image.min())
        log_file[id_, 6] = str(image.max())
        np.savetxt(
            join(dataset_dir, "log.csv"), log_file.astype(str), fmt="%s", delimiter=","
        )

        tardis_progress(
            title="Data pre-processing for CNN training",
            text_1="Building Training dataset:",
            text_2=f"Files: {i} {mask_name}",
            text_3=f"px: {pixel_size}",
            text_4=f"Scale: {round(scale_factor, 2)}",
            text_6=f"Image dtype: {image.dtype} min: {image.min()} max: {image.max()}",
            text_7=print_progress_bar(id_, len(img_list)),
        )

        """Voxelize Image and Mask"""
        count = trim_with_stride(
            image=image,
            mask=mask,
            scale=scale_shape,
            trim_size_xy=trim_xy,
            trim_size_z=trim_z,
            clean_empty=clean_empty,
            keep_if=10 if image.ndim() == 3 else 100,
            output=join(dataset_dir, "train"),
            image_counter=img_counter,
            log=True,
            pixel_size=pixel_size,
        )
        img_counter += 1
        log_file[id_, 7] = str(count[0])
        log_file[id_, 8] = str(count[1])
        log_file[id_, 9] = str(np.round(count[1] / count[0], 2) * 100) + "%"

        np.savetxt(
            join(dataset_dir, "log.csv"), log_file.astype(str), fmt="%s", delimiter=","
        )
        id_ += 1


def load_img_mask_data(
    image: str, mask: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image and mask data from various supported file formats (e.g., MRC, REC, Amira, TIFF, CSV).
    The function also processes correlated data from mask coordinate files where necessary.
    It supports Amira, TIFF, and MRC/REC file formats for flexibility in scientific imaging workflows.
    Normalization and scaling are optionally applied to the image as part of the loading process.
    The function returns the processed image, mask or coordinate data, and pixel size information.

    :param image: Path to the image file. Supported formats are `.mrc`, `.rec`,
                  `.map`, `.am`, `.tif`, and `.tiff`.
    :type image: str
    :param mask: Path to the mask file. Supported formats are `_mask.mrc`,
                 `_mask.rec`, `_mask.am`, `_mask.csv`, `.CorrelationLines.am`,
                 and `_mask.tif`.
    :type mask: str

    :return: A tuple containing the normalized image, loaded mask or coordinate
             data (if applicable), and the pixel size as either a float (for pixel
             size) or None.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray or numpy.ndarray, float or None]
    """
    coord = None
    img_px, mask_px = 1, 1

    mean_std = MeanStdNormalize()
    normalize = RescaleNormalize(clip_range=(1, 99))

    """Load Amira or MRC/REC image"""
    if image.endswith((".mrc", ".rec", ".map")):  # Image is MRC/REC(image)
        # Load image
        image, img_px = load_image(image, normalize=False, px=True)
    elif image.endswith(".am"):  # Image is Amira (image)
        if mask.endswith(".CorrelationLines.am"):  # Found Amira (coord)
            importer = ImportDataFromAmira(src_am=mask, src_img=image)
            image, img_px = importer.get_image()

            coord = importer.get_segmented_points()  # [ID x X x Y x Z]

            mask_px = img_px
        else:  # Image is Amira (image)
            image, img_px = load_image(image)
    elif image.endswith((".tif", ".tiff")):
        image, img_px = load_image(image)

    image = normalize(mean_std(image)).astype(np.float32)

    """Load Amira or MRC/REC or csv mask"""
    # Find maska file and load
    if mask.endswith(("_mask.mrc", "_mask.rec")):  # Mask is MRC/REC (mask)
        mask, mask_px = load_image(mask)
    elif mask.endswith("_mask.am"):  # Mask is Amira (mask)
        mask, mask_px = load_image(mask)
    elif (
        mask.endswith(".CorrelationLines.am") and coord is None
    ):  # Mask is Amira (coord)
        importer = ImportDataFromAmira(src_am=mask)
        coord = importer.get_segmented_points()  # [ID x X x Y x Z]
        mask_px = img_px
        coord[:, 1:] = coord[:, 1:] // mask_px
    elif mask.endswith("_mask.csv"):  # Mask is csv (coord)
        coord = np.genfromtxt(mask, delimiter=",")  # [ID x X x Y x (Z)]
        if np.all(coord[0, :].astype(str) == "nan"):
            coord = coord[1:, :]

        mask_px = img_px
    elif mask.endswith("_mask.tif"):
        mask, _ = load_image(mask)
        mask_px = img_px

    if not img_px == mask_px:
        if not mask_px == 1.0 and not img_px > 1:
            img_px = None

    if coord is not None:
        return image, coord, img_px
    else:
        return image, mask, img_px


def error_log_build_data(
    dir_name: str, log_file: np.ndarray, id_i: int, i: str
) -> np.ndarray:
    """
    Stores error data into a log file and saves the updated log file.

    This function updates the specified log file with error data corresponding
    to a given ID and an identifier string. After updating the log file, it
    saves the file in the specified directory.

    :param dir_name: The directory location where the updated log file should be
        saved.
    :type dir_name: str
    :param log_file: A two-dimensional NumPy array representing the log file
        where error information will be stored.
    :type log_file: np.ndarray
    :param id_i: The integer identifier used to locate and store error data
        within the log file.
    :type id_i: int
    :param i: The identifier string associated with the specific error or
        data entry being logged.
    :type i: str
    :return: The updated log file as a two-dimensional NumPy array after
        storing the new data.
    :rtype: np.ndarray
    """

    # Store fail in the log file
    log_file[id_i + 1, 0] = str(id)
    log_file[id_i + 1, 1] = str(i)
    log_file[id_i + 1, 2] = "NA"
    log_file[id_i + 1, 3] = "NA"
    np.savetxt(dir_name, log_file, fmt="%s", delimiter=",")

    return log_file
