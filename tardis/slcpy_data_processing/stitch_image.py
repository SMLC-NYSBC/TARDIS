from tardis.slcpy_data_processing.utils.stitch import StitchImages
from os.path import join, isdir
from os import mkdir, rename
from shutil import rmtree


def ImageStitcher(image_dir: str,
                  output_dir: str,
                  mask: bool,
                  prefix: str,
                  binary: bool,
                  dtype: str,
                  tqdm: bool):
    """
    Main module for stitch individual images into montaged image

    Args:
        image_dir: Directory to the folder with image dataset.
        output_dir: Output directory for saving transformed files.
        output_name: File name to save stitch image.
        mask: Indicate if stitched images are mask or images.
        prefix: if not None, indicate additional file prefix.
        binary: If True transform date to binary format.
        dtype: Data format type.
        tqdm: Stitch with progessbar.
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

    """Stitch image"""
    stitched_image = StitchImages(tqdm=tqdm)
    stitched_image(image_dir=image_dir,
                   output=output_dir,
                   mask=mask,
                   prefix=prefix,
                   dtype=dtype)

    if binary:
        stitched_image[stitched_image > 0] = 1

    """CleanUp to avoid memory loss"""
    stitched_image = None
    del stitched_image
