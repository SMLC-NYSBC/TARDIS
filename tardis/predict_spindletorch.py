import time
import warnings
from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import tifffile.tifffile as tif

from tardis.spindletorch.data_processing.semantic_mask import fill_gaps_in_semantic
from tardis.spindletorch.data_processing.stitch import StitchImages
from tardis.spindletorch.data_processing.trim import scale_image, trim_with_stride
from tardis.spindletorch.datasets.augment import MinMaxNormalize, RescaleNormalize
from tardis.spindletorch.datasets.dataloader import PredictionDataset
from tardis.utils.device import get_device
from tardis.utils.load_data import import_am, load_image
from tardis.utils.logo import Tardis_Logo, printProgressBar
from tardis.utils.predictor import Predictor
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.utils.utils import check_uint8
from tardis.version import version

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option('-dir', '--dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=96,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-cnn', '--cnn_network',
              default='unet_32',
              type=str,
              help='CNN network name.',
              show_default=True)
@click.option('-cch', '--cnn_checkpoint',
              default=None,
              type=str,
              help='If not None, str checkpoints for Big_Unet',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.1,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-d', '--device',
              default='0',
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 GPUs '
              'cpu: Usa CPU '
              '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-db', '--debug',
              default=False,
              type=bool,
              help='If True, save output from each step for debugging.',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         patch_size: int,
         cnn_network: str,
         cnn_threshold: float,
         device: str,
         debug: bool,
         visualizer: Optional[str] = None,
         cnn_checkpoint: Optional[str] = None):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    """Initial Setup"""
    if debug:
        str_debug = '<Debugging Mode>'
    else:
        str_debug = ''

    tardis_progress = Tardis_Logo()
    tardis_progress(title=f'Fully-automatic CNN prediction {str_debug}')

    # Searching for available images for prediction
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(dir, 'temp', 'Predictions')
    am_output = join(dir, 'Predictions')

    predict_list = [f for f in listdir(dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given directory!'

    # Tardis progress bard update
    tardis_progress(title=f'Fully-automatic CNN prediction {str_debug}',
                    text_1=f'Found {len(predict_list)} images to predict!',
                    text_5='Point Cloud: In processing...',
                    text_7='Current Task: Set-up environment...')

    # Hard fix for dealing with tif file lack of pixel sizes...
    tif_px = None
    if np.any([True for x in predict_list if x.endswith(('.tif', '.tiff'))]):
        tif_px = click.prompt('Detected .tif files, please provide pixel size:',
                              type=float)

    # Build handler's
    normalize = RescaleNormalize(range=(1, 99))  # Normalize histogram
    minmax = MinMaxNormalize()

    image_stitcher = StitchImages()

    # Build CNN from checkpoints
    checkpoints = (cnn_checkpoint)

    device = get_device(device)
    cnn_network = cnn_network.split('_')
    assert len(cnn_network) == 2, 'CNN type should be formatted as `unet_32`'

    # Build CNN network with loaded pre-trained weights
    predict_cnn = Predictor(checkpoint=checkpoints[0],
                            network=cnn_network[0],
                            subtype=cnn_network[1],
                            img_size=patch_size,
                            device=device)

    """Process each image with CNN and DIST"""
    tardis_progress = Tardis_Logo()
    for id, i in enumerate(sorted(predict_list)):
        """Pre-Processing"""
        if i.endswith('CorrelationLines.am'):
            continue

        out_format = 0
        if i.endswith(('.tif', '.mrc', '.rec')):
            out_format = 4
        elif i.endswith('.tiff'):
            out_format = 5
        elif i.endswith('.am'):
            out_format = 3

        # Tardis progress bard update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4='Pixel size: Nan A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Preprocessing for CNN...')

        # Build temp dir
        build_temp_dir(dir=dir)

        # Cut image for smaller image
        if i.endswith('.am'):
            image, px, _, transformation = import_am(am_file=join(dir, i))
        else:
            image, px = load_image(join(dir, i))
            transformation = [0, 0, 0]

        if tif_px is not None:
            px = tif_px

        # Check image structure and normalize histogram
        if image.min() > 5 or image.max() < 250:  # Rescale image intensity
            image = normalize(image)
        if not image.min() >= 0 or not image.max() <= 1:  # Normalized between 0 and 1
            image = minmax(image)
        assert image.dtype == np.float32

        # Calculate parameters for normalizing image pixel size
        scale_factor = px / 25
        org_shape = image.shape
        scale_shape = tuple(np.multiply(org_shape, scale_factor).astype(np.int16))

        # Tardis progress bard update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7=f'Current Task: Sub-dividing images for {patch_size} size')

        # Cut image for fix patch size and normalizing image pixel size
        trim_with_stride(image=image.astype(np.float32),
                         mask=None,
                         scale=scale_shape,
                         trim_size_xy=patch_size,
                         trim_size_z=patch_size,
                         output=join(dir, 'temp', 'Patches'),
                         image_counter=0,
                         clean_empty=False,
                         stride=10,
                         prefix='')
        image = None
        del image

        # Setup CNN dataloader
        patches_DL = PredictionDataset(img_dir=join(dir, 'temp', 'Patches'),
                                       out_channels=1)

        """CNN prediction"""
        iter_time = 1
        for j in range(len(patches_DL)):
            if j % iter_time == 0:
                # Tardis progress bard update
                tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                                text_1=f'Found {len(predict_list)} images to predict!',
                                text_3=f'Image: {i}',
                                text_4=f'Pixel size: {px} A; Image re-sample to 25 A',
                                text_5='Point Cloud: In processing...',
                                text_7='Current Task: CNN prediction...',
                                text_8=printProgressBar(j, len(patches_DL)))

            # Pick image['s]
            input, name = patches_DL.__getitem__(j)

            if j == 0:
                start = time.time()

                # Predict & Threshold
                out = np.where(predict_cnn._predict(input[None, :]) >= cnn_threshold, 1, 0)

                end = time.time()
                iter_time = 10 // (end - start)  # Scale progress bar refresh to 10s
                if iter_time <= 1:
                    iter_time = 1
            else:
                # Predict & Threshold
                out = np.where(predict_cnn._predict(input[None, :]) >= cnn_threshold, 1, 0)

            # Save threshold image
            assert out.min() == 0 and out.max() in [0, 1]
            tif.imwrite(join(output, f'{name}.tif'),
                        np.array(out, dtype=np.uint8))

        """Post-Processing"""
        scale_factor = org_shape

        # Tardis progress bard update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Stitching...')

        # Stitch predicted image patches
        image = check_uint8(image_stitcher(image_dir=output,
                                           output=None,
                                           mask=True,
                                           prefix='',
                                           dtype=np.int8)[:scale_shape[0],
                                                          :scale_shape[1],
                                                          :scale_shape[2]])
        assert image.shape == scale_shape, f'Image shape {image.shape} != scale shape {scale_shape}'

        # Restored original image pixel size
        image, _ = scale_image(image=image,
                               mask=None,
                               scale=scale_factor)

        # Fill gaps in binary mask after up/downsizing image to 2.5 nm pixel size
        image = fill_gaps_in_semantic(image)

        # Check if predicted image
        assert image.shape == org_shape, f'Error while converting to {px} A pixel size. ' \
            f'Org. shape {org_shape} is not the same as converted shape {image.shape}'

        # If prediction fail aka no prediction was produces continue with next image
        if image is None:
            continue

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; {np.max(segments[:, 0])} Segments',
                        text_7='Current Task: Segmentation finished!')

        """Clean-up temp dir"""
        clean_up(dir=dir)


if __name__ == '__main__':
    main()