from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import tifffile.tifffile as tif

from tardis.slcpy.utils.load_data import import_am, import_mrc, import_tiff
from tardis.slcpy.utils.stitch import StitchImages
from tardis.slcpy.utils.trim import trim_image
from tardis.spindletorch.predict import predict
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.version import version


@click.command()
@click.option('-dir', '--prediction_dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=64,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-cnn', '--cnn_type',
              default='unet',
              type=click.Choice(['unet', 'resunet', 'unet3plus', 'multi'],
                                case_sensitive=True),
              help='Type of NN used for training.',
              show_default=True)
@click.option('-co', '--cnn_out_channel',
              default=1,
              type=int,
              help='Number of output channels for the NN.',
              show_default=True)
@click.option('-cl', '--cnn_layers',
              default=5,
              type=int,
              help='Number of convolution layer for NN.',
              show_default=True)
@click.option('-cm', '--cnn_multiplayer',
              default=64,
              type=int,
              help='Convolution multiplayer for CNN layers.',
              show_default=True)
@click.option('-cs', '--cnn_structure',
              default='gcl',
              type=str,
              help='Define structure of the convolution layer.'
              '2 or 3 - dimensions in 2D or 3D'
              'c - convolution'
              'g - group normalization'
              'b - batch normalization'
              'r - ReLU'
              'l - LeakyReLU',
              show_default=True)
@click.option('-ck', '--conv_kernel',
              default=3,
              type=int,
              help='Kernel size for 2D or 3D convolution.',
              show_default=True)
@click.option('-cp', '--conv_padding',
              default=1,
              type=int,
              help='Padding size for convolution.',
              show_default=True)
@click.option('-cmxk', '--pool_kernel',
              default=2,
              type=int,
              help='Maxpooling kernel.',
              show_default=True)
@click.option('-cch', '--checkpoints',
              default=(None, None),
              type=(str, str),
              help='Convolution multiplayer for CNN layers.',
              show_default=True)
@click.option('-dp', '--dropout',
              default=None,
              type=float,
              help='If not None, define dropout rate.',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 gpus'
              'cpu: Usa CPU'
              '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-th', '--threshold',
              default=0.5,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-tq', '--tqdm',
              default=True,
              type=bool,
              help='If True, build with progressbar.',
              show_default=True)
@click.version_option(version=version)
def main(prediction_dir: str,
         patch_size: int,
         cnn_type: tuple,
         cnn_out_channel: int,
         cnn_layers: int,
         conv_kernel: int,
         conv_padding: int,
         pool_kernel: int,
         cnn_multiplayer: int,
         cnn_structure: str,
         checkpoints: tuple,
         device: str,
         tqdm: bool,
         threshold: float,
         dropout: Optional[float] = None):
    """
    MAIN MODULE FOR PREDICTION WITH CNN UNET/RESUNET/UNET3PLUS MODELS

    Supported 3D images only!
    """
    """Searching for available images for prediction"""
    available_format = ('.tif', '.mrc', '.rec', '.am')
    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given directory!'

    if cnn_type == 'multi':
        cnn_type = ['unet', 'unet3plus']
    else:
        cnn_type = [cnn_type]

    stitcher = StitchImages(tqdm=tqdm)

    for i in predict_list:
        """Build temp dir"""
        build_temp_dir(dir=prediction_dir)

        """Voxalize image"""
        check_format = False

        if i.endswith('.tif'):
            image, _ = import_tiff(img=join(prediction_dir, i),
                                   dtype=np.uint8)
            format = 4
            check_format = True
        elif i.endswith(('.mrc', '.rec')):
            image, _ = import_mrc(img=join(prediction_dir, i))
            format = 4
            check_format = True
        elif i.endswith('.am'):
            image, _ = import_am(img=join(prediction_dir, i))
            format = 3
            check_format = True

        if not check_format:
            continue

        org_shape = image.shape

        trim_image(image=image,
                   trim_size_xy=patch_size,
                   trim_size_z=patch_size,
                   output=join(prediction_dir, 'temp', 'Patches'),
                   image_counter=0,
                   clean_empty=False,
                   prefix='')

        image = None
        del(image)

        """Predict image patches"""
        patches_DL = PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                       size=patch_size,
                                       out_channels=cnn_out_channel)

        predict(image_DL=patches_DL,
                output=join(prediction_dir, 'temp', 'Predictions'),
                cnn_type=cnn_type,
                cnn_in_channel=1,
                cnn_out_channel=cnn_out_channel,
                image_patch_size=patch_size,
                cnn_layers=cnn_layers,
                cnn_multiplayer=cnn_multiplayer,
                cnn_composition=cnn_structure,
                conv_kernel=conv_kernel,
                conv_padding=conv_padding,
                pool_kernel=pool_kernel,
                checkpoints=checkpoints,
                tqdm=tqdm,
                device=device,
                threshold=threshold,
                cnn_dropout=dropout)

        """Stitch patches from temp dir and save image"""
        tif.imsave(file=join(prediction_dir, 'Predictions', f'{i[:-format]}.tif'),
                   data=stitcher(image_dir=join(prediction_dir, 'temp', 'Predictions'),
                                 output=None,
                                 mask=True,
                                 prefix='',
                                 dtype=np.int8)[:org_shape[0], :org_shape[1], :org_shape[2]])

        """Clean-up temp dir"""
        clean_up(dir=prediction_dir)


if __name__ == '__main__':
    main()
