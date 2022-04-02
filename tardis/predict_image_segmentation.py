from os import getcwd, listdir
from os.path import join
from shutil import rmtree
from typing import Optional

import click
import numpy as np
import tifffile.tifffile as tif
from torch.utils.data import DataLoader

from tardis.slcpy_data_processing.utils.load_data import (import_am,
                                                          import_mrc,
                                                          import_tiff)
from tardis.slcpy_data_processing.utils.trim import trim_image
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
              default=getcwd(),
              type=str,
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
              default=5,
              type=int,
              help='Convolution multiplayer for CNN layers.',
              show_default=True)
@click.option('-cs', '--cnn_structure',
              default='gcl',
              type=str,
              help='Define structure of the convolution layer.'
              'c - convolution'
              'g - group normalization'
              'b - batch normalization'
              'r - ReLU'
              'l - LeakyReLU',
              show_default=True)
@click.option('-dp', '--dropout',
              default=None,
              type=float,
              help='If not None, define dropout rate.',
              show_default=True)
@click.option('-th', '--threshold',
              default=None,
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
         cnn_multiplayer: int,
         cnn_structure: str,
         tqdm: bool,
         threshold: Optional[float] = None,
         dropout: Optional[float] = None):
    """
    MAIN MODULE FOR PREDICTION WITH CNN UNET/RESUNET/UNET3PLUS MODELS
    """
    """Searching for available images for prediction"""
    available_format = ['.tif', '.mrc', '.rec', '.am']
    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    for i in predict_list:
        """Build temp dir"""
        build_temp_dir(dir=prediction_dir)

        """Voxalize image"""
        if i.endswith('.tif'):
            image = import_tiff(img=join(prediction_dir, i),
                                dtype=np.uint8)
        elif i.endswith(['.mrc', '.rec']):
            image, _ = import_mrc(img=join(prediction_dir, i))
        elif i.endswith('.am'):
            image, _ = import_am(img=join(prediction_dir, i))

        trim_image(image=image,
                   trim_size_xy=patch_size,
                   trim_size_z=patch_size,
                   output=join(prediction_dir, 'temp', 'Patches'),
                   image_counter=0,
                   clean_empty=False,
                   prefix='')

        """Predict image patches"""
        patches_DL = DataLoader(dataset=PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                                          size=patch_size,
                                                          out_channels=cnn_out_channel),
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=5)

        predict(image_DL=patches_DL,
                output=join(prediction_dir, 'temp', 'Prediction'),
                cnn_type=cnn_type,
                cnn_in_channel=1,
                cnn_out_channel=cnn_out_channel,
                image_patch_size=patch_size,
                cnn_layers=cnn_layers,
                cnn_multiplayer=cnn_multiplayer,
                cnn_composition=cnn_structure,
                tqdm=tqdm,
                threshold=threshold,
                cnn_dropout=dropout)

        """Stitch patches from temp dir"""
        stitched_image = []

        """Save point cloud, (Optional) image"""

        tif.imsave(file=join(prediction_dir, 'Prediction', i),
                   data=np.array(stitched_image, dtype=np.int8))

        """Clean-up temp dir"""
        rmtree(join(prediction_dir, 'temp'))
        clean_up(dir=prediction_dir)


if __name__ == '__main__':
    main()
