from os import listdir, getcwd
from os.path import join
from shutil import rmtree
import numpy as np
import click

from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.load_data import import_tiff, import_mrc, \
    import_am
from tardis.slcpy_data_processing.utils.trim import trim_image
from tardis.slcpy_data_processing.utils.stitch import StitchImages
from tardis.spindletorch.predict import predict
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.version import version

from tardis.utils.setup_envir import build_temp_dir, clean_up
from torch.utils.data import DataLoader

from tardis.utils.utils import check_uint8


@click.command()
@click.option('-dir', '--prediction_dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=96,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.3,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-ch', '--checkpoints',
              default=(None, None),
              type=tuple,
              help='If not None, str for Unet and Unet3Plus checkpoints',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 GPUs '
              'cpu: Usa CPU '
              '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-tq', '--tqdm',
              default=True,
              type=bool,
              help='If True, build with progressbar.',
              show_default=True)
@click.version_option(version=version)
def main(prediction_dir: str,
         patch_size: int,
         cnn_threshold: float,
         checkpoints: tuple,
         device: str,
         tqdm: bool):
    """
    MAIN MODULE FOR PREDICTION MT with Tardis
    """
    """Searching for available images for prediction"""
    available_format = ('.tif', '.mrc', '.rec', '.am')
    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    stitcher = StitchImages(tqdm=tqdm)
    post_processer = ImageToPointCloud(tqdm=True)

    if tqdm:
        from tqdm import tqdm as tq

        batch_iter = tq(predict_list)
    else:
        batch_iter = predict_list
        
    for i in batch_iter:
        batch_iter.set_description(f'Predicting image {i} ...')
        """Build temp dir"""
        build_temp_dir(dir=prediction_dir)

        """Voxalize image"""
        if i.endswith('.tif'):
            image, _ = import_tiff(img=join(prediction_dir, i),
                                   dtype=np.uint8)
        elif i.endswith(('.mrc', '.rec')):
            image, _ = import_mrc(img=join(prediction_dir, i))
        elif i.endswith('.am'):
            if i.endswith('CorrelationLines.am'):
                continue

            image, _ = import_am(img=join(prediction_dir, i))

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
                                       out_channels=1)

        predict(image_DL=patches_DL,
                output=join(prediction_dir, 'temp', 'Predictions'),
                cnn_type=['unet', 'unet3plus'],
                cnn_in_channel=1,
                cnn_out_channel=1,
                image_patch_size=patch_size,
                cnn_layers=5,
                cnn_multiplayer=32,
                cnn_composition='gcl',
                device=device,
                tqdm=tqdm,
                checkpoints=checkpoints,
                threshold=cnn_threshold,
                cnn_dropout=None)

        """Stitch patches and post-process"""
        image = check_uint8(stitcher(image_dir=join(prediction_dir, 'temp', 'Predictions'),
                                     output=None,
                                     mask=True,
                                     prefix='',
                                     dtype=np.int8)[:org_shape[0],
                                                    :org_shape[1],
                                                    :org_shape[2]])

        if image is None:
            continue
        point_cloud_hd = post_processer(image=image,
                                        euclidean_transform=True,
                                        label_size=300,
                                        down_sampling_voxal_size=None)
        print(point_cloud_hd.shape)
        """Build Graphformer and predict point cloud"""

        """Graph representation to segmented point cloud"""

        """Save as .am"""

        """Clean-up temp dir"""
        rmtree(join(prediction_dir, 'temp'))
        clean_up(dir=prediction_dir)
