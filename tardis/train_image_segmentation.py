from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

import click
from torch.utils.data import DataLoader

from tardis.slcpy_data_processing.build_training_dataset import BuildTrainDataSet
from tardis.spindletorch.train import train
from tardis.spindletorch.utils.dataset_loader import VolumeDataset
from tardis.utils.device import get_device
from tardis.utils.utils import check_dir, BuildTestDataSet
from tardis.version import version


@click.command()
@click.option('-dir', '--training_dataset',
              default=getcwd(),
              type=str,
              help='Directory with train, test folder or folder with dataset '
              'to be used for training.',
              show_default=True)
@click.option('-ttr', '--train_test_ratio',
              default=10,
              type=float,
              help='Percentage value of train dataset that will become test.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=64,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-cnn', '--cnn_type',
              default='unet',
              type=click.Choice(['unet', 'resunet', 'unet3plus'],
                                case_sensitive=True),
              help='Type of NN used for training.',
              show_default=True)
@click.option('-co', '--cnn_out_channel',
              default=1,
              type=int,
              help='Number of output channels for the NN.',
              show_default=True)
@click.option('-b', '--taining_batch_size',
              default=25,
              type=int,
              help='Batch size.',
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
              'c - convolution'
              'g - group normalization'
              'b - batch normalization'
              'r - ReLU'
              'l - LeakyReLU',
              show_default=True)
@click.option('-l', '--cnn_loss',
              default='bce',
              type=click.Choice(['bce', 'dice', 'hybrid', 'adaptive_dice'],
                                case_sensitive=True),
              help='Loss function use for training.',
              show_default=True)
@click.option('-la', '--loss_alpha',
              default=None,
              type=float,
              help='Value of alpha used for adaptive dice loss.',
              show_default=True)
@click.option('-lr', '--loss_lr_rate',
              default=0.001,
              type=float,
              help='Learning rate for NN.',
              show_default=True)
@click.option('-lrs', '--lr_rate_schedule',
              default=False,
              type=bool,
              help='If True learning rate scheduler is used.',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 gpus'
              'cpu: Usa CPU'
              '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-e', '--epochs',
              default=100,
              type=int,
              help='Number of epoches.',
              show_default=True)
@click.option('-es', '--early_stop',
              default=10,
              type=int,
              help='Number of epoches without improvement after which early stop '
              'is initiated.',
              show_default=True)
@click.option('-cch', '--cnn_checkpoint',
              default=None,
              type=str,
              help='If indicated, dir to training checkpoint to reinitialized trainign.',
              show_default=True)
@click.option('-dr', '--dropout_rate',
              default=None,
              type=float,
              help='If indicated, value of dropout for CNN.',
              show_default=True)
@click.option('-tq', '--tqdm',
              default=True,
              type=bool,
              help='If True, build with progressbar.',
              show_default=True)
@click.version_option(version=version)
def main(training_dataset: str,
         train_test_ratio: float,
         patch_size: int,
         cnn_type: str,
         cnn_out_channel: int,
         taining_batch_size: int,
         cnn_layers: int,
         cnn_multiplayer: int,
         cnn_structure: str,
         cnn_loss: str,
         loss_lr_rate: float,
         lr_rate_schedule: bool,
         device: str,
         epochs: int,
         early_stop: int,
         tqdm: bool,
         loss_alpha: Optional[float] = None,
         cnn_checkpoint: Optional[str] = None,
         dropout_rate: Optional[float] = None):
    """
    MAIN MODULE FOR TRAINING CNN UNET/RESUNET/UNET3PLUS MODELS
    """
    """Set environment"""
    train_imgs_dir = join(training_dataset, 'train', 'imgs')
    train_masks_dir = join(training_dataset, 'train', 'masks')
    test_imgs_dir = join(training_dataset, 'test', 'imgs')
    test_masks_dir = join(training_dataset, 'test', 'masks')
    dataset_test = False
    img_format = ('.tif', '.am', '.rmc', '.rec')

    # Check if dir has train/test folder and if folder have data
    dataset_test = check_dir(dir=training_dataset,
                             with_img=True,
                             train_img=train_imgs_dir,
                             train_mask=train_masks_dir,
                             img_format='.tif',
                             test_img=test_imgs_dir,
                             test_mask=test_masks_dir,
                             mask_format='_mask.tif')

    # If any incompatibility and data exist, build dataset
    if not dataset_test:
        assert len([f for f in listdir(training_dataset) if f.endswith(img_format)]) > 0, \
            'Indicated folder for training do not have any compatible data or ' \
            'one of the following folders: '\
            'test/imgs; test/masks; train/imgs; train/masks'
        if isdir(join(training_dataset, 'train')):
            rmtree(join(training_dataset, 'train'))

        mkdir(join(training_dataset, 'train'))
        mkdir(train_imgs_dir)
        mkdir(train_masks_dir)

        if isdir(join(training_dataset, 'test')):
            rmtree(join(training_dataset, 'test'))

        mkdir(join(training_dataset, 'test'))
        mkdir(test_imgs_dir)
        mkdir(test_masks_dir)

        """Build train DataSets if they don't exist"""
        dataset_builder = BuildTrainDataSet(dataset_dir=training_dataset,
                                            circle_size=250,
                                            multi_layer=False,
                                            tqdm=True)
        dataset_builder.__builddataset__(trim_xy=patch_size,
                                         trim_z=patch_size)

        """Build test DataSets if they don't exist"""
        dataset_builder = BuildTestDataSet(dataset_dir=training_dataset,
                                           train_test_ration=train_test_ratio,
                                           prefix='_mask')
        dataset_builder.__builddataset__()

    """Build training and test dataset 2D/3D"""
    train_DL = DataLoader(dataset=VolumeDataset(img_dir=train_imgs_dir,
                                                mask_dir=train_masks_dir,
                                                size=patch_size,
                                                mask_suffix='_mask',
                                                normalize='simple',
                                                transform=True,
                                                out_channels=cnn_out_channel),
                          batch_size=taining_batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)

    test_DL = DataLoader(dataset=VolumeDataset(img_dir=test_imgs_dir,
                                               mask_dir=test_masks_dir,
                                               size=patch_size,
                                               mask_suffix='_mask',
                                               normalize='simple',
                                               transform=True,
                                               out_channels=cnn_out_channel),
                         batch_size=1,
                         shuffle=True,
                         num_workers=8,
                         pin_memory=True)

    """Get device"""
    device = get_device(device)

    """Run Training loop"""
    train(train_dataloader=train_DL,
          test_dataloader=test_DL,
          img_size=patch_size,
          cnn_type=cnn_type,
          classification=False,
          dropout=dropout_rate,
          convolution_layer=cnn_layers,
          convolution_multiplayer=cnn_multiplayer,
          convolution_structure=cnn_structure,
          cnn_checkpoint=cnn_checkpoint,
          loss_function=cnn_loss,
          loss_alpha=loss_alpha,
          learning_rate=loss_lr_rate,
          learning_rate_scheduler=lr_rate_schedule,
          early_stop_rate=early_stop,
          tqdm=tqdm,
          device=device,
          epochs=epochs)


if __name__ == '__main__':
    main()
