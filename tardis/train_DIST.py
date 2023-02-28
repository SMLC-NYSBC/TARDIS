#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import sys
from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import click
import torch

from tardis.dist_pytorch.datasets.dataloader import build_dataset
from tardis.dist_pytorch.train import train_dist
from tardis.utils.dataset import build_test_dataset, move_train_dataset
from tardis.utils.device import get_device
from tardis.utils.errors import TardisError
from tardis.utils.logo import TardisLogo
from tardis.utils.setup_envir import check_dir
from tardis.version import version


@click.command()
@click.option('-dir', '--dir',
              default=getcwd(),
              type=str,
              help='Directory with train, test folder or folder with dataset '
                   'to be used for training.',
              show_default=True)
@click.option('-dt', '--dataset_type',
              default='filament',
              type=str,
              help='Define training dataset type.',
              show_default=True)
@click.option('-no', '--n_out',
              default=1,
              type=int,
              help='Number of output channels in DIST.',
              show_default=True)
@click.option('-nd', '--node_dim',
              default=0,
              type=int,
              help='Number embedding channels for nodes.',
              show_default=True)
@click.option('-ed', '--edge_dim',
              default=128,
              type=int,
              help='Number embedding channels for edges.',
              show_default=True)
@click.option('-ly', '--layers',
              default=6,
              type=int,
              help='Number of DIST layers',
              show_default=True)
@click.option('-hd', '--heads',
              default=8,
              type=int,
              help='Number of DIST heads in MHA',
              show_default=True)
@click.option('-dp', '--dropout',
              default=0,
              type=float,
              help='If 0, dropout is turn-off. Else indicate dropout rate.',
              show_default=True)
@click.option('-sg', '--sigma',
              default=2,
              type=float,
              help='Sigma value for distance embedding.',
              show_default=True)
@click.option('-st', '--structure',
              default='triang',
              type=click.Choice(['full', 'full_af',
                                 'self_attn',
                                 'triang', 'dualtriang', 'quad']),
              help='Structure of the DIST layers.',
              show_default=True)
@click.option('-ds', '--dist_structure',
              default='instance',
              type=click.Choice(['instance', 'semantic']),
              help='Type of DIST model prediction.',
              show_default=True)
@click.option('-ps', '--pc_sampling',
              default=500,
              type=int,
              help='Max number of points per patch.',
              show_default=True)
@click.option('-lo', '--loss',
              default='BCELoss',
              type=click.Choice(['BCELoss', 'DiceLoss', 'SigmoidFocalLoss']),
              help='Type of loss function use for training.',
              show_default=True)
@click.option('-lr', '--loss_lr',
              default=0.001,
              type=float,
              help='Learning rate.',
              show_default=True)
@click.option('-ch', '--checkpoint',
              default=None,
              type=str,
              help='If not None, directory to checkpoint.',
              show_default=True)
@click.option('-dv', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
                   'gpu: Use ID 0 gpus '
                   'cpu: Usa CPU '
                   '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-ep', '--epochs',
              default=100,
              type=int,
              help='Number of epoches.',
              show_default=True)
@click.option('-er', '--early_stop',
              default=10,
              type=int,
              help="Number or epoch's without improvement, "
                   'after which training is stopped.',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         dataset_type: str,
         n_out: int,
         node_dim: int,
         edge_dim: int,
         layers: int,
         heads: int,
         dropout: float,
         sigma: int,
         structure: str,
         dist_structure: str,
         pc_sampling: int,
         loss: str,
         loss_lr: float,
         lr_rate_schedule: bool,
         checkpoint,
         device: str,
         epochs: int,
         early_stop: int):
    """Initialize TARDIS progress bar"""
    tardis_logo = TardisLogo()
    tardis_logo(title='DIST training module')

    """Stor all temp. directories"""
    TRAIN_IMAGE_DIR = join(dir, 'train', 'imgs')
    TRAIN_COORD_DIR = join(dir, 'train', 'masks')
    TEST_IMAGE_DIR = join(dir, 'test', 'imgs')
    TEST_COORD_DIR = join(dir, 'test', 'masks')

    COORD_FORMAT = '.txt'
    if dataset_type not in ['stanford', 'stanford_color']:
        COORD_FORMAT = ('.CorrelationLines.am', '.npy', '.csv', '.ply')

    """Check if dir has train/test folder and if f  older have compatible data"""
    DATASET_TEST = check_dir(dir=dir,
                             train_img=TRAIN_IMAGE_DIR,
                             train_mask=TRAIN_COORD_DIR,
                             img_format=(),
                             test_img=TEST_IMAGE_DIR,
                             test_mask=TEST_COORD_DIR,
                             mask_format=COORD_FORMAT,
                             with_img=False)

    """Optionally: Set-up environment if not existing"""
    if not DATASET_TEST:
        # Check and set-up environment
        if not len([f for f in listdir(dir) if f.endswith(COORD_FORMAT)]) > 0:
            if not dataset_type == 'stanford':
                TardisError('12',
                            'tardis/train_DIST.py',
                            'Indicated folder for training do not have any compatible '
                            'data or one of the following folders: '
                            'test/imgs; test/masks; train/imgs; train/masks')

        if isdir(join(dir, 'train')):
            rmtree(join(dir, 'train'))
        mkdir(join(dir, 'train'))
        mkdir(TRAIN_IMAGE_DIR)
        mkdir(TRAIN_COORD_DIR)

        if isdir(join(dir, 'test')):
            rmtree(join(dir, 'test'))
        mkdir(join(dir, 'test'))
        mkdir(TEST_IMAGE_DIR)
        mkdir(TEST_COORD_DIR)

        # Build train and test dataset
        move_train_dataset(dir=dir, coord_format=COORD_FORMAT, with_img=False)

        no_dataset = int(len([f for f in listdir(dir) if f.endswith(COORD_FORMAT)]) / 2)
        if dataset_type in ['stanford', 'stanford_color']:
            build_test_dataset(dataset_dir=dir, dataset_no=no_dataset, stanford=True)
        else:
            build_test_dataset(dataset_dir=dir, dataset_no=no_dataset)

    """Pre-setting for building DataLoader"""
    # Check for general dataset
    if dataset_type == 'general':
        tardis_logo(text_1=f'General DataSet loader is not supported in TARDIS {version}')
        sys.exit()

    """Build DataLoader for training/validation"""
    dl_train_graph, dl_test_graph = build_dataset(dataset_type=dataset_type,
                                                  dirs=[TRAIN_COORD_DIR, TEST_COORD_DIR],
                                                  max_points_per_patch=pc_sampling)

    """Setup training"""
    device = get_device(device)

    if node_dim > 0:
        node_input = 3
    else:
        node_input = 0

    if dist_structure == 'instance':
        num_cls = None
    elif dist_structure == 'semantic':
        num_cls = 200
    else:
        tardis_logo(text_1=f'ValueError: Wrong DIST type {dist_structure}!')
        sys.exit()

    """Optionally: pre-load model structure from checkpoint"""
    if checkpoint is not None:
        save_train = torch.load(join(checkpoint), map_location=device)

        if 'model_struct_dict' in save_train.keys():
            model_dict = save_train['model_struct_dict']
            globals().update(model_dict)

    model_dict = {'dist_type': dist_structure,
                  'n_out': n_out,
                  'node_input': node_input,
                  'node_dim': node_dim,
                  'edge_dim': edge_dim,
                  'num_cls': num_cls,
                  'num_layers': layers,
                  'num_heads': heads,
                  'coord_embed_sigma': sigma,
                  'dropout_rate': dropout,
                  'structure': structure}

    train_dist(train_dataloader=dl_train_graph,
               test_dataloader=dl_test_graph,
               model_structure=model_dict,
               checkpoint=checkpoint,
               loss_function=loss,
               learning_rate=loss_lr,
               early_stop_rate=early_stop,
               device=device,
               epochs=epochs)


if __name__ == '__main__':
    main()
