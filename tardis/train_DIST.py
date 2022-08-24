from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import click
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from tardis.dist_pytorch.transformer.losses import (BCELoss, DiceLoss,
                                                    SigmoidFocalLoss)
from tardis.dist_pytorch.transformer.network import C_DIST, DIST
from tardis.dist_pytorch.transformer.trainer import Trainer
from tardis.dist_pytorch.utils.dataloader import GraphDataset
from tardis.dist_pytorch.utils.utils import BuildTrainDataSet, cal_node_input
from tardis.utils.device import get_device
from tardis.utils.logo import Tardis_Logo
from tardis.utils.utils import BuildTestDataSet, check_dir
from tardis.version import version


@click.command()
@click.option('-dir', '--pointcloud_dir',
              default=getcwd(),
              type=str,
              help='Directory with train, test folder or folder with dataset '
              'to be used for training.',
              show_default=True)
@click.option('-wi', '--with_img',
              default=False,
              type=bool,
              help='Directory with train, test folder or folder with dataset ',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=None,
              type=int,
              help='If not None, patch size of images for GF',
              show_default=True)
@click.option('-pf', '--prefix',
              default=None,
              type=str,
              help='If not None, prefix name for coord data',
              show_default=True)
@click.option('-ttr', '--train_test_ratio',
              default=10,
              type=float,
              help='Percentage value of train dataset that will become test.',
              show_default=True)
@click.option('-td', '--train_dataset',
              default=True,
              type=bool,
              help='If True and .ply detected use dataloader for Scannet not partnet dataset',
              show_default=True)
@click.option('-go', '--gf_out',
              default=1,
              type=int,
              help='Number of output channels in GF.',
              show_default=True)
@click.option('-gn', '--gf_node_dim',
              default=None,
              type=int,
              help='Number embedding channels for nodes.',
              show_default=True)
@click.option('-ge', '--gf_edge_dim',
              default=128,
              type=int,
              help='Number embedding channels for edges.',
              show_default=True)
@click.option('-gl', '--gf_layers',
              default=6,
              type=int,
              help='Number of GF layers',
              show_default=True)
@click.option('-gh', '--gf_heads',
              default=8,
              type=int,
              help='Number of GF heads in MHA',
              show_default=True)
@click.option('-gd', '--gf_dropout',
              default=0,
              type=float,
              help='If 0, dropout is turn-off. Else indicate dropout rate',
              show_default=True)
@click.option('-gs', '--gf_sigma',
              default=2,
              type=float,
              help='Sigma value for distance embedding',
              show_default=True)
@click.option('-gst', '--gf_structure',
              default='triang',
              type=click.Choice(['full', 'full_af', 'self_attn', 'triang', 'dualtriang', 'quad']),
              help='Structure of the graphformer',
              show_default=True)
@click.option('-gt', '--gf_type',
              default='instance',
              type=click.Choice(['instance', 'semantic']),
              help='Structure of the graphformer',
              show_default=True)
@click.option('-gde', '--gf_dist',
              default=True,
              type=bool,
              help='If True use distance embedding as an input',
              show_default=True)
@click.option('-dds', '--dl_downsampling',
              default=500,
              type=int,
              help='Number of point threshold for voxal optimization',
              show_default=True)
@click.option('-ddsr', '--dl_downsampling_rate',
              default=5,
              type=float,
              help='Downsampling value for each point cloud',
              show_default=True)
@click.option('-glo', '--gf_loss',
              default='bce',
              type=click.Choice(['bce', 'dice', 'sfl']),
              help='Type of loss function use for training',
              show_default=True)
@click.option('-lr', '--loss_lr',
              default=0.001,
              type=float,
              help='Learning rate',
              show_default=True)
@click.option('-lrs', '--lr_rate_schedule',
              default=False,
              type=bool,
              help='If True, use learning rate scheduler [StepLR]',
              show_default=True)
@click.option('-gch', '--gf_checkpoint',
              default=None,
              type=str,
              help='If not None, directory to checkpoint',
              show_default=True)
@click.option('-d', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 gpus '
              'cpu: Usa CPU '
              '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-e', '--epochs',
              default=100,
              type=int,
              help='Number of epoches.',
              show_default=True)
@click.version_option(version=version)
def main(pointcloud_dir: str,
         with_img: bool,
         patch_size,
         prefix,
         train_test_ratio: float,
         train_dataset: str,
         gf_out: int,
         gf_node_dim: int,
         gf_edge_dim: int,
         gf_layers: int,
         gf_heads: int,
         gf_dropout: float,
         gf_sigma: int,
         dl_downsampling,
         dl_downsampling_rate,
         gf_loss: str,
         gf_structure: str,
         gf_type: str,
         gf_dist: bool,
         loss_lr: float,
         lr_rate_schedule: bool,
         gf_checkpoint,
         device: str,
         epochs: int):
    """
    MAIN MODULE FOR GRAPHFORMER TRAINING

    Training unit for DIST with 2D/3D dataset of point cloud with or without
    images.
    """
    tardis_logo = Tardis_Logo()
    tardis_logo(title='DIST training module')

    """Check directory for data compatibility"""
    train_imgs_dir = join(pointcloud_dir, 'train', 'imgs')
    train_coords_dir = join(pointcloud_dir, 'train', 'masks')
    test_imgs_dir = join(pointcloud_dir, 'test', 'imgs')
    test_coords_dir = join(pointcloud_dir, 'test', 'masks')

    img_format = ('.tif', '.tiff', '.am', '.mrc', '.rec')
    coord_format = ('.CorrelationLines.am', '.npy', '.csv', '.ply')
    dataset_test = False

    # Check if dir has train/test folder and if folder have data
    dataset_test = check_dir(dir=pointcloud_dir,
                             train_img=train_imgs_dir,
                             train_mask=train_coords_dir,
                             img_format=img_format,
                             test_img=test_imgs_dir,
                             test_mask=test_coords_dir,
                             mask_format=coord_format,
                             with_img=with_img)

    """Set-up environment"""
    if not dataset_test:
        assert len([f for f in listdir(pointcloud_dir) if f.endswith(coord_format)]) > 0, \
            'Indicated folder for training do not have any compatible data or ' \
            'one of the following folders: '\
            'test/imgs; test/masks; train/imgs; train/masks'

        if isdir(join(pointcloud_dir, 'train')):
            rmtree(join(pointcloud_dir, 'train'))
        mkdir(join(pointcloud_dir, 'train'))
        mkdir(train_imgs_dir)
        mkdir(train_coords_dir)

        if isdir(join(pointcloud_dir, 'test')):
            rmtree(join(pointcloud_dir, 'test'))
        mkdir(join(pointcloud_dir, 'test'))
        mkdir(test_imgs_dir)
        mkdir(test_coords_dir)

        """Move data to set-up dir"""
        _ = BuildTrainDataSet(dir=pointcloud_dir,
                              coord_format=coord_format,
                              with_img=with_img,
                              img_format=img_format)

        """Split train for train and test"""
        build_test = BuildTestDataSet(dataset_dir=pointcloud_dir,
                                      train_test_ration=train_test_ratio,
                                      prefix=prefix)
        build_test.__builddataset__()

    # else:
    mesh = [True for f in ['.ply'] if listdir(train_coords_dir)[0].endswith(f)]
    if len(mesh) > 0:
        mesh = True
    else:
        mesh = False

    if with_img:
        coord_format.append([f for f in img_format if listdir(train_imgs_dir)[0].endswith(f)][0])
    else:
        train_imgs_dir = None
        test_imgs_dir = None

    """Build dataset for training/validation"""
    dl_train_graph = DataLoader(dataset=GraphDataset(coord_dir=train_coords_dir,
                                                     coord_format=coord_format,
                                                     img_dir=train_imgs_dir,
                                                     prefix=prefix,
                                                     size=patch_size,
                                                     normalize="rescale",
                                                     mesh=mesh,
                                                     datatype=train_dataset,
                                                     downsampling_if=dl_downsampling,
                                                     downsampling_rate=dl_downsampling_rate,
                                                     memory_save=False),
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True)

    dl_test_graph = DataLoader(dataset=GraphDataset(coord_dir=test_coords_dir,
                                                    coord_format=coord_format,
                                                    img_dir=test_imgs_dir,
                                                    prefix=prefix,
                                                    size=patch_size,
                                                    normalize="rescale",
                                                    mesh=mesh,
                                                    datatype=train_dataset,
                                                    downsampling_if=dl_downsampling,
                                                    downsampling_rate=dl_downsampling_rate,
                                                    memory_save=False),
                               batch_size=1,
                               shuffle=False,
                               pin_memory=True)

    if train_dataset:
        coord, img, graph, _, _ = next(iter(dl_train_graph))
    else:
        coord, img, graph, _, _ = next(iter(dl_train_graph))

    print(f'cord = shape: {coord[0].shape}; '
          f'type: {coord[0].dtype}')
    print(f'img = shape: {img[0].shape}; '
          f'type: {img[0].dtype}')
    print(f'graph = shape: {graph[0].shape}; '
          f'class: {graph[0].unique()}; '
          f'type: {graph[0].dtype}')

    if patch_size is not None:
        if coord[0].shape[2] == 2:
            patch_size = (patch_size, patch_size)
        elif coord[0].shape[2] == 3:
            patch_size = (patch_size, patch_size, patch_size)

    """Setup training"""
    device = get_device(device)
    if gf_type == 'instance':
        model = DIST(n_out=gf_out,
                     node_input=cal_node_input(patch_size),
                     node_dim=gf_node_dim,
                     edge_dim=gf_edge_dim,
                     num_layers=gf_layers,
                     num_heads=gf_heads,
                     coord_embed_sigma=gf_sigma,
                     dropout_rate=gf_dropout,
                     structure=gf_structure,
                     dist_embed=gf_dist,
                     predict=False)
    elif gf_type == 'semantic':
        model = C_DIST(n_out=gf_out,
                       edge_dim=gf_edge_dim,
                       num_layers=gf_layers,
                       num_heads=gf_heads,
                       coord_embed_sigma=gf_sigma,
                       dropout_rate=gf_dropout,
                       structure=gf_structure,
                       dist_embed=gf_dist,
                       predict=False)

    if gf_loss == "dice":
        loss_fn = DiceLoss()
    if gf_loss == "bce":
        loss_fn = BCELoss()
    if gf_loss == 'sfl':
        loss_fn = SigmoidFocalLoss()

    """Checkpoint model and optimizer"""
    if gf_checkpoint is not None:
        save_train = join(gf_checkpoint)

        save_train = torch.load(join(save_train), map_location=device)
        model.to(device)
        model.load_state_dict(save_train['model_state_dict'])
    else:
        model.to(device)

    optimizer = optim.Adam(params=model.parameters(),
                           lr=loss_lr)
    if gf_checkpoint is not None:
        optimizer.load_state_dict(save_train['optimizer_state_dict'])

        save_train = None

    if lr_rate_schedule:
        learning_rate_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        learning_rate_scheduler = None

    print(f"Training is started on {device}, with: "
          f"Loss function = {gf_loss} "
          f"LR = {optimizer.param_groups[0]['lr']}, and "
          f"LRS = {learning_rate_scheduler}")

    print_setting = [f"Training is started on {device}",
                     f"Local dir: {getcwd()}",
                     f"Training for {gf_type} with No. of Layers: {gf_layers} with {gf_heads} heads",
                     "Layer are build of {} nodes, {} edges, {} max point per patch".format(gf_node_dim,
                                                                                            gf_edge_dim,
                                                                                            dl_downsampling)]
    """Train"""
    train = Trainer(model=model,
                    type=gf_type,
                    node_input=with_img,
                    device=device,
                    criterion=loss_fn,
                    optimizer=optimizer,
                    training_DataLoader=dl_train_graph,
                    validation_DataLoader=dl_test_graph,
                    epochs=epochs,
                    checkpoint_name='GF',
                    lr_scheduler=learning_rate_scheduler,
                    print_setting=print_setting)

    train.run_training()


if __name__ == '__main__':
    main()
