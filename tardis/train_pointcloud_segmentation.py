from os import listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

import click
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from tardis.sis_graphformer.graphformer.losses import (BCELoss, DiceLoss,
                                                       SigmoidFocalLoss)
from tardis.sis_graphformer.graphformer.network import CloudToGraph
from tardis.sis_graphformer.graphformer.trainer import Trainer
from tardis.sis_graphformer.utils.dataloader import GraphDataset
from tardis.sis_graphformer.utils.utils import (BuildTrainDataSet,
                                                cal_node_input)
from tardis.utils.device import get_device
from tardis.utils.utils import BuildTestDataSet, check_dir
from tardis.version import version


@click.command()
@click.version_option(version=version)
def main(pointcloud_dir: str,
         img_dir,
         patch_size,
         prefix,
         train_test_ratio: float,
         GF_out: int,
         GF_node_dim: int,
         GF_edge_dim: int,
         GF_layers: int,
         GF_heads: int,
         GF_dropout: float,
         GF_dist_sigma: int,
         DL_voxal_size: int,
         DL_drop_rate: float,
         DL_downsampling,
         DL_downsampling_rate,
         batch_size: int,
         GF_loss: str,
         loss_lr_rate: float,
         lr_rate_schedule: bool,
         GF_checkpoint,
         device: str,
         epoches: int,
         tqdm: bool,):
    """Check directory for data compatibility"""
    train_imgs_dir = join(pointcloud_dir, 'train', 'imgs')
    train_coords_dir = join(pointcloud_dir, 'train', 'coords')
    test_imgs_dir = join(pointcloud_dir, 'test', 'imgs')
    test_coords_dir = join(pointcloud_dir, 'test', 'coords')

    img_format = ['.tif', '.am', '.mrc', '.rec']
    coord_format = ['.CorrelationLines.am', '.npy', '.csv']
    dataset_test = False

    # Check if dir has train/test folder and if folder have data
    dataset_test = check_dir(dir=pointcloud_dir,
                             train_img=train_imgs_dir,
                             train_mask=train_coords_dir,
                             img_format=img_format,
                             test_img=test_imgs_dir,
                             test_mask=test_coords_dir,
                             mask_format=coord_format)

    """Set-up environment"""
    if not dataset_test:
        assert len([f for f in listdir(pointcloud_dir) if f.endswith(img_format)]) > 0, \
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
        mkdir(train_coords_dir)

    """Move data to setuped dir"""
    coord_format = BuildTrainDataSet(dir=pointcloud_dir,
                                     coord_format=coord_format,
                                     with_img=img_dir,
                                     img_format=img_format)
    BuildTestDataSet(dir=pointcloud_dir,
                     train_test_ration=train_test_ratio)

    """Build dataset for training/validation"""
    dl_train_graph = DataLoader(dataset=GraphDataset(coord_dir=train_coords_dir,
                                                     coord_format=coord_format[0][1:],
                                                     img_dir=train_imgs_dir,
                                                     prefix=prefix,
                                                     size=patch_size,
                                                     drop_rate=DL_drop_rate,
                                                     normalize="simple",
                                                     downsampling=DL_downsampling,
                                                     downsampling_rate=DL_downsampling_rate,
                                                     voxal_size=DL_voxal_size,
                                                     memory_save=False),
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True)

    dl_test_graph = DataLoader(dataset=GraphDataset(coord_dir=test_imgs_dir,
                                                    coord_format=coord_format,
                                                    img_dir=test_coords_dir,
                                                    prefix=prefix,
                                                    size=patch_size,
                                                    normalize="simple",
                                                    voxal_size=DL_voxal_size,
                                                    downsampling_if=DL_downsampling,
                                                    downsampling_rate=DL_downsampling_rate,
                                                    drop_rate=DL_drop_rate,
                                                    memory_save=False),
                               batch_size=1,
                               shuffle=False,
                               pin_memory=True)

    """Setup training"""
    if img_dir is None:
        train_with_images = False
    else:
        train_with_images = True

    device = get_device(device)
    model = CloudToGraph(n_out=GF_out,
                         node_input=cal_node_input(patch_size),
                         node_dim=GF_node_dim,
                         edge_dim=GF_edge_dim,
                         num_layers=GF_layers,
                         num_heads=GF_heads,
                         dropout_rate=GF_dropout,
                         coord_embed_sigma=GF_dist_sigma,
                         predict=False)

    coord, img, graph, _ = next(iter(dl_train_graph))
    print(f'cord = shape: {coord[0].shape}; '
          f'type: {coord[0].dtype}')
    print(f'img = shape: {img[0].shape}; '
          f'type: {img[0].dtype}')
    print(f'graph = shape: {graph[0].shape}; '
          f'class: {graph[0].unique()}; '
          f'type: {graph[0].dtype}')

    if GF_loss == "dice":
        loss_fn = DiceLoss()
    if GF_loss == "bce":
        loss_fn = BCELoss()
    if GF_loss == 'sfl':
        loss_fn = SigmoidFocalLoss()

    optimizer = optim.Adam(params=model.parameters(),
                           lr=loss_lr_rate)
    if GF_checkpoint is not None:
        save_train = join(GF_checkpoint)

        save_train = torch.load(join(save_train))
        model.load_state_dict(save_train['model_state_dict'])

        optimizer.load_state_dict(save_train['optimizer_state_dict'])

        save_train = None
        del(save_train)

    if lr_rate_schedule:
        learning_rate_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    else:
        learning_rate_scheduler = None

    print(f"Training is started on {device}, with: "
          f"Loss function = {GF_loss} "
          f"LR = {optimizer.param_groups[0]['lr']}, and "
          f"LRS = {learning_rate_scheduler}")

    print('The Network was build:')
    print(f"Network: Graphformer, "
          f"No. of Layers: {GF_layers} with {GF_heads} heads, "
          f"Each layer is build of {GF_node_dim} nodes, {GF_edge_dim} edges embedding, "
          f"Image patch size: {patch_size}, ")

    """Train"""
    train = Trainer(model=model.to(device),
                    node_input=train_with_images,
                    device=device,
                    batch=batch_size,
                    criterion=loss_fn,
                    optimizer=optimizer,
                    training_DataLoader=dl_train_graph,
                    validation_DataLoader=dl_test_graph,
                    validation_step=1,
                    epochs=epoches,
                    checkpoint_name='GF',
                    lr_scheduler=learning_rate_scheduler,
                    tqdm=tqdm)

    train.run_training()
