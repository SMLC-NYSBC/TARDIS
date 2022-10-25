import os
import shutil

import torch.nn as nn
from tardis_dev.dist_pytorch.train import C_DistTrainer, DistTrainer
from tardis_dev.dist_pytorch.trainer import train_dist
from tardis_dev.dist_pytorch.datasets.dataloader import (FilamentDataset,
                                                      ScannetColorDataset)
from tardis_dev.utils.device import get_device
from torch.utils.data import DataLoader


def test_init_dist_train():
    dl = DistTrainer(model=nn.Conv1d(1, 2, 3),
                     structure={'node_input': 0},
                     device='cpu',
                     criterion=None,
                     optimizer=None,
                     print_setting=[],
                     training_DataLoader=None,
                     validation_DataLoader=None,
                     lr_scheduler=None,
                     epochs=100,
                     early_stop_rate=10,
                     checkpoint_name='test')

    dl = C_DistTrainer(model=nn.Conv1d(1, 2, 3),
                       structure={'node_input': 0,
                                  'dist_type': 'semantic'},
                       device='cpu',
                       criterion=None,
                       optimizer=None,
                       print_setting=[],
                       training_DataLoader=None,
                       validation_DataLoader=None,
                       lr_scheduler=None,
                       epochs=100,
                       early_stop_rate=10,
                       checkpoint_name='test')

    dl = C_DistTrainer(model=nn.Conv1d(1, 2, 3),
                       structure={'node_input': 0,
                                  'dist_type': 'instance'},
                       device='cpu',
                       criterion=None,
                       optimizer=None,
                       print_setting=[],
                       training_DataLoader=None,
                       validation_DataLoader=None,
                       lr_scheduler=None,
                       epochs=100,
                       early_stop_rate=10,
                       checkpoint_name='test')


def test_dist_trainer():
    """Test DIST trainer"""
    structure = {'dist_type': 'instance',
                 'n_out': 1,
                 'node_input': 0,
                 'node_dim': None,
                 'edge_dim': 4,
                 'num_layers': 1,
                 'num_heads': 1,
                 'coord_embed_sigma': 0.2,
                 'dropout_rate': 0.0,
                 'structure': 'triang'}

    train_dl = FilamentDataset(coord_dir='./tests/test_data/data_loader/filament_mt/train/masks/',
                               coord_format=('.CorrelationLines.am', '.csv'),
                               patch_if=50,
                               train=True)
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl,
                          batch_size=1,
                          shuffle=True,
                          pin_memory=True)

    test_dl = FilamentDataset(coord_dir='./tests/test_data/data_loader/filament_mt/test/masks/',
                              coord_format=('.CorrelationLines.am', '.csv'),
                              patch_if=50,
                              train=False)
    test_dl = DataLoader(dataset=test_dl,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True)

    train_dist(train_dataloader=train_dl,
               test_dataloader=test_dl,
               model_structure=structure,
               dist_checkpoint=None,
               loss_function='bce',
               learning_rate=0.001,
               learning_rate_scheduler=False,
               early_stop_rate=10,
               device=get_device('cpu'),
               epochs=2)

    assert os.listdir('./temp_test') == ['coords_0.npy', 'out_0.npy', 'df_0.npy',
                                         'graph_0.npy']
    shutil.rmtree('./temp_test')

    assert os.listdir('./temp_train') == ['coords_0.npy', 'out_0.npy', 'df_0.npy',
                                          'graph_0.npy']
    shutil.rmtree('./temp_train')

    assert os.listdir('./DIST_checkpoint') == ['training_losses.csv', 'model_weights.pth',
                                               'validation_losses.csv', 'eval_metric.csv',
                                               'DIST_checkpoint.pth']
    shutil.rmtree('./DIST_checkpoint')


def test_c_dist_trainer():
    """Test C_DIST trainer"""
    structure = {'dist_type': 'semantic',
                 'n_out': 1,
                 'node_input': 3,
                 'node_dim': 4,
                 'edge_dim': 4,
                 'num_cls': 200,
                 'num_layers': 1,
                 'num_heads': 1,
                 'coord_embed_sigma': 0.2,
                 'dropout_rate': 0.0,
                 'structure': 'triang'}

    train_dl = ScannetColorDataset(coord_dir='./tests/test_data/data_loader/scannet/train/masks/',
                                   coord_format=('.ply'),
                                   patch_if=50,
                                   train=True)
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl,
                          batch_size=1,
                          shuffle=True,
                          pin_memory=True)

    test_dl = ScannetColorDataset(coord_dir='./tests/test_data/data_loader/scannet/test/masks/',
                                  coord_format=('.ply'),
                                  patch_if=50,
                                  train=False)
    test_dl = DataLoader(dataset=test_dl,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True)

    train_dist(train_dataloader=train_dl,
               test_dataloader=test_dl,
               model_structure=structure,
               dist_checkpoint=None,
               loss_function='bce',
               learning_rate=0.001,
               learning_rate_scheduler=False,
               early_stop_rate=10,
               device=get_device('cpu'),
               epochs=2)

    assert os.listdir('./temp_test') == ['coords_0.npy', 'rgb_0.npy', 'out_0.npy', 'cls_0.npy', 'graph_0.npy']
    shutil.rmtree('./temp_test')

    assert os.listdir('./temp_train') == ['coords_0.npy', 'rgb_0.npy', 'out_0.npy', 'cls_0.npy', 'graph_0.npy']
    shutil.rmtree('./temp_train')

    assert os.listdir('./DIST_checkpoint') == ['training_losses.csv', 'model_weights.pth',
                                               'validation_losses.csv', 'eval_metric.csv',
                                               'DIST_checkpoint.pth']
    shutil.rmtree('./DIST_checkpoint')
