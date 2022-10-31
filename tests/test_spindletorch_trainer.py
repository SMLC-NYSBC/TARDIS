import os
import shutil

import torch.nn as nn
from tardis.spindletorch.trainer import CNNTrainer
from tardis.spindletorch.train import train_cnn
from tardis.spindletorch.datasets.dataloader import CNNDataset
from tardis.utils.device import get_device
from torch.utils.data import DataLoader


def test_init_spindletorch_train():
    dl = CNNTrainer(model=nn.Conv1d(1, 2, 3),
                    structure={'cnn_type': 'unet'},
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


def test_unet_trainer():
    structure = {'cnn_type': 'unet',
                 'classification': False,
                 'in_channel': 1,
                 'out_channel': 1,
                 'img_size': 64,
                 'dropout': None,
                 'num_conv_layers': 2,
                 'conv_scaler': 4,
                 'conv_kernel': 3,
                 'conv_padding': 1,
                 'maxpool_kernel': 2,
                 'layer_components': '3gcl',
                 'num_group': 8,
                 'prediction': False}

    train_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/train/imgs',
                          mask_dir='./tests/test_data/data_loader/cnn/train/masks',
                          size=structure['img_size'],
                          mask_suffix='_mask',
                          transform=True,
                          out_channels=structure['out_channel'])
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl,
                          batch_size=2,
                          shuffle=True,
                          pin_memory=True)

    test_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/test/imgs',
                         mask_dir='./tests/test_data/data_loader/cnn/test/masks',
                         size=structure['img_size'],
                         mask_suffix='_mask',
                         transform=True,
                         out_channels=structure['out_channel'])
    test_dl = DataLoader(dataset=test_dl,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True)

    train_cnn(train_dataloader=train_dl,
              test_dataloader=test_dl,
              model_structure=structure,
              checkpoint=None,
              loss_function='bce',
              learning_rate=0.001,
              learning_rate_scheduler=False,
              early_stop_rate=10,
              device=get_device('cpu'),
              epochs=2)

    assert os.listdir('./unet_checkpoint') == ['training_losses.csv', 'unet_checkpoint.pth',
                                               'model_weights.pth', 'validation_losses.csv',
                                               'eval_metric.csv']

    shutil.rmtree('./unet_checkpoint')


def test_fnet_trainer():
    structure = {'cnn_type': 'fnet',
                 'classification': False,
                 'in_channel': 1,
                 'out_channel': 1,
                 'img_size': 64,
                 'dropout': None,
                 'num_conv_layers': 2,
                 'conv_scaler': 4,
                 'conv_kernel': 3,
                 'conv_padding': 1,
                 'maxpool_kernel': 2,
                 'layer_components': '3gcl',
                 'num_group': 8,
                 'prediction': False}

    train_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/train/imgs',
                          mask_dir='./tests/test_data/data_loader/cnn/train/masks',
                          size=structure['img_size'],
                          mask_suffix='_mask',
                          transform=True,
                          out_channels=structure['out_channel'])
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl,
                          batch_size=1,
                          shuffle=True,
                          pin_memory=True)

    test_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/test/imgs',
                         mask_dir='./tests/test_data/data_loader/cnn/test/masks',
                         size=structure['img_size'],
                         mask_suffix='_mask',
                         transform=True,
                         out_channels=structure['out_channel'])
    test_dl = DataLoader(dataset=test_dl,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True)

    train_cnn(train_dataloader=train_dl,
              test_dataloader=test_dl,
              model_structure=structure,
              checkpoint=None,
              loss_function='bce',
              learning_rate=0.001,
              learning_rate_scheduler=False,
              early_stop_rate=10,
              device=get_device('cpu'),
              epochs=2)

    assert os.listdir('./fnet_checkpoint') == ['training_losses.csv', 'model_weights.pth',
                                               'validation_losses.csv', 'eval_metric.csv',
                                               'fnet_checkpoint.pth']

    shutil.rmtree('./fnet_checkpoint')


def test_unet3plus_trainer():
    structure = {'cnn_type': 'unet3plus',
                 'classification': False,
                 'in_channel': 1,
                 'out_channel': 1,
                 'img_size': 64,
                 'dropout': None,
                 'num_conv_layers': 2,
                 'conv_scaler': 4,
                 'conv_kernel': 3,
                 'conv_padding': 1,
                 'maxpool_kernel': 2,
                 'layer_components': '3gcl',
                 'num_group': 8,
                 'prediction': False}

    train_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/train/imgs',
                          mask_dir='./tests/test_data/data_loader/cnn/train/masks',
                          size=structure['img_size'],
                          mask_suffix='_mask',
                          transform=True,
                          out_channels=structure['out_channel'])
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl,
                          batch_size=1,
                          shuffle=True,
                          pin_memory=True)

    test_dl = CNNDataset(img_dir='./tests/test_data/data_loader/cnn/test/imgs',
                         mask_dir='./tests/test_data/data_loader/cnn/test/masks',
                         size=structure['img_size'],
                         mask_suffix='_mask',
                         transform=True,
                         out_channels=structure['out_channel'])
    test_dl = DataLoader(dataset=test_dl,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True)

    train_cnn(train_dataloader=train_dl,
              test_dataloader=test_dl,
              model_structure=structure,
              checkpoint=None,
              loss_function='bce',
              learning_rate=0.001,
              learning_rate_scheduler=False,
              early_stop_rate=10,
              device=get_device('cpu'),
              epochs=2)

    assert os.listdir('./unet3plus_checkpoint') == ['training_losses.csv', 'model_weights.pth',
                                                    'validation_losses.csv', 'eval_metric.csv',
                                                    'unet3plus_checkpoint.pth']

    shutil.rmtree('./unet3plus_checkpoint')
