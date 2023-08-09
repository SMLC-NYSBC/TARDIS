#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import os
import shutil

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tardis_pytorch.spindletorch.datasets.dataloader import CNNDataset
from tardis_pytorch.spindletorch.train import train_cnn
from tardis_pytorch.spindletorch.trainer import CNNTrainer
from tardis_pytorch.utils.device import get_device


def test_init_spindletorch_train():
    CNNTrainer(
        model=nn.Conv1d(1, 2, 3),
        structure={"cnn_type": "unet"},
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        print_setting=[],
        training_DataLoader=None,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        checkpoint_name="test",
    )


def test_unet_trainer():
    structure = {
        "cnn_type": "unet",
        "classification": False,
        "in_channel": 1,
        "out_channel": 1,
        "img_size": 64,
        "dropout": None,
        "num_conv_layers": 2,
        "conv_scaler": 4,
        "conv_kernel": 3,
        "conv_padding": 1,
        "maxpool_kernel": 2,
        "layer_components": "3gcl",
        "num_group": 8,
        "prediction": False,
    }

    train_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/train/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/train/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl, batch_size=2, shuffle=True, pin_memory=True)

    test_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/test/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/test/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    test_dl = DataLoader(dataset=test_dl, shuffle=True, pin_memory=True)

    train_cnn(
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        model_structure=structure,
        learning_rate=0.01,
        device=get_device("cpu"),
        epochs=2,
    )

    assert len(os.listdir("./unet_checkpoint")) in [5, 6]

    shutil.rmtree("./unet_checkpoint")


def test_fnet_trainer():
    structure = {
        "cnn_type": "fnet",
        "classification": False,
        "in_channel": 1,
        "out_channel": 1,
        "img_size": 64,
        "dropout": None,
        "num_conv_layers": 2,
        "conv_scaler": 4,
        "conv_kernel": 3,
        "conv_padding": 1,
        "maxpool_kernel": 2,
        "layer_components": "3gcl",
        "num_group": 8,
        "prediction": False,
    }

    train_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/train/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/train/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl, shuffle=True, pin_memory=True)

    test_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/test/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/test/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    test_dl = DataLoader(dataset=test_dl, shuffle=True, pin_memory=True)

    train_cnn(
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        model_structure=structure,
        learning_rate=0.01,
        device=get_device("cpu"),
        epochs=2,
    )

    assert len(os.listdir("./fnet_checkpoint")) in [5, 6]

    shutil.rmtree("./fnet_checkpoint")


def test_unet3plus_trainer():
    structure = {
        "cnn_type": "unet3plus",
        "classification": False,
        "in_channel": 1,
        "out_channel": 1,
        "img_size": 64,
        "dropout": None,
        "num_conv_layers": 2,
        "conv_scaler": 4,
        "conv_kernel": 3,
        "conv_padding": 1,
        "maxpool_kernel": 2,
        "layer_components": "3gcl",
        "num_group": 8,
        "prediction": False,
    }

    train_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/train/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/train/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    train_dl = train_dl + train_dl + train_dl + train_dl
    train_dl = DataLoader(dataset=train_dl, shuffle=True, pin_memory=True)

    test_dl = CNNDataset(
        img_dir="./tests/test_data/data_loader/cnn/test/imgs",
        mask_dir="./tests/test_data/data_loader/cnn/test/masks",
        size=structure["img_size"],
        out_channels=structure["out_channel"],
    )
    test_dl = DataLoader(dataset=test_dl, shuffle=True, pin_memory=True)

    train_cnn(
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        learning_rate=0.01,
        model_structure=structure,
        device=get_device("cpu"),
        epochs=2,
    )

    assert len(os.listdir("./unet3plus_checkpoint")) in [5, 6]

    shutil.rmtree("./unet3plus_checkpoint")
