#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import sys
from os import getcwd
from typing import Optional

import torch
from torch import optim

from tardis_em.cnn.cnn import build_cnn_network
from tardis_em.cnn.trainer import CNNTrainer
from tardis_em.cnn.utils.utils import check_model_dict
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.losses import *
from tardis_em.utils.trainer import ISR_LR

# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train_cnn(
    train_dataloader,
    test_dataloader,
    model_structure: dict,
    checkpoint: Optional[str] = None,
    loss_function="bce",
    learning_rate=1,
    learning_rate_scheduler=False,
    early_stop_rate=10,
    device="gpu",
    warmup=100,
    epochs=1000,
):
    """
    Wrapper for CNN models.

    Args:
        train_dataloader (torch.DataLoader): DataLoader with train dataset.
        test_dataloader (torch.DataLoader): DataLoader with test dataset.
        model_structure (dict): Dictionary with model setting.
        checkpoint (None, optional): Optional, CNN model checkpoint.
        loss_function (str): Type of loss function.
        learning_rate (float): Learning rate.
        learning_rate_scheduler (bool): If True, LR_scheduler is used with training.
        early_stop_rate (int): Define max. number of epoch's without improvements
        after which training is stopped.
        device (torch.device): Device on which model is trained.
        warmup (int): Number of warm-up steps.
        epochs (int): Max number of epoch's.
    """
    """Losses"""
    loss_functions = [
        AdaptiveDiceLoss,
        BCELoss,
        WBCELoss,
        BCEDiceLoss,
        CELoss,
        DiceLoss,
        ClDiceLoss,
        ClBCELoss,
        SigmoidFocalLoss,
        LaplacianEigenmapsLoss,
        BCEMSELoss,
    ]
    losses_f = {f.__name__: f() for f in loss_functions}

    """Check input variable"""
    if checkpoint is not None:
        save_train = torch.load(checkpoint, map_location=device)
        if "model_struct_dict" in save_train.keys():
            model_structure = save_train["model_struct_dict"]

    model_structure = check_model_dict(model_structure)
    if not isinstance(device, torch.device) and isinstance(device, str):
        device = get_device(device)

    """Build DIST model"""
    try:
        model = build_cnn_network(
            network_type=model_structure["cnn_type"],
            structure=model_structure,
            img_size=model_structure["img_size"],
            prediction=False,
        )
    except:
        TardisError(
            "14",
            "tardis_em/cnn/train.py",
            f"CNNModelError: Model type: {type} was not build correctly!",
        )
        sys.exit()

    """Build TARDIS progress bar output"""
    print_setting = [
        f"Training is started for {model_structure['cnn_type']}:",
        f"Local dir: {getcwd()}",
        f"Training for {model_structure['cnn_type']} with "
        f"No. of Layers: {model_structure['num_conv_layers']} with "
        f"{model_structure['in_channel']} input and "
        f"{model_structure['out_channel']} output channel",
        f"Layers are build of {model_structure['layer_components']} modules, "
        f"train on {model_structure['img_size']} pixel images, "
        f"with {model_structure['conv_scaler']} up/down sampling "
        "channel scaler.",
    ]

    """Optionally: Load checkpoint for retraining"""
    if checkpoint is not None:
        model.load_state_dict(save_train["model_state_dict"])
    model = model.to(device)

    """Define loss function for training"""
    loss_fn = losses_f["BCELoss"]
    if loss_function in losses_f:
        loss_fn = losses_f[loss_function]

    """Build training optimizer"""
    if learning_rate_scheduler:
        optimizer = optim.NAdam(
            params=model.parameters(),
            betas=(0.9, 0.999),
            lr=learning_rate,
            eps=1e-8,
            momentum_decay=4e-3,
        )
    else:
        optimizer = optim.NAdam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            momentum_decay=4e-3,
        )

    """Optionally: Build learning rate scheduler"""
    if learning_rate_scheduler:
        optimizer = ISR_LR(
            optimizer, lr_mul=learning_rate, warmup_steps=warmup, scale=1
        )

    """Optionally: Checkpoint model"""
    if checkpoint is not None:
        optimizer.load_state_dict(save_train["optimizer_state_dict"])
        del save_train

    """Build trainer"""
    train = CNNTrainer(
        model=model,
        structure=model_structure,
        device=device,
        criterion=loss_fn,
        optimizer=optimizer,
        print_setting=print_setting,
        training_DataLoader=train_dataloader,
        validation_DataLoader=test_dataloader,
        lr_scheduler=learning_rate_scheduler,
        epochs=epochs,
        early_stop_rate=early_stop_rate,
        checkpoint_name=model_structure["cnn_type"],
    )

    """Train"""
    train.run_trainer()
